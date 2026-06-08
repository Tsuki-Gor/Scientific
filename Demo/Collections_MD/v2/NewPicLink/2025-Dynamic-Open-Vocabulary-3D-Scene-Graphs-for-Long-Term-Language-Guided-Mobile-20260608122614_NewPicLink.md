# Dynamic Open-Vocabulary 3D Scene Graphs for Long-Term Language-Guided Mobile Manipulation
# 面向长期语言引导的移动操作的动态开放词汇三维场景图


Zhijie Yan ${}^{\text{ ⓳ }}$ , Shufei Li ${}^{\circledR }$ , Zuoxu Wang ${}^{\circledR }$ , Lixiu Wu ${}^{\circledR }$ , Han Wang ${}^{\circledR }$ , Jun Zhu ${}^{\circledR }$ , Lijiang Chen ${}^{\circledR }$ , and Jihong Liu ${}^{\circledR }$
闫智杰 ${}^{\text{ ⓳ }}$，李舒菲 ${}^{\circledR }$，王左旭 ${}^{\circledR }$，吴丽秀 ${}^{\circledR }$，王涵 ${}^{\circledR }$，朱军 ${}^{\circledR }$，陈丽江 ${}^{\circledR }$，以及刘纪鸿 ${}^{\circledR }$


Abstract-Enabling mobile robots to perform long-term tasks in dynamic real-world environments is a formidable challenge, especially when the environment changes frequently due to human-robot interactions or the robot's own actions. Traditional methods typically assume static scenes, which limits their applicability in the continuously changing real world. To overcome these limitations, we present DoVSG, a novel mobile manipulation framework that leverages dynamic open-vocabulary 3D scene graphs and a language-guided task planning module for long-term task execution. DovSG takes RGB-D sequences as input and utilizes vision-language models (VLMs) for object detection to obtain high-level object semantic features. Based on the segmented objects, a structured 3D scene graph is generated for low-level spatial relationships. Furthermore, an efficient mechanism for locally updating the scene graph, allows the robot to adjust parts of the graph dynamically during interactions without the need for full scene reconstruction. This mechanism is particularly valuable in dynamic environments, enabling the robot to continually adapt to scene changes and effectively support the execution of long-term tasks. We validated our system in real-world environments with varying degrees of manual modifications, demonstrating its effectiveness and superior performance in long-term tasks.
使移动机器人在动态的真实环境中执行长期任务是一项艰巨挑战，尤其是在因人机交互或机器人自身动作而导致环境频繁变化时。传统方法通常假设场景静态，从而限制了其在不断变化的真实世界中的适用性。为克服这些局限，我们提出DoVSG，这是一种新颖的移动操作框架，它结合动态开放词汇三维场景图与面向语言的任务规划模块，用于长期任务执行。DovSG以RGB-D序列作为输入，并利用视觉-语言模型（VLM）进行目标检测，以获得高级目标语义特征。基于分割后的目标，会生成结构化的三维场景图，用于刻画低层次的空间关系。此外，提供一种高效的局部更新机制，使机器人在交互过程中可动态调整场景图的部分内容，而无需重建整个场景图。该机制在动态环境中特别有价值，能够让机器人持续适应场景变化，并有效支持长期任务的执行。我们在真实环境中对系统进行了验证，环境的人工改动程度不一，结果表明其有效且在长期任务上表现更优。


Index Terms-3D scene graph, long-term tasks, mobile manipulation, open vocabulary.
关键词-三维场景图，长期任务，移动操作，开放词汇。


## I. INTRODUCTION
## I. INTRODUCTION


MOBILE manipulation in real-world environments is increasingly expected to perform complex, long-term tasks that require adaptability and resilience to changes. These dynamic environments are characterized by frequent alterations caused by human activities, robot interactions, or the inherent variability of the surroundings. Traditional robotic systems often fall short in such settings because they typically rely on the assumption of a static or minimally changing environment [1], [2], [3], [4], [5]. This limitation restricts their applicability in real-world scenarios where adaptability is crucial.
在真实环境中进行移动式操作（MOBILE manipulation）正日益被期望能够执行复杂且长期的任务，这类任务需要具备对变化的适应性和韧性。这些动态环境的特征在于会因人类活动、机器人交互，或周围环境固有的波动而频繁发生改变。传统的机器人系统在此类场景下往往难以胜任，因为它们通常建立在静态或仅有极小变化的环境这一假设之上 [1]，[2]，[3]，[4]，[5]。这一局限性限制了其在适应性至关重要的真实场景中的应用。


In this work, we enhance robotic capabilities by introducing a novel and practical robotic framework, the DovSG system. This framework comprises five key modules: perception, memory, task planning, navigation, and manipulation, as illustrated in Fig. 1. To address the challenge of scene perception, our perception module integrates advanced tools such as Recognize-Anything [6], Grounding DINO [7], Segment Anything-2 [8], and CLIP [9] to detect objects or components and extract their semantic features. In the memory module, each object within the scene is represented as a node characterized by geometric and semantic features, with the relationships between objects encoded in the graph's edges. The module continuously updates object features and scene graphs by locally refining the areas where the robot interacts, preserving knowledge to support ongoing exploration and exploitation, while avoiding the need to reconstruct the entire scene and enabling more efficient adaptation to dynamic environments. Our task planning module uses the advanced large language models to decompose tasks into manageable subtasks. Then, the navigation and manipulation module are activated to execute the planned actions. In experiments, we deployed a mobile robot platform in multiple real-world indoor environments, conducting long-term task trials under varying degrees of manual environmental modifications. We demonstrated that DovSG can accurately update the 3D scene graph and perform excellently in long-term tasks and subtasks such as pickup, place, and navigation. Our contributions are as follows:
在本工作中，我们通过引入一种新颖且实用的机器人框架——DovSG 系统，来增强机器人的能力。该框架包含五个关键模块：感知（perception）、记忆（memory）、任务规划（task planning）、导航（navigation）和操作（manipulation），如图 1 所示。为应对场景感知挑战，我们的感知模块集成了 Recognize-Anything [6]、Grounding DINO [7]、Segment Anything-2 [8] 和 CLIP [9] 等先进工具，用于检测物体或部件并提取其语义特征。在记忆模块中，场景内的每个物体都被表示为一个节点，其特征由几何与语义信息构成，物体之间的关系通过图的边进行编码。该模块通过在机器人发生交互的位置进行局部精细化，持续更新物体特征和场景图；在支持持续探索与利用的同时，避免重建整个场景，从而实现对动态环境更高效的适应。我们的任务规划模块使用先进的大型语言模型，将任务分解为可管理的子任务。随后，激活导航与操作模块以执行规划的动作。在实验中，我们在多个真实室内环境中部署了移动机器人平台，在不同程度的人工环境修改下开展长期任务测试。我们表明，DovSG 能够准确更新 3D 场景图，并在长期任务及诸如拾取、放置和导航等子任务中表现出色。我们的贡献如下：


- We propose a novel robotic framework that integrates dynamic open-vocabulary 3D scene graphs with language-guided task planning, enabling accurate long-term task execution in dynamic and interactive environments.
- 我们提出了一种新颖的机器人框架，将动态开放词汇 3D 场景图与语言引导的任务规划相结合，使机器人能够在动态且交互式的环境中实现对长期任务的准确执行。


- We construct dynamic 3D scene graphs that capture rich object semantics and spatial relations, performing localized updates as the robot interacts with its environment, allowing it to adapt efficiently to incremental modifications.
- 我们构建了动态 3D 场景图，能够捕捉丰富的物体语义与空间关系；机器人与环境交互时进行局部更新，使其能够高效适应渐进式的修改。


- We develop a task planning method that decomposes complex tasks into manageable subtasks, including pick-up, place, and navigation, enhancing the robot's flexibility and scalability in long-term missions.
- 我们开发了一种任务规划方法，将复杂任务分解为可管理的子任务，包括拾取、放置与导航，从而提升机器人在长期任务中的灵活性与可扩展性。


- We implement DovSG on real-world mobile robots and demonstrate its capabilities across dynamic environments, showing excellent performance in both long-term tasks and subtasks like navigation and manipulation.
- 我们在真实世界的移动机器人上实现了 DovSG，并展示其在动态环境中的能力，证明其在长期任务以及导航与操作等子任务中均具有优秀表现。


---



Received 16 October 2024; accepted 3 February 2025. Date of publication 3 March 2025; date of current version 24 March 2025. This article was recommended for publication by Associate Editor L. Fiorini and Editor G. Venture upon evaluation of the reviewers' comments. This work was supported in part by the National Natural Science Foundation of China NSFC, under Grant 52205244, in part by the Ministry of Industry and Information Technology (MIIT) Key Laboratory of Intelligent Manufacturing for High-end Aerospace Products, and in part by the Beijing Key Laboratory of Digital Design and Manufacturing. (Corresponding author: Zuoxu Wang.)
收到日期：2024 年 10 月 16 日；接收日期：2025 年 2 月 3 日。发布日期：2025 年 3 月 3 日；当前版本日期：2025 年 3 月 24 日。本文经评审意见评估后，推荐由执行编辑 L. Fiorini 和编辑 G. Venture 发表。该工作得到中国国家自然科学基金（NSFC）的部分资助，资助号 52205244；并得到工业和信息化部（MIIT）面向高端航空航天产品智能制造重点实验室的部分资助；同时也得到北京市数字设计与制造重点实验室的资助。（通讯作者：Zuoxu Wang。）


Zhijie Yan, Zuoxu Wang, and Jihong Liu are with the School of Mechanical Engineering and Automation, Beihang University, Beijing 100191, China (email: zuoxu_wang@buaa.edu.cn).
Zhijie Yan、Zuoxu Wang 和 Jihong Liu 属于中国北京 100191 北航（Beihang University）机械工程与自动化学院（邮箱：zuoxu_wang@buaa.edu.cn）。


Shufei Li is with the Department of Systems Engineering, City University of Hong Kong, Hong Kong, SAR 518057, China.
Shufei Li 属于香港城市大学系统工程系，中国香港特别行政区 518057。


Lixiu Wu is with the School of Information Engineering, Minzu University of China, Beijing 100081, China.
Lixiu Wu 属于中国民族大学信息工程学院，中国北京 100081。


Han Wang, Jun Zhu, and Lijiang Chen are with the Afanti Tech LLC, Beijing 100192, China.
Han Wang、Jun Zhu 和 Lijiang Chen 属于 Afanti Tech LLC，中国北京 100192。


Our project page is available at: https://bjhyzj.github.io/dovsg-web.
我们的项目主页可在以下网址获取：https://bjhyzj.github.io/dovsg-web。


This article has supplementary downloadable material available at https://doi.org/10.1109/LRA.2025.3547643, provided by the authors.
本文提供了补充的可下载材料，网址为：https://doi.org/10.1109/LRA.2025.3547643，由作者提供。


Digital Object Identifier 10.1109/LRA.2025.3547643
数字对象标识符 DOI：10.1109/LRA.2025.3547643


---



<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/06/2026_06_08__12_44_58_5707c7.jpg"/>



Fig. 1. Overview of Our DovSG System. DovSG is a mobile robotic system designed to perform long-term tasks in real-world environments. It can detect changes in the scene during task execution, ensuring that subsequent subtasks are completed correctly. The system consists of five main components: perception, memory, task planning, navigation, and manipulation. The memory module includes a lower-level semantic memory and a higher-level scene graph, both of which are continuously updated as the robot explores the environment. This enables the robot to promptly detect manual changes (e.g., keys being moved from cabinet to table) and make the necessary adjustments for subsequent tasks (such as correctly executing Task 2-2).
图 1. 我们的 DovSG 系统概览。DovSG 是一种移动式机器人系统，旨在于真实环境中执行长期任务。它能够在任务执行过程中检测场景变化，从而确保后续子任务被正确完成。该系统由五个主要组件组成：感知、记忆、任务规划、导航和操作。记忆模块包含较低层级的语义记忆与较高层级的场景图，两者都随着机器人探索环境而持续更新。这使得机器人能够及时检测人工变化（例如：按键从柜子被移动到桌子），并为后续任务做出必要调整（例如：正确执行任务 2-2）。


## II. RELATED WORKS
## II. 相关工作


1) 3D Scene Representations for Perception: 3D scene representation in robotics often follows two main approaches: using foundation models to create 3D structures [10], [11], [12], or combining 2D image and vision-language models [6], [7], [8], [9], [13] to connect with the 3D world, showing impressive results in open-vocabulary tasks [14], [15], enabling language-guided object grounding [3], [12], [14], [16] and 3D reasoning [17], [18], [19]. However, dense semantic features for each point are redundant, memory-intensive, and not easily decomposable, limiting their use in dynamic robotics applications.
1) 用于感知的3D场景表示：机器人中的3D场景表示通常遵循两种主要方法：使用基础模型创建3D结构[10]、[11]、[12]，或将2D图像与视觉语言模型[6]、[7]、[8]、[9]、[13]结合以连接3D世界，在开放词汇任务中取得了出色结果[14]、[15]，支持语言引导的目标定位[3]、[12]、[14]、[16]和3D推理[17]、[18]、[19]。然而，每个点的稠密语义特征是冗余的、占用大量内存且难以分解，限制了其在动态机器人应用中的使用。


2) 3D Scene Graphs for Memory: 3D scene graphs use a hierarchical graph structure to represent objects as nodes and their relationships as edges within a scene [20]. This approach efficiently stores semantic information in robot memory, even in large and dynamic environments. Recent methods like Con-ceptGraphs [14] and HOV-SG [21] create more compact and efficient scene graphs by merging features of the same object across multiple views, thereby reducing memory redundancy. However, these methods often assume static environments and overlook scene updates during interactive robot manipulations. RoboEXP [15] addresses this by encoding both spatial relationships and logical associations that reflect the effects of robot actions, enabling the discovery of objects through interaction and supporting dynamic scene updates. Building on this, we have applied these concepts to mobile robots, allowing them to rapidly update memory and adjust the 3D scene graph in real time during dynamic interactions with environments.
2) 用于记忆的3D场景图：3D场景图使用分层图结构，将对象表示为节点，并将其关系表示为场景中的边[20]。这种方法能高效地在机器人记忆中存储语义信息，即使在大型且动态的环境中也是如此。近期方法如Con-ceptGraphs[14]和HOV-SG[21]通过跨多个视角合并同一对象的特征，构建了更紧凑、更高效的场景图，从而减少了记忆冗余。然而，这些方法往往假设环境静态，并忽视了机器人交互操作过程中的场景更新。RoboEXP[15]通过编码反映机器人动作影响的空间关系和逻辑关联来解决这一问题，从而支持通过交互发现物体并进行动态场景更新。在此基础上，我们将这些概念应用于移动机器人，使其能够在与环境的动态交互过程中快速更新记忆，并实时调整3D场景图。


3) Large Language Models for Planning: Large language models (LLMs) [22], [23] and VLMs demonstrate significant potential for zero-shot planning in robotics [24], [25]. These models have been used to generate trajectories and plan manipulations, improving robot adaptability. Integrating LLMs with 3D scene graphs offers further opportunities for task planning. In DovSG, we leverage GPT-4 to decompose tasks into subtasks that can be executed through graph memory, enabling robots to handle complex tasks flexibly and adaptively.
3) 用于规划的大语言模型：大语言模型（LLMs）[22]、[23]和VLM在机器人零样本规划中展现出显著潜力[24]、[25]。这些模型已被用于生成轨迹和规划操作，从而提升机器人的适应性。将LLMs与3D场景图结合为任务规划提供了更多机会。在DovSG中，我们利用GPT-4将任务分解为可通过图记忆执行的子任务，使机器人能够灵活、自适应地处理复杂任务。


4) Indoor Visual Localization for Interaction: Robot interaction relies on accurate tracking of 6-DoF poses using maps constructed from image sequences. Visual relocalization techniques can be categorized into structure-based and learning-based methods. Structure-based approaches employ local descriptors to match 2D pixels to 3D scene coordinates and use PnP algorithms for pose recovery, supported by image retrieval and advanced matching techniques like LightGlue [26] or Match-Former [27]. While effective for large-scale environments, they struggle in small, static indoor settings due to expanding image and feature databases. Learning-based methods, such as ACE [28] and DSAC* [29], predict poses via direct regression, enabling rapid end-to-end optimization but with limited precision due to reliance on approximate pose estimates. These methods also require scene-specific training, which limits scal-ability. To achieve fast and accurate relocalization in indoor environments, we combine the strengths of both approaches. We use ACE to estimate an initial pose from 2D images, employ LightGlue to match the most similar historical 2D images and their poses, and then further refine the pose by iterative closest point (ICP [30]).
4) 用于交互的室内视觉定位：机器人交互依赖于利用由图像序列构建的地图对6-DoF位姿进行准确跟踪。视觉重定位技术可分为基于结构的方法和基于学习的方法。基于结构的方法采用局部描述子将2D像素与3D场景坐标进行匹配，并使用PnP算法恢复位姿，同时结合图像检索和LightGlue[26]或Match-Former[27]等先进匹配技术。尽管这类方法在大规模环境中有效，但由于图像和特征数据库不断扩展，它们在小型静态室内环境中表现不佳。基于学习的方法，如ACE[28]和DSAC*[29]，通过直接回归预测位姿，实现快速的端到端优化，但由于依赖近似位姿估计，精度有限。这些方法还需要针对特定场景训练，限制了可扩展性。为在室内环境中实现快速且准确的重定位，我们结合了两类方法的优势：先使用ACE从2D图像估计初始位姿，再用LightGlue匹配最相似的历史2D图像及其位姿，最后通过迭代最近点（ICP[30]）进一步精细化位姿。


## III. METHOD
## III. 方法


DovSG enables mobile robots to perform long-term tasks in indoor environments by constructing dynamic 3D scene graphs and using large language models for task planning. The process starts with scanning the environment using an RGB-D camera to capture images, followed by open-vocabulary 3D object mapping that detects, associates, and fuses objects into a 3D representation. From these, a 3D scene graph is generated, capturing object relationships and continuously updated when the environment changes. Task planning is performed through language-guided decomposition of long-term tasks into subtasks, which are executed via navigation and manipulation modules.
DovSG 使移动机器人能够在室内环境中执行长期任务：通过构建动态三维场景图，并使用大语言模型进行任务规划。流程从使用 RGB-D 相机扫描环境、采集图像开始；随后进行开放词汇三维物体建图，检测、关联并融合物体，形成三维表示。基于这些信息生成三维场景图，用于刻画物体关系，并在环境变化时持续更新。任务规划通过将长期任务进行语言引导的分解为子任务来完成；子任务随后由导航与操作模块执行。


<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/06/2026_06_08__12_44_58_3f673a.jpg"/>



Fig. 2. Initialization and Construction of 3D Scene Graphs. We first use the RGB-D-based DROID-SLAM [31] model to predict the pose of each frame in the scene. Then, we apply an advanced Open-Vocal segmentation model to segment regions in the RGB images, extract semantic feature vectors for each region, and project them onto a 3D point cloud. Based on semantic, geometric, and CLIP feature similarities, the same object captured from multiple views is gradually associated and fused, resulting in a series of 3D objects. Next, we infer the relationships between objects based on their spatial positions and generate edges connecting these objects, forming a scene graph. This scene graph provides a structured and comprehensive understanding of the scene, allowing efficient localization of target objects and enabling easy reconstruction and updating in dynamic environments, and supports task planning for large language models.
图 2. 三维场景图的初始化与构建。我们首先使用基于 RGB-D 的 DROID-SLAM [31] 模型来预测场景中每一帧的位姿。接着，我们采用更先进的 Open-Vocal 分割模型对 RGB 图像中的区域进行分割，为每个区域提取语义特征向量，并将其投射到三维点云上。基于语义、几何以及 CLIP 特征相似性，从多个视角捕获到的同一物体会被逐步关联与融合，从而得到一系列三维物体。接下来，我们根据物体的空间位置推断它们之间的关系，并生成连接这些物体的边，形成场景图。该场景图为场景提供结构化且全面的理解，使目标物体的定位更加高效，并能在动态环境中方便地重建与更新，同时也支持大语言模型的任务规划。


## A. Home Scanning and Coordinate Transformation
## A. 家庭扫描与坐标变换


1) Home Scanning: Following [4], we captured a video of the room using an Intel Realsense D455 camera, resulting in a sequence of RGB-D images, ${\mathcal{I}}_{t} = {I}_{1},{I}_{2},\ldots ,{I}_{t},{I}_{t} = \; \left\langle  {{I}_{t}^{\text{ rgb }},{I}_{t}^{\text{ depth }}}\right\rangle$ (color and depth),with $t$ as the sequence length. The capture focused on both objects of interest and navigable surfaces, especially around objects and containers.
1) 家庭扫描：按照[4]，我们使用 Intel Realsense D455 相机录制了房间视频，得到一系列 RGB-D 图像，${\mathcal{I}}_{t} = {I}_{1},{I}_{2},\ldots ,{I}_{t},{I}_{t} = \; \left\langle  {{I}_{t}^{\text{ rgb }},{I}_{t}^{\text{ depth }}}\right\rangle$（彩色和深度），$t$ 为序列长度。采集重点覆盖感兴趣的物体和可通行表面，尤其是物体和容器周围。


2) Coordinate Transformation: After data collection, we used DROID-SLAM to estimate the camera poses throughout the sequence, as shown in Fig. 2(top left). To achieve accurate pose estimation at true scale, we replaced DROID-SLAM's depth prediction with actual depth data from the camera sensor,resulting in camera poses, ${\mathcal{P}}^{\text{ droid }}$ . In many SLAM algorithms [31], [32], [33], the first frame's pose is used as the origin, making subsequent poses relative to it, which is often unsuitable for spatial relationships. To normalize the scene, we used the 'floor' as a reference and processed each RGB frame with Grounding DINO and Segment Anything-2 to obtain a floor mask. Since the query only focused on the floor, this process was computationally efficient. Once we obtained the mask, we projected 2D pixels to a 3D point cloud. To align the scene relative to the detected floor, we applied RANSAC to fit a plane to the 3D floor points and computed a transformation matrix ${T}^{\text{ floor }}$ ,aligning the floor with the global $z = 0$ plane. Each pose in ${\mathcal{P}}^{\text{ droid }}$ was then transformed as: $\mathcal{P} = {\mathcal{R}}_{x}{T}^{\text{ floor }}{\mathcal{P}}^{\text{ droid }}$ ,where ${\mathcal{R}}_{x}$ aligns the coordinate system to the x-axis. This yielded the final set of poses, $\mathcal{P} = \left\{  {{I}_{1}^{\text{ pose }},{I}_{2}^{\text{ pose }},\ldots ,{I}_{t}^{\text{ pose }}}\right\}$ ,with the floor properly aligned. By integrating $\mathcal{P}$ with the image sequence ${\mathcal{I}}_{t}$ , we obtained ${I}_{t} = \left\langle  {{I}_{t}^{\text{ rgb }},{I}_{t}^{\text{ depth }},{I}_{t}^{\text{ pose }}}\right\rangle$ .
2) 坐标变换：数据采集后，我们使用 DROID-SLAM 估计整个序列中的相机位姿，如图 2（左上）所示。为获得真实尺度下的准确位姿估计，我们用相机传感器的实际深度数据替换了 DROID-SLAM 的深度预测，得到相机位姿，${\mathcal{P}}^{\text{ droid }}$。在许多 SLAM 算法[31]、[32]、[33]中，首帧位姿被用作原点，使后续位姿相对于其定义，这通常不适合表示空间关系。为使场景归一化，我们以“地面”为参考，使用 Grounding DINO 和 Segment Anything-2 处理每一帧 RGB 图像以获得地面掩码。由于查询仅关注地面，该过程计算开销较低。获得掩码后，我们将二维像素投影为三维点云。为使场景相对于检测到的地面对齐，我们使用 RANSAC 对三维地面点拟合平面，并计算变换矩阵${T}^{\text{ floor }}$，将地面对齐到全局$z = 0$平面。随后，${\mathcal{P}}^{\text{ droid }}$中的每个位姿都按如下方式变换：$\mathcal{P} = {\mathcal{R}}_{x}{T}^{\text{ floor }}{\mathcal{P}}^{\text{ droid }}$，其中${\mathcal{R}}_{x}$将坐标系对齐到 x 轴。这样得到最终位姿集合$\mathcal{P} = \left\{  {{I}_{1}^{\text{ pose }},{I}_{2}^{\text{ pose }},\ldots ,{I}_{t}^{\text{ pose }}}\right\}$，地面对齐正确。通过将$\mathcal{P}$与图像序列${\mathcal{I}}_{t}$结合，我们得到了${I}_{t} = \left\langle  {{I}_{t}^{\text{ rgb }},{I}_{t}^{\text{ depth }},{I}_{t}^{\text{ pose }}}\right\rangle$。


## B. Open-Vocabulary 3D Object Mapping
## B. 开放词汇3D对象映射


With the RGB-D sequences and camera poses, we proceed to construct an object-centric 3D representation from RGB-D observations ${\mathcal{I}}_{t} = \left\{  {{I}_{1},{I}_{2},\ldots ,{I}_{t}}\right\}$ to objects ${\mathbf{O}}_{t} = \left\{  {\mathbf{o}}_{j}\right\}  ,j \in \; \{ 1,\ldots ,J\}$ is all object length,each object ${\mathbf{o}}_{j}$ is characterized by a 3D point cloud ${pc}{d}_{{\mathbf{o}}_{j}}$ ,visual feature ${f}_{{\mathbf{o}}_{j}}^{\text{ rgb }}$ and text feature ${f}_{{\mathbf{o}}_{j}}^{\text{ text }}$ . This map is built incrementally, incorporating each incoming frame ${I}_{t}$ into the existing object set ${\mathbf{O}}_{t - 1}$ ,by either adding to existing objects or instantiating new ones.
利用RGB-D序列和相机位姿，我们继续从RGB-D观测${\mathcal{I}}_{t} = \left\{  {{I}_{1},{I}_{2},\ldots ,{I}_{t}}\right\}$构建以对象为中心的3D表示，对象${\mathbf{O}}_{t} = \left\{  {\mathbf{o}}_{j}\right\}  ,j \in \; \{ 1,\ldots ,J\}$为所有对象长度，每个对象${\mathbf{o}}_{j}$由3D点云${pc}{d}_{{\mathbf{o}}_{j}}$、视觉特征${f}_{{\mathbf{o}}_{j}}^{\text{ rgb }}$和文本特征${f}_{{\mathbf{o}}_{j}}^{\text{ text }}$表征。该地图以增量方式构建，通过将每一帧${I}_{t}$并入现有对象集合${\mathbf{O}}_{t - 1}$，或添加到已有对象，或实例化新对象。


1) Open-Vocabuary 2D Segmentation: To maximize object recognition in the scene, we first apply the image tagging model Recognize-Anything [6] to each frame ${I}_{t}$ ,generating a set of object classes $\left\{  {\mathbf{c}}_{t,i}\right\}  ,i \in  \{ 1,\ldots ,M\}$ detected in the image. We then use $\left\{  {\mathbf{c}}_{t,i}\right\}  ,i \in  \{ 1,\ldots ,M\}$ as input to the 2D detector Grounding DINO to obtain object bounding boxes $\left\{  {\mathbf{b}}_{t,i}\right\}  ,i \in \; \{ 1,\ldots ,M\}$ . Finally,we refine these bounding boxes into object masks $\left\{  {\mathbf{m}}_{t,i}\right\}  ,i \in  \{ 1,\ldots ,M\}$ using the advanced segmentation model Segment Anything-2. For each obtained 2D mask ${\mathbf{m}}_{t,i}$ , we extract a cropped image based on its bounding box, as well as an isolated mask image without background, as illustrated in Fig. 2(bottom-left). We then extract the visual features of each object using two mask-based images with CLIP, and fuse them using a weighted sum method, as described in HOV-SG [21]. This combines the CLIP features from both the cropped target image and the isolated mask image to generate a visual feature descriptor:
1) 开放词汇2D分割：为最大化场景中的对象识别效果，我们首先将图像标注模型Recognize-Anything [6]应用于每一帧${I}_{t}$，生成图像中检测到的一组对象类别$\left\{  {\mathbf{c}}_{t,i}\right\}  ,i \in  \{ 1,\ldots ,M\}$。随后，我们将$\left\{  {\mathbf{c}}_{t,i}\right\}  ,i \in  \{ 1,\ldots ,M\}$作为2D检测器Grounding DINO的输入，以获得对象边界框$\left\{  {\mathbf{b}}_{t,i}\right\}  ,i \in \; \{ 1,\ldots ,M\}$。最后，我们使用先进的分割模型Segment Anything-2将这些边界框细化为对象掩码$\left\{  {\mathbf{m}}_{t,i}\right\}  ,i \in  \{ 1,\ldots ,M\}$。对于每个获得的2D掩码${\mathbf{m}}_{t,i}$，我们根据其边界框提取裁剪图像，并生成不含背景的独立掩码图像，如图2（左下）所示。然后，我们使用CLIP从这两种基于掩码的图像中提取每个对象的视觉特征，并按HOV-SG [21]所述用加权求和方法进行融合。这将裁剪目标图像和独立掩码图像中的CLIP特征结合起来，生成一个视觉特征描述符：


$$
{f}_{t,i}^{\mathrm{{rgb}}} = \operatorname{Embed}\left( {{I}_{t}^{\mathrm{{rgb}}},{\mathbf{b}}_{t,i},{\mathbf{m}}_{t,i}}\right) , \tag{1}
$$



while the text descriptor is obtained by:
而文本描述符则由下式获得：


$$
{f}_{t,i}^{\text{ text }} = \operatorname{Embed}\left( {\mathbf{c}}_{i,t}\right) , \tag{2}
$$



each masked region is then projected into 3D, denoised using DBSCAN clustering with an adaptively computed $\varepsilon$ parameter based on the sorted distances to the $k$ -nearest neighbors,and transformed to the map frame,resulting in a point cloud ${pc}{d}_{t,i}$ , along with unit-normalized semantic feature ${f}_{t,i}^{\text{ rgb }}$ and ${f}_{t,i}^{\text{ text }}$ .
随后将每个掩码区域投影到3D中，使用DBSCAN聚类进行去噪，其参数$\varepsilon$依据到第$k$近邻的排序距离自适应计算，并变换到地图坐标系，得到点云${pc}{d}_{t,i}$，以及单位归一化的语义特征${f}_{t,i}^{\text{ rgb }}$和${f}_{t,i}^{\text{ text }}$。


2) Object Association: For every newly detected object $\left\langle  {{pc}{d}_{t,i},{f}_{t,i}^{\text{ rgb }},{f}_{t,i}^{\text{ text }}}\right\rangle$ ,we compute geometric and semantic similarity with respect to all objects ${\mathbf{o}}_{t - 1,j} = \left\langle  {{pc}{d}_{{\mathbf{o}}_{j}},{f}_{{\mathbf{o}}_{j}}^{\text{ rgb }},{f}_{{\mathbf{o}}_{j}}^{\text{ text }}}\right\rangle$ in the map that shares any partial geometric overlap. The geometric similarity:
2) 对象关联：对于每个新检测到的对象$\left\langle  {{pc}{d}_{t,i},{f}_{t,i}^{\text{ rgb }},{f}_{t,i}^{\text{ text }}}\right\rangle$，我们计算其与地图中所有具有任意部分几何重叠的对象${\mathbf{o}}_{t - 1,j} = \left\langle  {{pc}{d}_{{\mathbf{o}}_{j}},{f}_{{\mathbf{o}}_{j}}^{\text{ rgb }},{f}_{{\mathbf{o}}_{j}}^{\text{ text }}}\right\rangle$的几何和语义相似度。几何相似度：


$$
{\mathbf{s}}_{\text{ geo }}\left( {i,j}\right)  = \text{ Nnrate }\left( {{\operatorname{pcd}}_{t,i},{\mathrm{{pcd}}}_{{\mathbf{o}}_{j}}}\right) \tag{3}
$$



is defined as the ratio of the number of points in point cloud ${pc}{d}_{t,i}$ that have nearest neighbors in point cloud ${pc}{d}_{{\mathbf{o}}_{j}}$ ,within a distance threshold of ${\delta }_{\mathrm{{nn}}}$ ,to the total number of points in ${pc}{d}_{t,i}$ . The visual and text similarity:
定义为点云${pc}{d}_{t,i}$中在距离阈值${\delta }_{\mathrm{{nn}}}$内、其最近邻位于点云${pc}{d}_{{\mathbf{o}}_{j}}$中的点数，与${pc}{d}_{t,i}$中总点数的比值。视觉和文本相似度：


$$
{\mathbf{s}}_{\text{ vis }}\left( {i,j}\right)  = {\left( {f}_{t,i}^{\mathrm{{rgb}}}\right) }^{\top }{f}_{{\mathbf{o}}_{j}}^{\mathrm{{rgb}}}/2 + 1/2 \tag{4}
$$



$$
{\mathbf{s}}_{\text{ text }}\left( {i,j}\right)  = {\left( {f}_{t,i}^{\text{ text }}\right) }^{\top }{f}_{{\mathbf{o}}_{j}}^{\text{ text }}/2 + 1/2 \tag{5}
$$



is the normalized cosine distance between the corresponding visual descriptors. The overall similarity measure $s\left( {i,j}\right)$ is a weighted sum of the individual similarity measures:
是对应视觉描述符之间的归一化余弦距离。整体相似度$s\left( {i,j}\right)$是各单项相似度的加权和：


$$
s\left( {i,j}\right)  = {\mathbf{s}}_{\mathrm{{vis}}}\left( {i,j}\right)  \cdot  {\omega }_{\mathrm{v}} + {\mathbf{s}}_{\mathrm{{geo}}}\left( {i,j}\right)  \cdot  {\omega }_{\mathrm{g}} + {\mathbf{s}}_{\mathrm{{text}}}\left( {i,j}\right)  \cdot  {\omega }_{\mathrm{t}}, \tag{6}
$$



where ${\omega }_{\mathrm{v}} + {\omega }_{\mathrm{g}} + {\omega }_{\mathrm{t}} = 1$ . We perform object association by a greedy assignment strategy where each new detection is matched with an existing object with the highest similarity score. If no match is found with a similarity higher than ${\delta }_{\text{ sim }}$ ,we initialize a new object.
其中${\omega }_{\mathrm{v}} + {\omega }_{\mathrm{g}} + {\omega }_{\mathrm{t}} = 1$。我们采用贪心分配策略进行目标关联，即将每个新检测结果与现有目标中相似度最高的对象匹配。若未找到相似度高于${\delta }_{\text{ sim }}$的匹配，则初始化一个新对象。


3) Object Fusion: If a detection ${\mathbf{o}}_{t - 1,j}$ is associated with a mapped object ${\mathbf{o}}_{j}$ ,we fuse the detection with the map. This is achieved by updating the object's features as
3) 目标融合：如果检测${\mathbf{o}}_{t - 1,j}$与一个已映射对象${\mathbf{o}}_{j}$相关联，我们将该检测与地图融合。其方法是将该对象的特征更新为


$$
{f}_{{\mathbf{o}}_{j}}^{\mathrm{{rgb}}} = \left( {{n}_{{\mathbf{o}}_{j}}{f}_{{\mathbf{o}}_{j}}^{\mathrm{{rgb}}} + {f}_{t,i}^{\mathrm{{rgb}}}}\right) /\left( {{n}_{{\mathbf{o}}_{j}} + 1}\right) , \tag{7}
$$



where ${n}_{{\mathbf{o}}_{j}}$ is the number of detections that have been associated to ${\mathbf{o}}_{j}$ so far; and updating the pointcloud as ${pc}{d}_{t,i} \cup  {pc}{d}_{{\mathbf{o}}_{j}}$ , followed by down-sampling to remove redundant points.
其中${n}_{{\mathbf{o}}_{j}}$是截至目前已与${\mathbf{o}}_{j}$关联的检测数量；并将点云更新为${pc}{d}_{t,i} \cup  {pc}{d}_{{\mathbf{o}}_{j}}$，随后进行下采样以去除冗余点。


### C.3D Scene Graph Generation
### C.3D 场景图生成


DovSG constructs a 3D scene graph ${\mathcal{G}}_{t} = \left\langle  {{\mathbf{O}}_{t},{\mathbf{E}}_{t}}\right\rangle$ ,where and ${\mathbf{E}}_{t} = \left\{  {\mathbf{e}}_{k}\right\}  ,k \in  \{ 1,\ldots ,K\}$ represent the sets of objects and edges,respectively. Given the set of 3D objects ${\mathbf{O}}_{T}$ obtained from the previous step, we estimate their spatial relationships, i.e.,the edges ${\mathbf{E}}_{T}$ ,to complete the 3D scene graph,as shown in Fig. 2(column 3). Leveraging the transformation of the scene's coordinate system from the previous steps-where the ground plane serves as the origin and the z-axis points upwards-we can efficiently extract the fundamental spatial relationships among the objects. While focusing on these relationships, we first voxelize the point cloud ${pc}{d}_{{\mathrm{o}}_{j}}$ of each object ${\mathbf{o}}_{j}$ in the scene. This reduces computation and storage by converting dense point clouds into sparse voxel grids, ensuring efficient scene updates in future tasks. We focus on three relationships: "on", "belong", and "inside", as mentioned in RoboEXP [15], as these relationships are sufficient for downstream tasks. The "on" relationship denotes a stacking or positional hierarchy between objects, such as an apple being placed on a table. The "belong" relationship captures ownership or attachment between objects, like a refrigerator handle belonging to the refrigerator. Lastly, the "inside" relationship is particularly relevant during the robot's exploration when it opens containers such as drawers or cabinets, revealing that certain objects are contained within others. It is important to note that for "inside" relationship, we limit our focus to small-scale containment between objects, as technically, every object in the scene could be considered "inside" the overall home environment.
DovSG 构建一个三维场景图 ${\mathcal{G}}_{t} = \left\langle  {{\mathbf{O}}_{t},{\mathbf{E}}_{t}}\right\rangle$ ，其中 ${\mathbf{E}}_{t} = \left\{  {\mathbf{e}}_{k}\right\}  ,k \in  \{ 1,\ldots ,K\}$ 分别表示对象集合与边的集合。给定上一阶段得到的三维对象集合 ${\mathbf{O}}_{T}$ ，我们估计它们的空间关系，即边 ${\mathbf{E}}_{T}$ ，以补全三维场景图，如图 2（第 3 列）所示。通过沿用前述步骤中场景坐标系的变换——其中地面平面作为原点，z 轴指向上方——我们可以高效提取对象之间的基础空间关系。在关注这些关系时，我们首先对场景中每个对象 ${\mathbf{o}}_{j}$ 的点云 ${pc}{d}_{{\mathrm{o}}_{j}}$ 进行体素化。这通过将密集点云转换为稀疏体素网格，降低了计算与存储开销，从而确保后续任务中场景更新的高效性。我们聚焦 RoboEXP [15] 中提到的三种关系：“on”“belong”和“inside”，因为这些关系足以满足下游任务。“on”关系表示对象之间的堆叠或位置层级，例如一个苹果放在桌子上。“belong”关系刻画对象之间的从属或连接关系，例如冰箱把手属于冰箱。最后，“inside”关系在机器人探索过程中尤其相关：当它打开抽屉或柜子等容器时，会发现某些对象被包含在其他对象内部。需要注意的是，对于“inside”关系，我们仅关注对象间的小范围包含；从技术上讲，场景中的每个对象都可以被视为“位于”整个家居环境之内。


<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/06/2026_06_08__12_44_58_26a918.jpg"/>



Fig. 3. Adaptation in interactions with manually modified scenes. (1) We train the scene-specific regression MLP of the ACE model using RGB images and their poses, making the process highly efficient. (2) After manual scene modification, multi-view observations allow rough global pose estimation via ACE, refined further using LightGlue and ICP. The new viewpoint's point cloud closely aligns with the stored pose. (3) The bottom image shows accurate local updates to the scene based on observations from the new viewpoint.
图 3. 在手动修改场景中的交互适应。（1）使用 RGB 图像及其位姿训练 ACE 模型的场景特定回归 MLP，使过程高度高效。（2）手动修改场景后，多视角观测可借助 ACE 进行粗略全局位姿估计，并进一步使用 LightGlue 与 ICP 细化。新视角的点云与存储的位姿高度贴合。（3）下方图像展示了基于新视角观测对场景进行的准确局部更新。


## D. Dynamic Scene Adaptation
## D. 动态场景自适应


In dynamic indoor interaction scenarios, the layout of the environment changes due to human activities or the robot's task execution. These changes are often invisible to previous approaches [3], [14], [21], and in such environments, if the robot cannot dynamically update its memory, it will soon face failure. To address this issue, we have designed a simple memory update module that can quickly perform local updates to the memory based on new RGB-D observations captured by the robot, as shown in Fig. 3.
在动态室内交互场景中，由于人类活动或机器人的任务执行，环境的布局会发生变化。这些变化通常对以往方法来说是不可见的[3]，[14]，[21]；在此类环境中，如果机器人无法动态更新其记忆，就很快会失败。为解决这一问题，我们设计了一个简单的记忆更新模块：基于机器人采集的新的 RGB-D 观测，可快速对记忆进行局部更新，如图 3 所示。


1) Relocalization and Refinement: Accurate re-localization is crucial for a mobile robot performing manipulations in dynamic environments to maintain an up-to-date scene graph. To ensure precise local updates to the voxelized map, we employ a multi-stage relocalization process that efficiently integrates new observations into the existing scene representation. We begin by training an ACE scene-specific regression MLP using memory $\left\langle  {{I}_{i}^{\text{ rgb }},{I}_{i}^{\text{ pose }}}\right\rangle  ,i \in  \{ 1,\ldots ,t\}$ (see the top left corner in Fig. 3). After the robot collects new RGB-D observations ${I}_{k}$ for $k \in  \{ t + 1,\ldots ,t + n\}$ ,where each observation ${I}_{k} = \; \left\langle  {{I}_{k}^{\text{ rgb }},{I}_{k}^{\text{ depth }},{I}_{k}^{\text{ c2b }}}\right\rangle$ includes the RGB image,depth image,and the camera-to-arm-base transformation, we proceed to estimate the robot's pose within the existing map. Then, leveraging the trained ACE model,we predict a rough global pose ${I}_{k}^{\text{ pose }}$ for each new RGB image ${I}_{k}^{\text{ rgb }}$ by utilizing the prior mapping data $\left\langle  {{I}_{i}^{\text{ rgb }},{I}_{i}^{\text{ pose }}}\right\rangle$ ,where $i \in  \{ 1,\ldots ,t\}$ . While ACE provides a coarse alignment, its precision may not suffice for detailed scene updates. To refine these pose estimates, we perform local feature matching using LightGlue [26], identifying the historical image ${I}_{\widehat{k}}^{\text{ rgb }},\widehat{k} \in  \{ 1,\ldots ,t\}$ with the most feature correspondences to each new observation ${I}_{k}^{\text{ rgb }}$ . This step enhances the robustness of the pose estimation by anchoring it to the most similar view in the prior map. With the matched image pairs, we extract the corresponding point clouds from the RGB-D data and perform multi-scale colored iterative closest point (ICP) alignment. This refinement step minimizes both geometric and photometric discrepancies between the new observations and the existing map,yielding a precise transformation ${T}_{k}^{\text{ icp }}$ that aligns the new observations within the global coordinate frame. The refined poses ${I}_{k}^{\text{ pose }}$ are then updated using the transformation ${I}_{k}^{\text{ pose }} \leftarrow  {T}^{\text{ icp }}{I}_{k}^{\text{ pose }}.$
1) 重定位与精炼：在动态环境中执行操作的移动机器人要保持最新的场景图，准确的重定位至关重要。为确保对体素化地图进行精确的局部更新，我们采用多阶段重定位流程，将新的观测高效融合到现有场景表征中。首先，我们使用记忆 $\left\langle  {{I}_{i}^{\text{ rgb }},{I}_{i}^{\text{ pose }}}\right\rangle  ,i \in  \{ 1,\ldots ,t\}$（见图 3 左上角）训练一个面向 ACE 的场景特定回归 MLP。机器人收集新的 RGB-D 观测 ${I}_{k}$（每个观测 ${I}_{k} = \; \left\langle  {{I}_{k}^{\text{ rgb }},{I}_{k}^{\text{ depth }},{I}_{k}^{\text{ c2b }}}\right\rangle$ 包含 RGB 图像、深度图像以及相机到机械臂基座的变换 $k \in  \{ t + 1,\ldots ,t + n\}$）后，我们在现有地图中估计机器人的位姿。随后，利用已训练的 ACE 模型，通过先验映射数据 $\left\langle  {{I}_{i}^{\text{ rgb }},{I}_{i}^{\text{ pose }}}\right\rangle$，为每幅新的 RGB 图像 ${I}_{k}^{\text{ rgb }}$ 预测一个粗略的全局位姿 ${I}_{k}^{\text{ pose }}$，其中 $i \in  \{ 1,\ldots ,t\}$。尽管 ACE 能提供粗配准，但其精度可能不足以支持对场景的细致更新。为精炼位姿估计，我们使用 LightGlue [26] 进行局部特征匹配：为每个新的观测 ${I}_{k}^{\text{ rgb }}$ 找到历史图像 ${I}_{\widehat{k}}^{\text{ rgb }},\widehat{k} \in  \{ 1,\ldots ,t\}$，使其与新观测具有最多的特征对应关系。该步骤通过将位姿锚定到先验地图中最相似的视角来提升估计的鲁棒性。利用匹配的图像对，我们从 RGB-D 数据中提取对应点云，并执行多尺度彩色迭代最近点（ICP）配准。精炼步骤可同时最小化新观测与现有地图之间的几何差异与光度差异，从而得到一个精准的变换 ${T}_{k}^{\text{ icp }}$，使新观测在全局坐标系中对齐。随后，通过变换 ${I}_{k}^{\text{ pose }} \leftarrow  {T}^{\text{ icp }}{I}_{k}^{\text{ pose }}.$ 更新精炼后的位姿 ${I}_{k}^{\text{ pose }}$。


2) Remove Obsolete Indices: This step is to identify and remove obsolete voxels from the memory map. We propose an efficient method that leverages new RGB-D observations to update the volumetric representation accordingly. Given the set of stored voxel indices, each associated with 3D positions and color information,we process each new observation ${I}_{k}$ as follows: (1) We transform the all voxel point cloud (points) $\left\{  {{pc}{d}_{{\mathbf{o}}_{j}}}\right\}  ,j \in  \{ 1,\ldots ,J\}$ into the current camera coordinate by:
2) 移除过时索引：该步骤用于识别并从记忆地图中移除过时的体素。我们提出一种高效方法：利用新的 RGB-D 观测来相应更新体积表征。给定已存储的体素索引集合（每个索引对应三维位置和颜色信息），我们按如下方式处理每个新的观测 ${I}_{k}$： (1) 将所有体素点云（点） $\left\{  {{pc}{d}_{{\mathbf{o}}_{j}}}\right\}  ,j \in  \{ 1,\ldots ,J\}$ 通过如下方式变换到当前相机坐标系：


$$
{pc}{d}_{{\mathbf{o}}_{j}}^{\text{ cam }} = {\left( {I}_{k}^{\text{ pose }}\right) }^{-1}{pc}{d}_{{\mathbf{o}}_{j}} \tag{8}
$$



(2) The transformed points are then projected onto the image plane using the camera's intrinsic parameters to obtain pixel coordinates:
(2) 变换后的点再使用相机的内参投影到图像平面，得到像素坐标：


$$
\left\lbrack  {{\mathbf{u}}_{j},{\mathbf{v}}_{j}}\right\rbrack   = \Pi \left( {{pc}{d}_{{\mathbf{o}}_{j}}^{\mathrm{{cam}}}}\right) , \tag{9}
$$



where $\Pi$ denotes the projection function. We consider only points that project within the image boundaries and are in front of the camera,that is, ${z}_{j} > 0,{z}_{j}$ is the z-axis coordinates of the ${pc}{d}_{{\mathbf{o}}_{j}}^{\text{ cam }}$ . (3) For each valid projected point $i$ ,we compute the depth and color difference:
其中 $\Pi$ 表示投影函数。我们只考虑投影落在图像边界内、且位于相机前方的点，即 ${z}_{j} > 0,{z}_{j}$ 为 ${pc}{d}_{{\mathbf{o}}_{j}}^{\text{ cam }}$ 的 z 轴坐标。(3) 对于每个有效投影点 $i$ ，我们计算深度与颜色差异：


$$
\Delta {z}_{i} = \left| {{I}_{k}^{\text{ depth }}\left\lbrack  {{\mathbf{u}}_{j}^{i},{\mathbf{v}}_{j}^{i}}\right\rbrack   - {z}_{j}^{i}}\right| , \tag{10}
$$



$$
\Delta {c}_{i} = \left| {{I}_{k}^{\mathrm{{rgb}}}\left\lbrack  {{\mathbf{u}}_{j}^{i},{\mathbf{v}}_{j}^{i}}\right\rbrack   - {\mathbf{c}}_{j}^{i}}\right| , \tag{11}
$$



where ${\mathbf{c}}_{j}$ is the stored color of ${pc}{d}_{{\mathbf{o}}_{j}}$ . (4) We define thresholds ${\delta }_{z},{\delta }_{z}^{\prime }$ ,and ${\delta }_{c}$ for depth and color differences. A point $i$ is marked for deletion if:
其中 ${\mathbf{c}}_{j}$ 为 ${pc}{d}_{{\mathbf{o}}_{j}}$ 已存的颜色。(4) 我们为深度和颜色差异定义阈值 ${\delta }_{z},{\delta }_{z}^{\prime }$ 和 ${\delta }_{c}$ 。若满足下列条件，点 $i$ 将被标记为删除：


$$
\text{ point }i\text{ is deleted if }\left\{  \begin{array}{l} \Delta {z}_{i} > {\delta }_{z}, \\  \Delta {z}_{i} > {\delta }_{z}^{\prime }\text{ and }\Delta {c}_{i} > {\delta }_{c}. \end{array}\right. \tag{12}
$$



3) Update Low-Level Memory: After the above step, the local scene in the historical low-level memory will be updated to the latest state based on the visual information provided by the new RGB-D observations ${I}_{k},k \in  t + 1,\ldots ,t + n$ . We process ${I}_{k}$ sequentially in the same way as described in Section III-B,and fuse it with the historical ${\mathbf{O}}_{t}$ to ${\mathbf{O}}_{t + 1}$ . The set $\left\langle  {{I}_{k}^{\text{ rgb }},{I}_{k}^{\text{ depth }},{I}_{k}^{\text{ pose }}}\right\rangle  ,k \in  t + 1,\ldots ,t + n$ will also be updated into ${\mathcal{I}}_{t + 1}$ ,providing new reference viewpoints for the next re-localization to ensure long-term localization accuracy.
3) 更新低层内存：完成上述步骤后，将根据新的 RGB-D 观测 ${I}_{k},k \in  t + 1,\ldots ,t + n$ 提供的视觉信息，把历史低层内存中的局部场景更新为最新状态。我们以与第 III-B 节相同的方式依次处理 ${I}_{k}$ ，并将其与历史 ${\mathbf{O}}_{t}$ 融合到 ${\mathbf{O}}_{t + 1}$ 。集合 $\left\langle  {{I}_{k}^{\text{ rgb }},{I}_{k}^{\text{ depth }},{I}_{k}^{\text{ pose }}}\right\rangle  ,k \in  t + 1,\ldots ,t + n$ 也将更新为 ${\mathcal{I}}_{t + 1}$ ，从而为下一次重定位提供新的参考视点，以保证长期定位精度。


4) Update High-Level Memory: For updating the scene graph, we adopt a simple local sub-graph-based update strategy to avoid redundant global updates. Specifically, (1) We compare the historical object set ${\mathbf{O}}_{t}$ with the updated object set ${\mathbf{O}}_{t + 1}$ to identify the objects that have been deleted or whose points have been changed, forming the set ${\mathbf{O}}_{\text{ affected }}$ . (2) Next,we examine the edges in ${\mathbf{E}}_{t}$ to find the parent objects of all objects in ${\mathbf{O}}_{\text{ affected }}$ and the child objects of those parent objects. These related objects are also added to ${\mathbf{O}}_{\text{ affected }}$ ,and all edges associated with ${\mathbf{O}}_{\text{ affected }}$ are removed from ${\mathbf{E}}_{t}$ . (3) We then identify the objects in ${\mathbf{O}}_{\text{ affected }}$ that still exist in ${\mathbf{O}}_{t + 1}$ and locate any newly added objects in ${\mathbf{O}}_{t + 1}$ . Both these types of objects are added to the set ${\mathbf{O}}_{\text{ need\_process }}$ for further processing. (4) For each object in ${\mathbf{O}}_{\text{ need\_process }}$ ,we recompute the spatial relationships as described in Section III-C. This involves determining the relevant edges e for each object, updating ${\mathbf{E}}_{t + 1}$ with the new connections and builds ${\mathcal{G}}_{t + n}$ .
4) 更新高层内存：为更新场景图，我们采用一种基于局部子图的简单更新策略，以避免冗余的全局更新。具体而言：(1) 我们将历史物体集合 ${\mathbf{O}}_{t}$ 与更新后的物体集合 ${\mathbf{O}}_{t + 1}$ 进行比较，找出已被删除的物体或其点发生变化的物体，形成集合 ${\mathbf{O}}_{\text{ affected }}$ 。(2) 接着，我们检查 ${\mathbf{E}}_{t}$ 中的边，以找到 ${\mathbf{O}}_{\text{ affected }}$ 中所有物体的父物体以及这些父物体的子物体。与之相关的物体也被加入 ${\mathbf{O}}_{\text{ affected }}$ ，并从 ${\mathbf{E}}_{t}$ 中移除所有与 ${\mathbf{O}}_{\text{ affected }}$ 相关的边。(3) 然后，我们识别 ${\mathbf{O}}_{\text{ affected }}$ 中仍存在于 ${\mathbf{O}}_{t + 1}$ 的物体，并定位 ${\mathbf{O}}_{t + 1}$ 中新增的物体。上述两类物体都会被加入集合 ${\mathbf{O}}_{\text{ need\_process }}$ 以供进一步处理。(4) 对 ${\mathbf{O}}_{\text{ need\_process }}$ 中的每个物体，我们按第 III-C 节所述重新计算空间关系。该过程包括为每个物体确定相关边 e ，用新的连接更新 ${\mathbf{E}}_{t + 1}$ 并构建 ${\mathcal{G}}_{t + n}$ 。


## E. Language-Guided Task Planning
## E. 语言引导任务规划


Benefiting from the system's long-term workability in indoor environments, we integrated the advanced large language model GPT-40. Based on class-agnostic and highly extensible prompt text descriptions, we decompose long-term tasks described in natural language into multiple subtasks that the robot can easily adapt to. Each subtask output by GPT consists of an "action_name" and multiple "object_name", which are directly extracted from the description and maintain the same level of abstraction as the described objects, ensuring that the original meaning is not distorted during task decomposition.
鉴于该系统在室内环境中的长期可用性，我们集成了先进的大语言模型 GPT-40。基于与类别无关且高度可扩展的提示文本描述，我们将自然语言中描述的长期任务拆分为多个子任务，以便机器人轻松适配。由 GPT 输出的每个子任务包含一个“action_name”和多个“object_name”，这些信息将直接从描述中提取，并保持与所述对象相同的抽象层级，从而在任务分解过程中确保原意不被扭曲。


## F. Navigation
## F. 导航


1) Localization: Before initiating each navigation task, we first determine the robot's precise pose as described in Section III-D1, establishing its position in the world coordinate system (start point). To locate the target object(s), we utilize CLIP to obtain embeddings of the specified object names. If only object A is specified, we compute its CLIP embedding and compare it with the embeddings of all objects in the scene using cosine similarity. The object with the highest similarity score is identified as the target location of A (as illustrated in the upper section of Fig. 2). For tasks involving a spatial relationship between two objects (e.g., "object A is on object B"), we compute CLIP embeddings for both A and B. We then compare these embeddings with those of the scene objects to obtain similarity scores. For each object,we select the top- $k$ most similar scene objects (top- $k$ A and top- $k$ B,respectively). Next, we calculate the Euclidean distances between each pair of candidate locations from top- $k$ A and top- $k$ B. The pair with the shortest distance is deemed the most probable configuration, and the location of the corresponding A candidate is selected as the target point.
1) 定位：在启动每个导航任务之前，我们首先按照第 III-D1 节所述确定机器人的精确位姿，建立其在世界坐标系中的位置（起点）。为定位目标对象，我们使用 CLIP 获取指定对象名称的嵌入表示。若仅指定对象 A，则计算其 CLIP 嵌入，并与场景中所有对象的嵌入通过余弦相似度进行比较。相似度最高的对象被确定为 A 的目标位置（如图 2 上半部分所示）。对于涉及两个对象空间关系的任务（例如“对象 A 在对象 B 上”），我们分别计算 A 和 B 的 CLIP 嵌入。随后将这些嵌入与场景对象的嵌入进行比较，得到相似度分数。对于每个对象，我们选取最相似的前- $k$ 个场景对象（分别为前- $k$ 个 A 和前- $k$ 个 B）。接着，计算前- $k$ 个 A 与前- $k$ 个 B 中每一对候选位置之间的欧氏距离。距离最短的一对被视为最可能的配置，并选取对应 A 候选的位置作为目标点。


2) Mobile Control: Once the target location is determined, we use the A* [34] algorithm to generate a collision-free navigation path from the start point to the target point. The robot then follows this path using a PID controller to ensure accurate and smooth navigation.
2) 移动控制：一旦确定目标位置，我们使用 A* [34] 算法从起点到目标点生成无碰撞导航路径。随后机器人沿此路径在 PID 控制器的作用下运行，以确保导航准确而平稳。


## G. Manipulation
## G. 操作


Once the robot reaches the target location, we employ a pick-and-place strategy similar to Ok-Robot [3]. First, the robot retrieves the object's 3D coordinates from semantic memory and points its camera at the target to capture RGBD images. Then, the robot performs either a "Pick up" or "Place".
当机器人到达目标位置后，我们采用一种类似于 Ok-Robot [3] 的抓取-放置策略。首先，机器人从语义记忆中检索物体的三维坐标，并将相机对准目标以采集 RGBD 图像。随后，机器人执行“抓取”或“放置”。


1) Pick up: To focus the AnyGrasp model on the target object, we first preprocess the point cloud by cropping it to a region around the target object, based on its detected bounding box. This step refines the input for AnyGrasp, leading to more accurate and efficient grasp predictions. After generating candidate grasps, we apply cost-based filtering to select the best option. The robot executes the grasp with the highest confidence, leveraging additional segmentation from Grounding DINO and Segment Anything-2 for precise targeting. Furthermore, we introduce a heuristic grasp strategy that uses the object's bounding box information to rotate and align the gripper for the most suitable grasp orientation, as shown in Fig. 4. This heuristic strategy is only activated when AnyGrasp can't provide a suitable grasp, ensuring optimal interaction with the object's geometry
1) 抓取：为使 AnyGrasp 模型聚焦于目标物体，我们首先根据其检测到的边界框，对点云进行预处理，将其裁剪到目标物体周围的区域。这一步优化了 AnyGrasp 的输入，从而带来更准确、更高效的抓取预测。在生成候选抓取后，我们采用基于代价的筛选来选择最佳方案。机器人执行置信度最高的抓取，并借助 Grounding DINO 和 Segment Anything-2 的额外分割实现精确定位。此外，我们提出了一种启发式抓取策略，利用物体的边界框信息旋转并对齐夹爪，以获得最合适的抓取姿态，如图 4 所示。该启发式策略仅在 AnyGrasp 无法提供合适抓取时启用，从而确保与物体几何形状的最优交互


<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/06/2026_06_08__12_44_58_a79e2a.jpg"/>



Fig. 4. Two proposed grasp strategies in DoVSG. In the first row, we cropped the point cloud input into anyGrasp within a certain range around the target object, allowing anyGrasp to focus more on the target object without compromising the generation of collision-free grasps. Furthermore, we filtered the grasps based on translational and rotational costs, with the red grasps indicating the highest confidence. In the second row, we show our heuristic grasp strategy, which leverages the object's bounding box information to rotate and select the most appropriate grasp orientation.
图 4. DoVSG 中提出的两种抓取策略。第一行中，我们将输入 AnyGrasp 的点云裁剪到目标物体周围一定范围内，使 AnyGrasp 能更聚焦于目标物体，同时不影响无碰撞抓取的生成。此外，我们基于平移和旋转代价对抓取进行筛选，其中红色抓取表示置信度最高。第二行中，我们展示了启发式抓取策略，它利用物体的边界框信息来旋转并选择最合适的抓取姿态。


2) Place: We first obtain the point cloud ${pc}{d}_{a}$ of the target object using Grounding DINO and Segment Anything-2 (SAM2), as described in Section III-B1. This point cloud is then transformed into the robot's base coordinate frame. Next, we compute the median coordinates ${x}_{m}$ and ${y}_{m}$ ,and determine the drop height as follows:
2) 放置：我们首先使用 Grounding DINO 和 Segment Anything-2（SAM2）获得目标物体的点云 ${pc}{d}_{a}$，如第 III-B1 节所述。随后将该点云转换到机器人的基坐标系中。接着，我们计算中位数坐标 ${x}_{m}$ 和 ${y}_{m}$，并如下确定下落高度：


$$
{z}_{\max } = {0.1} + \max \left( {z\left| {0 \leq  x \leq  {x}_{m},}\right| y - {y}_{m} \mid   < {0.1}}\right) , \tag{13}
$$



where $\left( {x,y,z}\right)  \in  {pc}{d}_{a}$ . A buffer of 0.1 is added to account for potential collisions. The robot executes the placement manipulation at the computed coordinates $\left( {{x}_{m},{y}_{m},{z}_{\max }}\right)$ .
其中 $\left( {x,y,z}\right)  \in  {pc}{d}_{a}$。为考虑潜在碰撞，额外加入 0.1 的缓冲量。机器人在计算得到的坐标 $\left( {{x}_{m},{y}_{m},{z}_{\max }}\right)$ 处执行放置操作。


## IV. EXPERIMENTS
## IV. 实验


In this section, We evaluate DovSG's performance in dynamic, real-world environments to answer two key questions: (1) How well does our system adapt to changes by updating the dynamic scene graph? (2) How effectively does this facilitate the completion of consecutive tasks without manual resets?
在本节中，我们评估 DovSG 在动态、真实世界环境中的性能，以回答两个关键问题：（1）通过更新动态场景图，我们的系统对变化的适应能力有多强？（2）它能否在无需手动重置的情况下，帮助连续任务更有效地完成？


1) Robot Setups: We used a real-world setup with a UFAC-TORY xARM6 robotic arm on an Agilex Ranger Mini 3 mobile base, equipped with a RealSense D455 camera for perception and a basket for item transport.
1）机器人设置：我们使用了真实世界设置：在 Agilex Ranger Mini 3 移动底盘上配备 UFAC-TORY xARM6 机械臂，并安装 RealSense D455 相机用于感知，同时放置了一个用于物品运输的篮子。


2) Environment and Task Setups: To verify our method's ability to enable robots to perform long-term tasks in dynamic environments, we designed an experiment in 4 real-world rooms. The experiment simulated dynamic environments through human interactions, which modified object positions, added new objects, or revealed hidden ones. Each experiment began with a 3D scene graph and involved two consecutive tasks. Before the first task, we manually adjusted objects related to the second task by changing their positions, adding new objects, or revealing hidden ones. These modifications were designed to be detectable during the execution of the first task. After completing the first task, instead of being repositioned to an initial start point, the robot continued to the next task from its current state. This setup simulated real-world scenarios with continuous environmental changes caused by human interference.
2）环境与任务设置：为验证我们的方法能否让机器人在动态环境中执行长期任务，我们在4个真实房间中设计了实验。实验通过人与环境的交互来模拟动态环境：这会改变物体位置、添加新物体，或暴露隐藏物体。每次实验都从一个3D场景图开始，并包含两个连续任务。在第一个任务开始前，我们通过改变相关物体的位置、添加新物体或暴露隐藏物体，手动调整与第二个任务相关的对象。这些修改旨在能在执行第一个任务期间被检测到。完成第一个任务后，机器人不会被重新放置到预设的起始点，而是从其当前状态继续执行下一个任务。该设置模拟了由人为干扰引起、环境持续变化的真实场景。


<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/06/2026_06_08__12_44_58_e741ab.jpg"/>



Fig. 5. Degrees of environmental modifications. The left column shows the initial state of the scene, while the two columns on the right represent the state of the scene after manual modifications.
图 5. 环境修改的程度。左栏展示场景的初始状态，右侧两栏表示经过手动修改后的场景状态。


To evaluate the robot's ability to detect and adapt to environmental changes, we categorized the modifications to the objects in the second task into three levels (see Fig. 5):
为评估机器人检测并适应环境变化的能力，我们将第二个任务中物体的修改分为三个层级（见图5）：


(1) Minor Adjustment: Slight position changes that remain detectable from the original location. For instance, a slightly moved object may still be visible within the robot's field of view. (2) Appearance: Previously hidden objects become visible, such as items revealed by opening a drawer or placing a key on the table. These changes introduce new nodes into the scene graph. (3) Positional Shift: Significant relocations of objects render them undetectable from their original positions, altering their spatial relationships with other objects in the scene graph.
（1）轻微调整：在原位置附近发生轻微位置变化，但仍能从原位置被检测到。例如，一个稍微移动的物体可能仍在机器人的视野范围内可见。（2）外观出现：先前隐藏的物体变得可见，例如通过打开抽屉露出物品，或将钥匙放在桌上。这些变化会向场景图引入新的节点。（3）位置偏移：物体发生显著迁移，使其在原位置处无法被检测，并改变它们与场景图中其他物体之间的空间关系。


Overall, for each method, each level of modification was tested through 20 long-term tasks per room across 4 rooms, resulting in a total of 80 trials per modification type. With three modification levels, each method was subjected to 240 long-term task experiments, in which objects or their positions were randomized in every trial.
总体而言，对于每种方法，每个修改层级在每个房间中都进行了20次长期任务测试，共4个房间，因此每种修改类型总计80次试验。在三个修改层级下，每种方法共进行了240次长期任务实验，且每次试验中物体或其位置都会被随机化。


3) Baselines: To demonstrate our approach's adaptability to changing environments and the effectiveness of scene graph prediction, we evaluated scene changes and scene graph predictions. Using GPT-40 as the base model, enhanced with chain-of-thought (CoT) reasoning [35], similar to the method proposed by Jiang et al. [15], the model takes RGB observations from the robot, matches them to the most similar image in memory, detects differences, and predicts the scene graph based on the new RGB observation. For long-term task execution, we implemented Ok-Robot [3] on our mobile robot as a comparison method. Additionally, we compared our approach with Ok-Robot and ConceptGraphs [14] in terms of memory usage and scene update time.
3）基线：为展示我们方法对变化环境的适应性以及场景图预测的有效性，我们评估了场景变化与场景图预测。以 GPT-40 作为基础模型，并增强链式思维（CoT）推理[35]，其方式类似于 Jiang 等人提出的方法[15]：模型接收来自机器人的 RGB 观测，将其与记忆中最相似的图像匹配，检测差异，并基于新的 RGB 观测预测场景图。对于长期任务执行，我们在移动机器人上实现了 Ok-Robot[3] 作为对比方法。此外，我们还从内存使用和场景更新耗时两个方面，将我们的方法与 Ok-Robot 和 ConceptGraphs[14] 进行了比较。


4) Evaluation: To thoroughly evaluate DovSG's adaptability to dynamic environments and its effectiveness in long-term tasks, we involved human evaluators in modifying the environment, constructing ground truth (GT) scene graphs, and determining task execution success. Our evaluation focuses on three key aspects: Dynamic Scene Adaptation and Scene Graph Generation: We recorded RGB observations during the robot's task execution and compared DOvSG's performance with baseline methods in detecting scene changes and generating scene graphs. Long-Term Tasks Evaluation: In addition to assessing the overall success rate of long-term tasks, we evaluated the success rates of individual subtasks such as picking, placing, and navigation. This detailed evaluation provides a comprehensive analysis of the effectiveness of our method across different components of task execution. The three main metrics used for evaluation are as follows: (1) Scene Change Detection Accuracy (SCDA): This metric measures the accuracy of detecting changes in the observed scene and matching them to the corresponding memory-stored RGB information. (2) Scene Graph Accuracy (SGA): This metric evaluates whether the generated scene graph matches the GT scene graph and only recognizes three types of relations: "on", "belong", and "inside". If the final scene graph perfectly matches the GT version, the experiment is considered successful and assigned a score of 1, otherwise 0. (3) Task Success Rate: This metric represents the overall task completion success rate. For long-term tasks, success is only counted if all subtasks are completed successfully.
4）评估：为全面评估 DovSG 对动态环境的适应性以及在长期任务中的有效性，我们引入人工评测者来修改环境、构建真实场景图（GT），并判断任务执行是否成功。我们的评估聚焦三个关键方面：动态场景适应与场景图生成：我们在机器人执行任务期间记录 RGB 观测，并将 DOvSG 与基线方法在检测场景变化与生成场景图方面的表现进行对比。长期任务评估：除评估长期任务的整体成功率外，我们还评估了诸如拾取、放置和导航等各个子任务的成功率。该细粒度评估能够对我们方法在任务执行不同组成部分上的有效性进行全面分析。用于评估的三个主要指标如下：（1）场景变化检测准确率（SCDA）：该指标衡量检测到的变化与对应的记忆中 RGB 信息匹配的准确性。（2）场景图准确率（SGA）：该指标评估生成的场景图是否与 GT 场景图一致，并且仅识别三类关系：“on”、“belong”和“inside”。如果最终场景图与 GT 版本完全匹配，则实验视为成功并赋值为1，否则为0。（3）任务成功率：该指标表示整体任务完成成功率。对于长期任务，只有当所有子任务都成功完成时，才算成功。


TABLE I
表 I


QUANTITATIVE RESULTS ON DYNAMIC SCENE ADAPTATION AND SCENE GRAPH GENERATION
动态场景适应与场景图生成的定量结果


<table><tr><td>Task</td><td colspan="2">Minor Adjustment</td><td colspan="2">Appearance</td><td colspan="2">Positional Shift</td></tr><tr><td>Metrics</td><td>GPT-40</td><td>Ours</td><td>GPT-40</td><td>Ours</td><td>GPT-40</td><td>Ours</td></tr><tr><td>SCDA</td><td>41.44%</td><td>95.37%</td><td>64.25%</td><td>93.22%</td><td>66.35%</td><td>94.23%</td></tr><tr><td>SGA</td><td>54.60%</td><td>88.75%</td><td>52.25%</td><td>84.86%</td><td>46.18%</td><td>$\mathbf{{83.72}\% }$</td></tr></table>
<table><tbody><tr><td>任务</td><td colspan="2">微调</td><td colspan="2">外观</td><td colspan="2">位置偏移</td></tr><tr><td>指标</td><td>GPT-40</td><td>我们</td><td>GPT-40</td><td>我们</td><td>GPT-40</td><td>我们</td></tr><tr><td>SCDA</td><td>41.44%</td><td>95.37%</td><td>64.25%</td><td>93.22%</td><td>66.35%</td><td>94.23%</td></tr><tr><td>SGA</td><td>54.60%</td><td>88.75%</td><td>52.25%</td><td>84.86%</td><td>46.18%</td><td>$\mathbf{{83.72}\% }$</td></tr></tbody></table>


5) Results: As shown in Table I, in the SCDA evaluation, GPT-40 often struggles with inconsistencies between the scope of historical observations and new observations, which introduces a significant amount of noise in the responses, leading to poor performance. This inconsistency also makes accurate detection under "Minor Adjustment" scenarios very challenging. In contrast, DovSG, supported by precise re-localization, can accurately identify the voxel index where changes have occurred in the scene, significantly outperforming the baseline. Specifically, in "Minor Adjustment" scenarios, DovSG exceeds the baseline by nearly 54%, primarily because GPT-40 struggles to recognize smaller positional changes. Additionally, in "Appearance" and "Positional Shift" scenarios, DovSG achieves a scene change recognition success rate approximately ${28}\%$ higher than the GPT-40. In the SGA evaluation, both approaches generate scene graphs based on new observations, with SGA accuracy in "Minor Adjustment" scenarios being higher than in "Appearance" and "Position Shift" scenarios. The main issue with our method arises when the same object is decomposed into multiple nodes, affecting the scene graph's accuracy. On the other hand, GPT-40 struggles to accurately recognize relationships or redundantly defines objects, leading to erroneous results.
5）结果：如表 I 所示，在 SCDA 评估中，GPT-40 往往难以处理历史观测范围与新观测之间的不一致，这会在响应中引入大量噪声，导致性能较差。这种不一致也使得在“Minor Adjustment”场景下进行准确检测极具挑战。相比之下，DovSG 借助精确重定位，能够准确识别场景中发生变化的体素索引，显著优于基线。具体而言，在“Minor Adjustment”场景下，DovSG 比基线高出近 54%，这主要是因为 GPT-40 难以识别较小的位置变化。此外，在“Appearance”和“Positional Shift”场景中，DovSG 的场景变化识别成功率约比 GPT-40 高 ${28}\%$。在 SGA 评估中，两种方法都基于新观测生成场景图，其中“Minor Adjustment”场景下的 SGA 准确率高于“Appearance”和“Position Shift”场景。我们方法的主要问题出现在同一对象被拆分为多个节点时，这会影响场景图的准确性。另一方面，GPT-40 在准确识别关系或重复定义对象方面表现不佳，导致错误结果。


In Table II, we have the fellow observations: (1) In the "Minor Adjustment" environment, although the object is slightly moved, it remains within the robot's field of view. This makes it highly likely for the robot to navigate near the target, resulting in a significantly higher success rate compared to "Appearance" and "Positional Shift" (in 80 trials, it achieved 5 and 10 more successes, respectively). In the "Positional Shift" scenario, the residual effect of CLIP features can occasionally mislead the robot into navigating toward the object's historical location, ultimately causing navigation failure. In contrast, for "Appearance", where a new object emerges, the robot does not face the challenge of misjudging the original object's position, generally leading to a higher success rate than "Positional Shift" (with 5 more successes out of 80 trials). (2) DovSG demonstrates significantly enhanced manipulation capabilities over Ok-Robot. For the "Pick up" task, compared to Ok-Robot, DOVSG employs two proposed grasp strategies that focus specifically on the target object. By optimizing the selection of grasp candidates and integrating a heuristic grasping method, DovSG reduces environmental interference and ensures the robot selects the optimal grasp. This results in a 10.7% higher pick-up success rate than Ok-Robot, which relies solely on AnyGrasp. Regarding "Place" task, DovSG utilizes a lower placement position and an inclined placement method (see our https://bjhyzj.github.io/dovsg-web project page for details), achieving an overall success rate that is 5.49% higher than Ok-Robot. (3) In dynamic environments, DovSG significantly outperforms Ok-Robot (which assumes a static scene) in long-term tasks, thanks to its ability to adapt to scene changes. Although Ok-Robot can occasionally succeed in locating the correct object under minor changes (e.g., "Minor Adjustment"), it struggles with larger modifications such as "Appearance" or "Positional Shift" because it cannot partially update its scene representation-making success in these scenarios nearly impossible. As a result, Ok-Robot's success rate for long-term tasks in dynamic environments is approximately 30% lower than DOVSG.
在表 II 中，我们有如下观察：(1) 在“Minor Adjustment”环境中，尽管对象略有移动，但仍位于机器人视野内。这使机器人极有可能靠近目标导航，因此其成功率显著高于“Appearance”和“Positional Shift”（在 80 次试验中，分别多成功 5 次和 10 次）。“Positional Shift”场景下，CLIP 特征的残余效应有时会误导机器人导航到对象的历史位置，最终导致导航失败。相比之下，在“Appearance”场景中，新的对象出现后，机器人不再面临误判原始对象位置的挑战，因此通常比“Positional Shift”有更高的成功率（在 80 次试验中多成功 5 次）。（2）DovSG 的操作能力显著优于 Ok-Robot。对于“Pick up”任务，与 Ok-Robot 相比，DOVSG 采用了两种专门针对目标对象的抓取策略。通过优化抓取候选的选择并结合启发式抓取方法，DovSG 减少了环境干扰，并确保机器人选择最优抓取。因此，其拾取成功率比仅依赖 AnyGrasp 的 Ok-Robot 高 10.7%。在“Place”任务中，DovSG 采用更低的放置位置和倾斜放置方法（详情见我们的 https://bjhyzj.github.io/dovsg-web 项目页面），总体成功率比 Ok-Robot 高 5.49%。（3）在动态环境中，DovSG 依托其适应场景变化的能力，在长期任务上显著优于假设场景静止的 Ok-Robot。尽管 Ok-Robot 在轻微变化下（如“Minor Adjustment”）有时也能成功定位到正确对象，但在“Appearance”或“Positional Shift”等更大幅度的变化下则表现不佳，因为它无法对场景表示进行部分更新——使得这些场景下的成功几乎不可能。因此，Ok-Robot 在动态环境中的长期任务成功率约比 DOVSG 低 30%。


TABLE II
表 II


SUCCESS RATE OF LONG-TERM TASKS AND SUBTASKS
长期任务与子任务成功率


<table><tr><td>Task</td><td>Method</td><td>Minor Adjustment</td><td>Appearance</td><td>Positional Shift</td><td>Total(%)</td></tr><tr><td rowspan="2">Pick up</td><td>Ok-Robot</td><td>84 / 110</td><td>61 / 92</td><td>59 / 87</td><td>70.58%</td></tr><tr><td>Ours</td><td>111 / 137</td><td>111 / 136</td><td>108 / 133</td><td>81.28%</td></tr><tr><td rowspan="2">Place</td><td>Ok-Robot</td><td>53 / 64</td><td>40 / 51</td><td>42 / 51</td><td>81.32%</td></tr><tr><td>Ours</td><td>80 / 93</td><td>83 / 95</td><td>74 / 85</td><td>86.81%</td></tr><tr><td rowspan="2">Navigation</td><td>Ok-Robot</td><td>179 / 210</td><td>146 / 184</td><td>137 / 180</td><td>80.48%</td></tr><tr><td>Ours</td><td>228 / 236</td><td>239 / 254</td><td>224 / 245</td><td>94.01%</td></tr><tr><td rowspan="2">Long-term</td><td>Ok-Robot</td><td>12 / 80</td><td>0 / 80</td><td>0 / 80</td><td>5.00%</td></tr><tr><td>Ours</td><td>33 / 80</td><td>28 / 80</td><td>23 / 80</td><td>35.00%</td></tr></table>
<table><tbody><tr><td>任务</td><td>方法</td><td>微调</td><td>外观</td><td>位置偏移</td><td>总计(%)</td></tr><tr><td rowspan="2">拿起</td><td>Ok-Robot</td><td>84 / 110</td><td>61 / 92</td><td>59 / 87</td><td>70.58%</td></tr><tr><td>我们的方法</td><td>111 / 137</td><td>111 / 136</td><td>108 / 133</td><td>81.28%</td></tr><tr><td rowspan="2">放下</td><td>Ok-Robot</td><td>53 / 64</td><td>40 / 51</td><td>42 / 51</td><td>81.32%</td></tr><tr><td>我们的方法</td><td>80 / 93</td><td>83 / 95</td><td>74 / 85</td><td>86.81%</td></tr><tr><td rowspan="2">导航</td><td>Ok-Robot</td><td>179 / 210</td><td>146 / 184</td><td>137 / 180</td><td>80.48%</td></tr><tr><td>我们的方法</td><td>228 / 236</td><td>239 / 254</td><td>224 / 245</td><td>94.01%</td></tr><tr><td rowspan="2">长期</td><td>Ok-Robot</td><td>12 / 80</td><td>0 / 80</td><td>0 / 80</td><td>5.00%</td></tr><tr><td>我们的方法</td><td>33 / 80</td><td>28 / 80</td><td>23 / 80</td><td>35.00%</td></tr></tbody></table>


TABLE III
表 III


EFFIECIENCY COMPARISON BETWEEN OK-ROBOT AND DOvSG
OK-ROBOT 与 DovSG 的效率比较


<table><tr><td>Method $\left( {\mathrm{m}}^{2}\right)$</td><td>Memory (GB) $\downarrow$</td><td>Time (min) $\downarrow$</td></tr><tr><td>Ok-Robot</td><td>2</td><td>20</td></tr><tr><td>ConceptGraphs</td><td>0.15</td><td>27</td></tr><tr><td>Ours</td><td>0.15</td><td>1</td></tr></table>
<table><tbody><tr><td>方法 $\left( {\mathrm{m}}^{2}\right)$</td><td>内存 (GB) $\downarrow$</td><td>时间 (分钟) $\downarrow$</td></tr><tr><td>Ok-Robot</td><td>2</td><td>20</td></tr><tr><td>ConceptGraphs</td><td>0.15</td><td>27</td></tr><tr><td>我们</td><td>0.15</td><td>1</td></tr></tbody></table>


Table III further demonstrates the memory consumption required to store memory at a $1\mathrm{\;{cm}}$ resolution and the update times for DovSG, Ok-Robot, and ConceptGraphs after processing 1200 frames of RGBD data within a ${40}{\mathrm{\;m}}^{2}$ scene. It is important to note that since Ok-Robot and ConceptGraphs cannot perform partial updates, the table only presents the time required for a complete scene update. The results show that Ok-Robot consumes approximately 13 times more memory than DovSG and has update times 20 times longer. While ConceptGraphs and DovSG have similar memory usage due to both using 3D scene graphs, DovSG's ability to perform local updates allows it to achieve update times 27 times faster than ConceptGraphs.
表 III 进一步展示了在 $1\mathrm{\;{cm}}$ 分辨率下存储记忆所需的内存消耗，以及在一个 ${40}{\mathrm{\;m}}^{2}$ 场景中处理 1200 帧 RGBD 数据后，DovSG、Ok-Robot 和 ConceptGraphs 的更新时间。需要注意的是，由于 Ok-Robot 和 ConceptGraphs 不能进行局部更新，表中仅给出了完整场景更新所需的时间。结果表明，Ok-Robot 的内存消耗约为 DovSG 的 13 倍，更新时间长 20 倍。尽管 ConceptGraphs 和 DovSG 由于都使用 3D 场景图而具有相近的内存占用，但 DovSG 能够进行局部更新，使其更新速度比 ConceptGraphs 快 27 倍。


## V. LIMITATION AND FUTURE WORK
## V. 限制与未来工作


Limitation: The performance of DovSG's visual relocaliza-tion depends on the presence of distinctive visual cues (e.g., textures, edges, objects) for key point matching. In environments with sparse, minimal, or repetitive features, the system may struggle to extract enough distinctive points, impacting localization accuracy.
限制：DovSG 的视觉重定位性能取决于关键点匹配所需的显著视觉线索（例如纹理、边缘、物体）的存在。在特征稀疏、极简或重复的环境中，系统可能难以提取足够的显著点，从而影响定位精度。


Future work: Exploring real-time precise localization through multi-sensor fusion, efficient memory update mechanisms, and developing lightweight methods for scene representation is crucial. Additionally, enabling efficient collaboration between mobile robots and manipulators will be key to making mobile manipulation more practical in real-world scenarios.
未来工作：通过多传感器融合实现实时精确定位，研究高效的记忆更新机制，并开发用于场景表示的轻量化方法至关重要。此外，促成移动机器人与机械臂之间的高效协作，将是让移动操作在真实场景中更具可行性的关键。


## VI. CONCLUSION
## VI. 结论


In this letter, we introduce DovSG, an innovative framework designed to enable mobile robots to perform long-term tasks in dynamic environments by continuously local updates of 3D scene graphs. DovSG enables the robot to adapt to changes caused by human interaction or by the robot's own task execution. These capabilities ensure the precise execution of long-term tasks without being affected by cumulative temporal errors. By evaluating long-term tasks and subtasks such as pickup, place, and navigation, our results highlight the robustness and effectiveness of DovSG in handling the complexity of dynamic real-world scenes. Compared to methods that assume static environments and cannot adjust to scene changes, DovSG achieves a 30% higher success rate in long-term tasks, accelerates updates by up to 27 times in typical room scenarios, and reduces memory consumption by a factor of 13.
在这封信中，我们介绍了DovSG，这一创新框架旨在通过持续局部更新3D场景图，使移动机器人能够在动态环境中执行长期任务。DovSG使机器人能够适应由人类交互或机器人自身任务执行所引起的变化。这些能力确保了长期任务的精确执行，而不会受到累积时间误差的影响。通过对拾取、放置和导航等长期任务及子任务的评估，我们的结果突显了DovSG在应对动态现实场景复杂性方面的鲁棒性和有效性。与假设环境静态且无法适应场景变化的方法相比，DovSG在长期任务中的成功率提高了30%，在典型房间场景中的更新速度最高提升了27倍，并将内存消耗降低了13倍。


## REFERENCES
## 参考文献


[1] A. Brohan et al., "RT-2: Vision-language-action models transfer web knowledge to robotic control," 2023, arXiv: 2307.15818.
[1] A. Brohan 等，“RT-2：视觉-语言-行动模型将网络知识迁移到机器人控制”，2023，arXiv：2307.15818。


[2] D. Driess et al., "PALM-E: An embodied multimodal language model," 2023, arXiv: 2303.03378.
[2] D. Driess 等，“PALM-E：具身多模态语言模型”，2023，arXiv：2303.03378。


[3] P. Liu et al., "OK-Robot: What really matters in integrating open-knowledge models for robotics," 2024, arXiv:2401.12202.
[3] P. Liu 等，“OK-Robot：将开放知识模型整合到机器人领域时真正重要的是什么”，2024，arXiv：2401.12202。


[4] T. Gilles, S. Sabatini, D. Tsishkou, B. Stanciulescu, and F. Moutarde, "GO-HOME: Graph-oriented heatmap output for future motion estimation," in Proc. Int. Conf. Robot. Automat., IEEE, 2022, pp. 9107-9114.
[4] T. Gilles，S. Sabatini，D. Tsishkou，B. Stanciulescu 和 F. Moutarde，“GO-HOME：面向未来运动估计的基于图的热力图输出”，载于：机器人自动化国际会议论文集，IEEE，2022，pp. 9107-9114。


[5] Z. Wang et al., "Towards cognitive intelligence-enabled product design: The evolution, state-of-the-art, and future of AI-enabled product design," J. Ind. Inf. Integration, vol. 43, 2025, Art. no. 100759.
[5] Z. Wang 等，“面向具备认知智能的产品设计：AI 赋能产品设计的发展、现状与未来”，《工业信息融合学报》，第 43 卷，2025，论文号 100759。


[6] Y. Zhang et al., "Recognize anything: A strong image tagging model," 2023, arXiv: 2306.03514.
[6] Y. Zhang 等，“识别任何事物：一种强大的图像标注模型”，2023，arXiv：2306.03514。


[7] S. Liu et al., "Grounding DINO: Marrying DINO with grounded pretraining for open-set object detection," 2024, arXiv: 2303.05499.
[7] S. Liu 等，“Grounding DINO：将 DINO 与基于真实标注的预训练结合以实现开放集目标检测”，2024，arXiv：2303.05499。


[8] N. Ravi et al., "SAM 2: Segment anything in images and videos," 2024, arXiv: 2408.00714.
[8] N. Ravi 等，“SAM 2：图像与视频中的分割任意内容”，2024，arXiv：2408.00714。


[9] A. Radford et al., "Learning transferable visual models from natural language supervision," 2021, arXiv: 2103.00020.
[9] A. Radford 等，“从自然语言监督中学习可迁移的视觉模型”，2021，arXiv：2103.00020。


[10] W. Shen, G. Yang, A. Yu, J. Wong, L. P. Kaelbling, and P. Isola, "Distilled feature fields enable few-shot language-guided manipulation," 2023, arXiv: 2308.07931.
[10] W. Shen，G. Yang，A. Yu，J. Wong，L. P. Kaelbling 和 P. Isola，“蒸馏特征场支持少样本、基于语言的引导操作”，2023，arXiv：2308.07931。


[11] J. Kerr, C. M. Kim, K. Goldberg, A. Kanazawa, and M. Tancik, "LERF: Language embedded radiance fields," in Proc. IEEE/CVF Int. Conf. Com-put. Vis., 2023, pp. 19729-19739.
[11] J. Kerr，C. M. Kim，K. Goldberg，A. Kanazawa 和 M. Tancik，“LERF：语言嵌入辐射场”，载于：IEEE/CVF 计算机视觉国际会议论文集，2023，pp. 19729-19739。


[12] Y. Zheng et al., "GaussianGrasper: 3D language Gaussian splatting for open-vocabulary robotic grasping," IEEE Robot. Automat. Lett., vol. 9, no. 9, pp. 7827-7834, Sep. 2024.
[12] Y. Zheng 等，“GaussianGrasper：用于开放词汇机器人抓取的 3D 语言高斯泼洒”，《IEEE 机器人与自动化快报》，第 9 卷，第 9 期，pp. 7827-7834，2024 年 9 月。


[13] Z. Yan, Z. Wang, S. Li, M. Li, X. Liang, and J. Liu, "ManufVisSGG: A vision-language-model approach for cognitive scene graph generation in manufacturing systems," in Proc. IEEE 20th Int. Conf. Automat. Sci. Eng.. IEEE, 2024, pp. 1632-1637.
[13] Z. Yan，Z. Wang，S. Li，M. Li，X. Liang 和 J. Liu，“ManufVisSGG：一种面向制造系统中认知场景图生成的视觉-语言模型方法”，载于：IEEE 第 20 届自动化科学与工程国际会议论文集，IEEE，2024，pp. 1632-1637。


[14] Q. Gu et al., "ConceptGraphs: Open-vocabulary 3D scene graphs for perception and planning," in Proc. IEEE Int. Conf. Robot. Automat., 2024, pp. 5021-5028.
[14] Q. Gu 等，“ConceptGraphs：用于感知与规划的开放词汇 3D 场景图”，载于：IEEE 机器人与自动化国际会议论文集，2024，pp. 5021-5028。


[15] H. Jiang et al., "RoboEXP: Action-conditioned scene graph via interactive exploration for robotic manipulation," 2024, arXiv: 2402.15487 .
[15] H. Jiang 等，“RoboEXP：通过交互式探索实现面向动作条件的场景图用于机器人操作”，2024，arXiv：2402.15487 。


[16] P. Liu et al., "DynaMem: Online dynamic spatio-semantic memory for open world mobile manipulation," 2024, arXiv:2411.04999.
[16] P. Liu 等人，《DynaMem：面向开放世界移动操作的在线动态时空语义记忆》，2024，arXiv:2411.04999。


[17] S. Peng, K. Genova, C. M. Jiang, A. Tagliasacchi, M. Pollefeys, and T. Funkhouser, "Openscene: 3D scene understanding with open vocabularies," in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2023, pp. 815-824.
[17] S. Peng、K. Genova、C. M. Jiang、A. Tagliasacchi、M. Pollefeys 和 T. Funkhouser，《Openscene：具有开放词汇的3D场景理解》，载于：IEEE/CVF计算机视觉与模式识别会议论文集，2023，页815-824。


[18] S. Li, P. Zheng, J. Fan, and L. Wang, "Toward proactive human-robot collaborative assembly: A multimodal transfer-learning-enabled action prediction approach," IEEE Trans. Ind. Electron., vol. 69, no. 8, pp. 8579-8588, Aug. 2022.
[18] S. Li、P. Zheng、J. Fan 和 L. Wang，《面向主动式人机协作装配：一种基于多模态迁移学习的动作预测方法》，《IEEE工业电子学汇刊》，第69卷，第8期，页8579-8588，2022年8月。


[19] S. Li et al., "Proactive human-robot collaboration: Mutual-cognitive, predictable, and self-organising perspectives," Robot. Comput.- Integr. Manuf., vol. 81, 2023, Art. no. 102510.
[19] S. Li 等人，《主动式人机协作：互认知、可预测和自组织视角》，《机器人与计算机集成制造》，第81卷，2023，文章号102510。


[20] R. Wu, K. Xu, C. Liu, N. Zhuang, and Y. Mu, "Localize, assemble, and predicate: Contextual object proposal embedding for visual relation detection," in Proc. Conf. Assoc. Advance. Artif. Intell., 2020, vol. 34, no. 7, pp. 12297-12304.
[20] R. Wu、K. Xu、C. Liu、N. Zhuang 和 Y. Mu，《定位、装配与预测：用于视觉关系检测的上下文目标提议嵌入》，载于：AAAI（美国人工智能协会）会议论文集，2020，第34卷，第7期，页12297-12304。


[21] A. Werby, C. Huang, M. Büchner, A. Valada, and W. Burgard, "Hierarchical open-vocabulary 3D scene graphs for language-grounded robot navigation," in Proc. Robot.: Sci. Syst., Delft, Netherlands, Jul. 2024, doi: 10.15607/RSS.2024.XX.077.
[21] A. Werby、C. Huang、M. Büchner、A. Valada 和 W. Burgard，《面向语言引导机器人导航的分层开放词汇3D场景图》，载于：Robot.: Sci. Syst.（机器人：科学系统），荷兰代尔夫特，2024年7月，doi: 10.15607/RSS.2024.XX.077。


[22] OpenAI, "GPT-4 technical report," 2024. [Online]. Available: https://arxiv.org/abs/2303.08774
[22] OpenAI，《GPT-4技术报告》，2024。[在线] 可得：https://arxiv.org/abs/2303.08774。


[23] G. T. et al, "Gemini: A family of highly capable multimodal models," 2024. [Online]. Available: https://arxiv.org/abs/2312.11805
[23] G. T. 等人，《Gemini：一族高度胜任的多模态模型》，2024。[在线] 可得：https://arxiv.org/abs/2312.11805。


[24] Z. Yan et al., "INT2: Interactive trajectory prediction at intersections," in Proc. Int. Conf. Comput. Vis., Oct. 2023, pp. 8536-8547.
[24] Z. Yan 等人，《INT2：交叉口的交互式轨迹预测》，载于：国际计算机视觉会议论文集，2023年10月，页8536-8547。


[25] X. Zheng et al., "Large language models powered context-aware motion prediction in autonomous driving," 2024, arXiv:2403.11057.
[25] X. Zheng 等人，《由大语言模型驱动的自动驾驶情境感知运动预测》，2024，arXiv:2403.11057。


[26] P. Lindenberger, P.-E. Sarlin, and M. Pollefeys, "LightGlue: Local feature matching at light speed," in Proc. Int. Conf. Comput. Vis., 2023, pp. 17627-17638.
[26] P. Lindenberger、P.-E. Sarlin 和 M. Pollefeys，《LightGlue：以光速进行局部特征匹配》，载于：国际计算机视觉会议论文集，2023，页17627-17638。


[27] Q. Wang, J. Zhang, K. Yang, K. Peng, and R. Stiefelhagen, "Matchformer: Interleaving attention in transformers for feature matching," in Proc. Asian Conf. Comput. Vis., 2022, pp. 2746-2762.
[27] Q. Wang、J. Zhang、K. Yang、K. Peng 和 R. Stiefelhagen，《Matchformer：用于特征匹配的Transformer中交错注意力》，载于：亚洲计算机视觉会议论文集，2022，页2746-2762。


[28] E. Brachmann, T. Cavallari, and V. A. Prisacariu, "Accelerated coordinate encoding: Learning to relocalize in minutes using RGB and poses," in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2023, pp. 5044-5053.
[28] E. Brachmann、T. Cavallari 和 V. A. Prisacariu，《加速坐标编码：利用RGB和位姿在几分钟内重新定位》，载于：IEEE/CVF计算机视觉与模式识别会议论文集，2023，页5044-5053。


[29] E. Brachmann and C. Rother, "Visual camera re-localization from RGB and RGB-D images using DSAC," IEEE Trans. Pattern Anal. Mach. Intell., vol. 44, no. 9, pp. 5847-5865, Sep. 2022.
[29] E. Brachmann 和 C. Rother，《基于RGB与RGB-D图像的视觉相机重新定位：使用DSAC》，《IEEE模式分析与机器智能汇刊》，第44卷，第9期，页5847-5865，2022年9月。


[30] P. J. Besl and N. D. McKay, "Method for registration of 3-D shapes," IEEE Trans. Pattern Analy. Mach. Intell., vol. 14, no. 2, pp. 239-256, 1992.
[30] P. J. Besl 和 N. D. McKay，《3D形状配准方法》，《IEEE模式分析与机器智能汇刊》，第14卷，第2期，页239-256，1992年。


[31] Z. Teed and J. Deng, "DROID-SLAM: Deep visual SLAM for monocular, stereo, and RGB-D cameras," NeurIPS, pp. 16558-16569, 2021.
[31] Z. Teed 和 J. Deng，《DROID-SLAM：面向单目、双目和RGB-D相机的深度视觉SLAM》，NeurIPS，页16558-16569，2021年。


[32] R. Mur-Artal and J. D. Tardós, "ORB-SLAM2: An open-source SLAM system for monocular, stereo and RGB-D cameras," IEEE Trans. Robot., vol. 33, no. 5, pp. 1255-1262, Oct. 2017.
[32] R. Mur-Artal 和 J. D. Tardós，“ORB-SLAM2：一种用于单目、双目和 RGB-D 相机的开源 SLAM 系统”，IEEE Transactions on Robotics，第 33 卷，第 5 期，第 1255-1262 页，2017 年 10 月。


[33] Z. Zhu et al., "NICE-SLAM: Neural implicit scalable encoding for SLAM," in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2022, pp. 12786-12796.
[33] Z. Zhu 等，“NICE-SLAM：用于 SLAM 的神经隐式可扩展编码”，载于：IEEE/CVF 计算机视觉与模式识别会议论文集，2022 年，第 12786-12796 页。


[34] P. E. Hart, N. J. Nilsson, and B. Raphael, "A formal basis for the heuristic determination of minimum cost paths," IEEE Trans. Syst. Sci. Cybern., vol. SSC-4, no. 2, pp. 100-107, Jul. 1968.
[34] P. E. Hart、N. J. Nilsson 和 B. Raphael，“启发式确定最小代价路径的形式化基础”，IEEE Transactions on Systems Science and Cybernetics，第 SSC-4 卷，第 2 期，第 100-107 页，1968 年 7 月。


[35] J. Wei et al., "Chain-of-thought prompting elicits reasoning in large language models," in Proc. 36th Int. Conf. Neural Inf. Process. Syst., 2023, pp. 24824-24837.
[35] J. Wei 等，“思维链式提示能激发大型语言模型的推理”，载于：第 36 届神经信息处理系统国际会议论文集，2023 年，第 24824-24837 页。