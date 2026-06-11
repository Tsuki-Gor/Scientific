# Open-Vocabulary Functional 3D Scene Graphs for Real-World Indoor Spaces
# 开放词汇功能性三维场景图用于真实室内空间


Chenyangguang Zhang ${}^{1,2}$ Alexandros Delitzas ${}^{2,3}$ Fangjinhua Wang ${}^{2}$ Ruida Zhang ${}^{1}$
张晨阳光 ${}^{1,2}$ 亚历山德罗斯·德利察斯 ${}^{2,3}$ 王芳静华 ${}^{2}$ 张瑞达 ${}^{1}$


Xiangyang Ji ${}^{1}$ Marc Pollefeys ${}^{2,4}$ Francis Engelmann ${}^{2,5}$
姬向阳 ${}^{1}$ 马克·波勒费斯 ${}^{2,4}$ 弗朗西斯·恩格尔曼 ${}^{2,5}$


${}^{1}$ Tsinghua University ${}^{2}$ ETH Zürich ${}^{3}$ MPI for Informatics ${}^{4}$ Microsoft ${}^{5}$ Stanford University
${}^{1}$ 清华大学 ${}^{2}$ ETH苏黎世 ${}^{3}$ 信息学马克斯·普朗克研究所 ${}^{4}$ 微软 ${}^{5}$ 斯坦福大学


## Abstract
## 摘要


We introduce the task of predicting functional 3D scene graphs for real-world indoor environments from posed RGB-D images. Unlike traditional 3D scene graphs that focus on spatial relationships of objects, functional 3D scene graphs capture objects, interactive elements, and their functional relationships. Due to the lack of training data, we leverage foundation models, including visual language models (VLMs) and large language models (LLMs), to encode functional knowledge. We evaluate our approach on an extended SceneFun3D dataset and a newly collected dataset, FunGraph3D, both annotated with functional 3D scene graphs. Our method significantly outperforms adapted baselines, including Open3DSG and ConceptGraph, demonstrating its effectiveness in modeling complex scene functionalities. We also demonstrate downstream applications such as 3D question answering and robotic manipulation using functional 3D scene graphs. See our project page at https://openfungraph.github.io.
我们提出了一个任务：从带位姿的 RGB-D 图像中预测真实室内环境的功能性 3D 场景图。不同于侧重物体空间关系的传统 3D 场景图，功能性 3D 场景图刻画物体、交互元素及其功能关系。由于缺乏训练数据，我们利用基础模型，包括视觉语言模型（VLMs）和大语言模型（LLMs），来编码功能知识。我们在扩展的 SceneFun3D 数据集和新收集的 FunGraph3D 数据集上评估了我们的方法，这两个数据集都标注了功能性 3D 场景图。我们的方法显著优于改进后的基线方法，包括 Open3DSG 和 ConceptGraph，展示了其在建模复杂场景功能方面的有效性。我们还展示了功能性 3D 场景图在 3D 问答和机器人操作等下游应用中的作用。项目页面见 https://openfungraph.github.io。


## 1. Introduction
## 1. 引言


This paper introduces functional 3D scene graphs for real-world indoor spaces from posed RGB-D images. 3D scene graphs offer a lightweight, abstract representation for capturing the comprehensive semantic structure of an environment [4]. They support a variety of applications, including 3D scene alignment [66], image localization [51], graph-conditioned 3D scene generation [21, 97], as well as robotics navigation [83] and task planning [2, 61].
本文从带位姿的RGB-D图像中，为真实室内空间引入功能性3D场景图。3D场景图提供了一种轻量、抽象的表示，用于捕获环境的完整语义结构[4]。它们支持多种应用，包括3D场景对齐[66]、图像定位[51]、图条件3D场景生成[21, 97]，以及机器人导航[83]和任务规划[2, 61]。


Recent advances in 3D scene graph prediction [4, 11, 27, 40, 41, 63, 64, 78, 84], have enabled exciting developments across multiple areas, including scene graph inference from 3D reconstructions [11, 78], applications for robotic interactions [27, 84], online scene graph generation [84], open-vocabulary 3D scene graphs [40, 41] and large-scale, hierarchical scene graphs [4, 63, 64]. The performance of recent scene graph methods also benefits from advancements in 3D scene understanding techniques [14, 57, 70], which they rely on to extract objects and their semantics for modeling inter-object relationships. However, existing 3D scene graph estimation methods [27, 40, 78, 84] face important limitations: graph nodes are typically restricted to objects, and edges represent only spatial relationships. For instance, edges primarily capture relative positions, such as 'the TV is mounted on the wall' or 'the flower is placed on the table'-information already implicitly encoded by object positions. Crucially, these methods lack representations of small interactive elements [17] and their functional relationships with other scene objects, which are essential for finer-grained interactions (e.g., flipping a switch to turn on a light), making them less suitable for higher-level functional reasoning. The key idea of this paper is to enhance 3D scene graphs with the capability to represent functional relationships between objects and their interactive elements. A 3D scene graph that captures both functionalities and interactions opens up significant opportunities. For example, robotic agents can identify interactive elements and their functional relationships with objects to perform effective manipulation tasks, or graph-guided 3D scene generation methods [21, 97] can, with this enriched representation, generate more dynamic and realistic environments by incorporating interactive elements and their effects. However, creating functional 3D scene graphs is challenging. Most importantly, there is a lack of training data to learn the complex functional relationships between objects and their interactive elements. Unlike existing 3D scene graphs, functional 3D scene graphs require a more nuanced understanding of interactions and object affordances. To address this, our approach implements an open-vocabulary pipeline for functional 3D scene graph inference, termed OpenFunGraph, leveraging the extensive knowledge encoded within foundation models, including visual language models (VLM) and large language models (LLM). These models, pre-trained on vast amounts of multimodal data, include rich semantic information that can potentially be adapted for functional understanding. This leads us to the central question of this work: "Can we harness foundation models to construct functional 3D scene graphs?"
近年来3D场景图预测[4, 11, 27, 40, 41, 63, 64, 78, 84]的进展，推动了多个方向的令人振奋的发展，包括从3D重建中推断场景图[11, 78]、用于机器人交互的应用[27, 84]、在线场景图生成[84]、开放词汇3D场景图[40, 41]以及大规模层次化场景图[4, 63, 64]。近期场景图方法的性能也受益于3D场景理解技术[14, 57, 70]的进步，因为它们依赖这些技术来提取物体及其语义，以建模物体间关系。然而，现有3D场景图估计方法[27, 40, 78, 84]仍面临重要局限：图节点通常仅限于物体，边也只表示空间关系。例如，边主要捕捉相对位置，如“电视挂在墙上”或“花放在桌子上”——这些信息已由物体位置隐式编码。关键在于，这些方法缺少对小型交互元素[17]及其与其他场景物体之间功能关系的表示，而这对于更细粒度的交互（如拨动开关以打开灯）至关重要，因此不适用于更高层次的功能推理。本文的核心思想是增强3D场景图，使其能够表示物体与其交互元素之间的功能关系。捕获功能与交互的3D场景图带来了重要机遇。例如，机器人代理可以识别交互元素及其与物体的功能关系，以执行有效的操作任务；或者，图引导的3D场景生成方法[21, 97]可以借助这种更丰富的表示，通过纳入交互元素及其作用，生成更动态、更真实的环境。然而，构建功能性3D场景图具有挑战性。最重要的是，缺乏可用于学习物体与其交互元素之间复杂功能关系的训练数据。不同于现有3D场景图，功能性3D场景图需要对交互和物体可供性有更细致的理解。为此，我们的方法实现了一个用于功能性3D场景图推断的开放词汇流程，称为OpenFunGraph，借助基础模型中编码的广泛知识，包括视觉语言模型（VLM）和大语言模型（LLM）。这些模型在海量多模态数据上预训练，包含丰富的语义信息，可潜在地迁移用于功能理解。这引出了本文的核心问题：“我们能否利用基础模型来构建功能性3D场景图？”


<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/06/2026_06_11__23_42_36_0bdd4f.jpg"/>



Fig. 1. Functional 3D Scene Graphs. Given an input sequence of posed RGB-D frames of an indoor environment, our method predicts a functional 3D scene graph by detecting objects, identifying interactive elements, and inferring functional relationships. This enables the representation of interactions, functions, and scene dynamics, going beyond existing 3D scene graph methods that are constrained to spatial relationships between static objects.
图1. 功能性3D场景图。给定一段带位姿的室内RGB-D帧输入序列，我们的方法通过检测物体、识别交互元素并推断功能关系来预测功能性3D场景图。这使得能够表示交互、功能和场景动态，超越了仅受限于静态物体间空间关系的现有3D场景图方法。


We evaluate our approach on two challenging datasets: an extended version of SceneFun3D [17] with newly added functional relationship annotations, and FunGraph3D, a freshly collected real-world dataset featuring high-precision 3D laser scans, accurately registered To address these limitations, we introduce functional 3D scene graphs, which model objects, interactive elements, and their functional relationships within a unified structure (formally defined in Section 3). This representation extends traditional 3D scene graphs by incorporating interactive sub-parts alongside objects and representing functional relationships beyond simple spatial ones. We argue that functional 3D scene graphs should possess the following characteristics. First, the representation should operate in an open-vocabulary manner to enhance generalization and applicability. Second, it should be flexible, allowing various attributes to be attached to nodes (e.g., sensor data, natural language captions, semantic features) and edges (e.g., relationship descriptions), thus ensuring adaptability for downstream applications.
我们在两个具有挑战性的数据集上评估了我们的方法：扩展版SceneFun3D[17]，新增了功能关系标注；以及FunGraph3D，一个新收集的真实世界数据集，具有高精度3D激光扫描、准确配准。为解决这些局限，我们引入了功能性3D场景图，在统一结构中建模物体、交互元素及其功能关系（其形式定义见第3节）。该表示通过纳入物体旁的交互子部件，并表示超越简单空间关系的功能关系，扩展了传统3D场景图。我们认为，功能性3D场景图应具备以下特性。首先，该表示应以开放词汇方式运行，以增强泛化性和适用性。其次，它应具有灵活性，允许在节点上附加多种属性（如传感器数据、自然语言描述、语义特征）以及在边上附加多种属性（如关系描述），从而确保对下游应用的适应性。


In summary, our key contributions are:
总之，我们的主要贡献如下：


- We introduce functional 3D scene graphs that extend traditional 3D scene graphs by capturing functional relationships between objects and interactive elements.
- 我们引入了功能性3D场景图，通过捕获物体与交互元素之间的功能关系，扩展了传统3D场景图。


- We propose a novel approach that leverages the knowledge embedded in foundation models, specifically VLMs and LLMs, to construct functional 3D scene graphs without task-specific training.
- 我们提出了一种新方法，利用基础模型中嵌入的知识，特别是VLM和LLM，在无需任务特定训练的情况下构建功能性3D场景图。


- We present a new real-world dataset, FunGraph3D, with ground-truth functional annotations, and demonstrate that our method outperforms adapted baselines, including Open3DSG and ConceptGraph.
- 我们提出了一个新的真实世界数据集 FunGraph3D，包含真实的功能注释，并证明我们的方法优于改进后的基线方法，包括 Open3DSG 和 ConceptGraph。


## 2. Related Work
## 2. 相关工作


3D indoor scene understanding. Many works concentrate on closed-set 3D semantic segmentation [5, 14, 31- 33, 42, 45, 57, 58, 76, 80, 81] or instance segmentation [23, 28, 29, 38, 70, 74, 77, 96] on the existing 3D indoor scene understanding benchmarks [3, 7, 10, 15, 37, 65, 72, 93]. With the development of foundation models, subsequent researches explore open-vocabulary 3D semantic segmentation [24, 36, 39, 56, 59, 73, 75, 94, 105, 107], and complex 3D visual language grounding tasks [8, 16, 30, 34, 55, 62, 90, 103]. However, current studies mainly focus on object-level perception in indoor scene and seldom consider part-level interactive elements. Recently, SceneFun3D [17] proposes a benchmark for functionality and affordance understanding, with exhaustive annotations of indoor interactive elements. However, it does not provide the object annotations as well as the relationships between the elements and objects. This work extends SceneFun3D by exploiting such relationships with functional 3D scene graphs.
3D室内场景理解。许多工作聚焦于封闭类别的3D语义分割 [5, 14, 31- 33, 42, 45, 57, 58, 76, 80, 81] 或实例分割 [23, 28, 29, 38, 70, 74, 77, 96]，在现有3D室内场景理解基准 [3, 7, 10, 15, 37, 65, 72, 93] 上进行。随着基础模型的发展，后续研究探索开放词汇的3D语义分割 [24, 36, 39, 56, 59, 73, 75, 94, 105, 107]，以及复杂的3D视觉语言定位任务 [8, 16, 30, 34, 55, 62, 90, 103]。然而，当前研究主要关注室内场景中的对象级感知，较少考虑部分级的交互元素。近期，SceneFun3D [17] 提出功能性与可用性（affordance）理解的基准，对室内交互元素进行了详尽标注。但它并不提供对象标注，也不给出元素与对象之间的关系。本文通过利用此类关系来扩展 SceneFun3D，构建功能性的3D场景图。


Affordance understanding. Understanding affordance, i.e., properties of an environment to interact with, is a vital task in computer vision and robotics. Existing learning-based methods usually take inputs such as images [22, 98], videos [26, 54, 95] or 3D representations [18, 52, 53, 86], and then predict affordance maps. Some works learn affordance from human-scene interaction demonstrations [6, 12, 13, 25, 91, 92, 100, 101]. Nevertheless, existing works are often limited to object-level predictions and model affordances located on the corresponding objects. On the contrary, OpenFunGraph excavates all interactive elements at scene level, handling all kinds of functional relationships, especially those for remote operations.
可用性理解。理解可用性（即环境中可用于交互的属性）是计算机视觉与机器人领域的一项关键任务。现有基于学习的方法通常以图像 [22, 98]、视频 [26, 54, 95] 或3D表征 [18, 52, 53, 86] 等作为输入，然后预测可用性图。有些工作从人-场景交互演示中学习可用性 [6, 12, 13, 25, 91, 92, 100, 101]。尽管如此，现有方法往往局限于对象级预测，并且将模型的可用性仅定位在对应对象上。相反，OpenFunGraph 在场景层面挖掘所有交互元素，处理各种功能性关系，尤其是用于远程操作的关系。


3D scene graphs. 3D scene graph combines indoor entities into a unified structure and models inter-object relationships by building a graph of objects $\lbrack 4,{40},{63},{64},{75},{78},{79},{84}$ , 85, 99, 102]. Functional 3D scene graph differs from the traditional 3D scene graph by adding interactive elements as nodes and modeling the functional relationships between objects and elements. Similarly, IFR-Explore [44] tries to excavate inter-object functional relationships based on reinforcement learning in synthetic scenarios. However, it is hard to be applied in complex real-world scenes due to its closed-set setting, requirement of ground-truth instances, and lack of consideration on part-level elements. In this paper, we propose an open-vocabulary framework for functional scene graph inference in complex real-world scenes. While there have been related efforts on open-vocabulary 3D scene graph generation, they are not well-suited for functional scene graph inference, particularly for interactive element recognition and functional relationship prediction. For example, Open3DSG [41] relies on object-level CLIP features [60]. It struggles with part-level interactive element recognition and is limited to inferring spatial relationships due to its design based on spatial-proximity edge feature distillation. ConceptGraph [27] uses a direct inference pipeline but focuses solely on object nodes and a narrow set of spatial relationships (e.g., on, in). In contrast, our approach introduces adaptive detection and description stages for both objects and interactive elements, alongside a sequential reasoning strategy for accurately modeling a wide range of functional relationships.
3D场景图。3D场景图将室内实体组织为统一结构，并通过构建对象图 $\lbrack 4,{40},{63},{64},{75},{78},{79},{84}$ , 85, 99, 102] 来建模对象间关系。功能性3D场景图不同于传统3D场景图：它将交互元素作为节点加入，并建模对象与元素之间的功能性关系。类似地，IFR-Explore [44] 也试图在合成场景中基于强化学习挖掘对象间的功能关系。然而，由于其封闭类别设定、需要真实实例（ground-truth instances），且未考虑部分级元素，它很难应用于复杂的真实世界场景。本文提出了一个面向复杂真实场景的开放词汇功能场景图推理框架。尽管已经有一些开放词汇3D场景图生成的相关工作，但它们并不适合功能场景图推理，尤其是在交互元素识别与功能关系预测方面。例如，Open3DSG [41] 依赖对象级CLIP特征 [60]。它在部分级交互元素识别上表现困难，并且由于其基于空间邻近（spatial-proximity）边特征蒸馏的设计，仅能推断空间关系。ConceptGraph [27] 使用直接推理流程，但只关注对象节点，并且仅包含一小类空间关系（如 on、in）。相比之下，我们的方法为对象和交互元素分别引入自适应的检测与描述阶段，并采用顺序推理策略，以准确建模广泛的功能性关系。


<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/06/2026_06_11__23_42_36_a20f30.jpg"/>



Fig. 2. Illustration of the OpenFunGraph architecture. Given a sequence of posed RGB-D frames ${\left\{  \left( {\mathcal{I}}_{i},{\mathcal{D}}_{i}\right) \right\}  }_{i = 1}^{n}$ ,we use RAM++ [104] and GroundingDINO [49] to detect and segment objects $\mathcal{O}$ and interactive elemens $\mathcal{I}$ ,forming the node candidates of the functional 3D scene graph. Next, a mechanism using the large language model (LLM) GPT [1] and the visual language model (VLM) LLAVA [48] generates natural language descriptions $\mathcal{L}$ for each node. Finally,we infer functional relationships $\mathcal{R}$ between objects $\mathcal{O}$ and interactive elements $\mathcal{I}$ ,represented as the edges in the functional 3D scene graph $\mathcal{G}$ .
图2. OpenFunGraph架构示意。给定一组姿态已给定的RGB-D帧序列 ${\left\{  \left( {\mathcal{I}}_{i},{\mathcal{D}}_{i}\right) \right\}  }_{i = 1}^{n}$ ，我们使用RAM++ [104] 和 GroundingDINO [49] 来检测并分割对象 $\mathcal{O}$ 和交互元素 $\mathcal{I}$ ，从而形成功能性3D场景图的节点候选。接下来，利用大型语言模型（LLM）GPT [1] 和视觉语言模型（VLM）LLAVA [48] 为每个节点生成自然语言描述 $\mathcal{L}$ 。最后，我们推理对象 $\mathcal{O}$ 与交互元素 $\mathcal{I}$ 之间的功能关系 $\mathcal{R}$ ，并将其表示为功能性3D场景图 $\mathcal{G}$ 中的边。


## 3. Problem Formulation
## 3. 问题定义


Functional 3D Scene Graphs. We extend traditional 3D scene graphs [27, 41, 78] to facilitate their use in real-world scene interaction scenarios. Specifically, we introduce Functional 3D Scene Graphs, a representation designed to enable functional reasoning by jointly modeling objects, interactive elements and their functional relationships. We define a functional 3D scene graph as a directed graph $\mathcal{G} = \left( {\mathcal{O},\mathcal{I},\mathcal{R}}\right)$ where $\mathcal{O}$ are the objects in the scene, $\mathcal{I}$ are the interactive elements and $\mathcal{R}$ are the functional relationships which point from the interactive element nodes $\mathcal{I}$ to object nodes $\mathcal{O}$ . Following the definition in [17],we define interactive elements as components that agents interact with (e.g., handles, knobs and buttons) to trigger specific functions within the environment such as opening a cabinet or turning off a light. Additionally, functional relationships fall into two categories: local, where the interactive element is part of the object (e.g., door-handle), or remote, where the interactive element operates the object from a distance (e.g., TV-remote control).
功能性3D场景图。我们将传统的3D场景图[27,41,78]加以扩展，以便其可用于真实场景交互。具体而言，我们提出功能性3D场景图，这是一种通过联合建模物体、交互元素及其功能关系来支持功能推理的表示。我们将功能性3D场景图定义为一个有向图$\mathcal{G} = \left( {\mathcal{O},\mathcal{I},\mathcal{R}}\right)$，其中$\mathcal{O}$是场景中的物体，$\mathcal{I}$是交互元素，$\mathcal{R}$是指向从交互元素节点$\mathcal{I}$到物体节点$\mathcal{O}$的功能关系。根据[17]中的定义，我们将交互元素定义为智能体与之交互的组件（例如把手、旋钮和按钮），用于触发环境中的特定功能，如打开柜门或关闭灯。此外，功能关系分为两类：局部关系，即交互元素是物体的一部分（例如门把手）；远程关系，即交互元素从远处操控物体（例如电视遥控器）。


Task definition. We formulate the following novel 3D scene understanding task: Given an input sequence of posed RGB-D frames ${\left\{  \left( {\mathcal{I}}_{i},{\mathcal{D}}_{i}\right) \right\}  }_{i = 1}^{n}$ of an unseen indoor environment, the task is to construct the functional 3D scene graph $\mathcal{G}$ by inferring the functional relationships $\mathcal{R}$ among the objects $\mathcal{O}$ and interactive elements $\mathcal{I}$ in the scene.
任务定义。我们提出如下新的3D场景理解任务：给定一个未见过的室内环境中带位姿的RGB-D帧序列${\left\{  \left( {\mathcal{I}}_{i},{\mathcal{D}}_{i}\right) \right\}  }_{i = 1}^{n}$，任务是通过推断场景中物体$\mathcal{O}$与交互元素$\mathcal{I}$之间的功能关系$\mathcal{R}$，构建功能性3D场景图$\mathcal{G}$。


## 4. Method
## 4. 方法


The goal of our method, OpenFunGraph, is to predict the functional 3D scene graph of a 3D environment, by accurately detecting objects and interactive elements, and inferring the functional relationships among them in an open-vocabulary manner (Figure 2). To overcome the challenge of limited training data, we harness the knowledge of foundation models [9] to detect objects and interactive elements within the scene, describe them in natural language, and reason about their functional relationships. In the detection stage (Section 4.1), we follow a progressive strategy where we prompt the foundation model to systematically first identify objects and then transition to finer-grained interactive elements given the input image sequence. The 2D detection results are then fused across multiple viewpoints in 3D space, constructing an initial set of node candidates. Next, we utilize a VLM and an LLM to collaboratively generate multi-view aware natural language descriptions of the candidate nodes (Section 4.2). To construct the graph, we proceed with inferring the functional relationships, i.e., edges, among the object and interactive element nodes (Section 4.3). Specifically, we follow a sequential reasoning strategy, starting with local functional relationships (e.g., door - handle) and extending to remote functional relationships (e.g., TV - remote control), by leveraging the common sense knowledge of VLMs and LLMs. This allows us to progressively build the scene's functional graph by incrementally establishing connections between nodes.
我们的方法 OpenFunGraph 的目标，是通过准确检测物体和交互元素，并以开放词汇的方式推断它们之间的功能关系，来预测三维环境中的功能性三维场景图（图 2）。为克服训练数据有限的挑战，我们利用基础模型 [9] 的知识来检测场景中的物体和交互元素，用自然语言描述它们，并推理它们的功能关系。在检测阶段（第 4.1 节），我们采用渐进式策略，先提示基础模型系统地识别物体，再在输入图像序列的基础上过渡到更细粒度的交互元素。随后，将 2D 检测结果在三维空间的多个视角中进行融合，构建初始节点候选集。接着，我们利用 VLM 和 LLM 协同生成具备多视角感知的候选节点自然语言描述（第 4.2 节）。为构建图，我们进一步推断物体节点与交互元素节点之间的功能关系，即边（第 4.3 节）。具体而言，我们遵循顺序推理策略，从局部功能关系（如 door - handle）出发，并借助 VLM 和 LLM 的常识知识扩展到远程功能关系（如 TV - remote control）。这使我们能够通过逐步建立节点之间的连接，渐进式构建场景的功能图。


### 4.1. Node Candidate Detection
### 4.1 节点候选检测


In the first stage, we detect objects and interactive elements in the scene to construct a set of node candidates. We start by detecting 2D candidates on the input frames with a progressive foundation-model-based strategy that transitions from objects to finer-grained part-level interactive elements. Then, we associate and fuse the 2D detection results from multiple frames using geometric consistency, yielding the initial set of 3D node candidates.
在第一阶段，我们在场景中检测物体和交互元素，以构建一组节点候选。我们首先采用基于渐进式基础模型的方法，在输入帧上检测二维候选：从物体逐步过渡到更细粒度的部件级交互元素。随后，我们利用几何一致性，将多帧的二维检测结果进行关联与融合，得到初始的三维节点候选集合。


Object candidates. To identify object candidates ${\mathcal{C}}_{o}^{{\mathcal{I}}_{i}}$ ,we utilize RAM++ [35, 104] to recognize objects in each input image ${\mathcal{I}}_{i}$ ,producing object tags ${\mathcal{T}}_{obj}^{\mathcal{I}i}$ ,such as ’cabinet’ or 'door'. These object tags then serve as prompts for Ground-ingDINO [49],which detects 2D bounding boxes ${\mathcal{B}}^{{\mathcal{I}}_{i}}$ ,segmentation masks ${\mathcal{M}}^{{\mathcal{I}}_{i}}$ ,and confidence scores ${\mathcal{S}}^{{\mathcal{I}}_{i}}$ .
物体候选。为识别物体候选 ${\mathcal{C}}_{o}^{{\mathcal{I}}_{i}}$ ，我们使用 RAM++ [35, 104] 对每张输入图像 ${\mathcal{I}}_{i}$ 进行识别，生成物体标签 ${\mathcal{T}}_{obj}^{\mathcal{I}i}$ ，例如 '柜子' 或 '门'。这些物体标签随后作为 Ground-ingDINO [49] 的提示，用于检测二维边界框 ${\mathcal{B}}^{{\mathcal{I}}_{i}}$ 、分割掩码 ${\mathcal{M}}^{{\mathcal{I}}_{i}}$ 以及置信度分数 ${\mathcal{S}}^{{\mathcal{I}}_{i}}$ 。


Interactive element candidates. Despite the increasing success of foundation models in detecting object instances within scenes, the development of prompting strategies for identifying smaller elements, including interactive object parts (e.g., knobs, handles), remains largely unexplored. Here, we propose a simple yet effective strategy to generate suitable text prompts for GroundingDINO to improve the detection of small interactive parts. We ask the LLM GPT-4 to provide a list of potential interactive element tags corresponding to each object candidate tag ${\mathcal{T}}_{obj}^{{\mathcal{I}}_{i}}$ . We hold the valid object tags ${\mathcal{T}}_{\text{ val }}^{{\mathcal{I}}_{i}}$ by filtering the cases where the LLM thinks the object is not interactable (e.g., wall, bed). To create prompts for GroundingDINO,we concatenate ${\mathcal{T}}_{\text{ val }}^{{\mathcal{I}}_{i}}$ (e.g., door) as assistive tags with the functional element tags (e.g., handle), forming prompts such as "door. handle". Finally, we yield the interactive element candidates ${\mathcal{C}}_{ie}^{{\mathcal{I}}_{i}}$ in each input image ${\mathcal{I}}_{i}$ by maintaining the detections corresponding to the functional element tags. Empirically, we observe that this approach leads to more accurate detection of small interactive parts. We support this observation with an ablation study in Section 6.3.
交互元素候选。尽管基础模型在检测场景中的物体实例方面取得了越来越大的成功，但用于识别更小元素（包括交互物体部件，如旋钮、把手）的提示策略开发仍几乎未被探索。这里，我们提出一种简单且有效的策略：为 GroundingDINO 生成合适的文本提示，从而提升对小型交互部件的检测。我们请 LLM GPT-4 为每个物体候选标签 ${\mathcal{T}}_{obj}^{{\mathcal{I}}_{i}}$ 提供一份可能的交互元素标签列表。我们通过过滤掉 LLM 判断为不可交互的情况来保留有效物体标签 ${\mathcal{T}}_{\text{ val }}^{{\mathcal{I}}_{i}}$（例如墙、床）。为了为 GroundingDINO 创建提示，我们将 ${\mathcal{T}}_{\text{ val }}^{{\mathcal{I}}_{i}}$（例如门）作为辅助标签与功能元素标签（例如把手）拼接，形成类似“door. handle”的提示。最后，我们通过保留对应功能元素标签的检测结果，在每张输入图像 ${\mathcal{I}}_{i}$ 中得到交互元素候选 ${\mathcal{C}}_{ie}^{{\mathcal{I}}_{i}}$。通过实证观察，我们发现这种方法能更准确地检测小型交互部件。我们在第 6.3 节通过消融实验支持这一观察。


3D candidate fusion. After identifying the object and functional element candidates ${\mathcal{C}}_{obj}^{{\mathcal{I}}_{i}}$ and ${\mathcal{C}}_{ie}^{{\mathcal{I}}_{i}}$ in each image ${\mathcal{I}}_{i}$ ,we fuse their 2D segmentation masks using multi-view information to obtain the 3D node candidates of the graph. Following [27],we utilize the corresponding depth map ${\mathcal{D}}_{i}$ and camera projection matrix ${\Pi }_{i}$ to backproject the 2D mask to the 3D space, and merge them to receive the 3D object candidates ${\mathcal{C}}_{o}$ and interactive element candidates ${\mathcal{C}}_{ie}$ . For each node candidate, we store the backprojected 3D point cloud $\mathcal{P}$ and 3D bounding box $\mathcal{B}$ along with the associated 2D image assets, i.e., images, masks, 2D bounding boxes and confidence scores.
三维候选融合。在每张图像 ${\mathcal{I}}_{i}$ 中识别出物体与功能元素候选 ${\mathcal{C}}_{obj}^{{\mathcal{I}}_{i}}$ 和 ${\mathcal{C}}_{ie}^{{\mathcal{I}}_{i}}$ 后，我们利用多视角信息融合它们的二维分割掩码，从而获得图的三维节点候选。按照 [27]，我们使用对应的深度图 ${\mathcal{D}}_{i}$ 和相机投影矩阵 ${\Pi }_{i}$ 将二维掩码反投影到三维空间，并对其进行合并，得到三维物体候选 ${\mathcal{C}}_{o}$ 和交互元素候选 ${\mathcal{C}}_{ie}$ 。对于每个节点候选，我们将反投影得到的三维点云 $\mathcal{P}$ 和三维边界框 $\mathcal{B}$ 与对应的二维图像资源一起存储，即图像、掩码、二维边界框以及置信度分数。


### 4.2. Node Candidate Description
### 4.2. 节点候选描述


We next outline the process of generating natural language descriptions $\mathcal{L}$ for each node by leveraging a combination of VLMs and LLMs. Precise language descriptions are critical for establishing functional relationships in the final phase.
接下来，我们概述通过结合VLM和LLM为每个节点生成自然语言描述$\mathcal{L}$的过程。精确的语言描述对于最终阶段建立功能关系至关重要。


Object candidates. To generate natural language descriptions for each object candidate node, we first select the top ${N}_{v}$ views of each object,ranked by ${\mathcal{S}}^{{\mathcal{I}}_{i}} \times  \frac{{n}_{{\mathcal{P}}^{{\mathcal{I}}_{i}}}}{{n}_{\mathcal{P}}}$ ,where ${\mathcal{S}}^{{\mathcal{I}}_{i}}$ is the 2D confidence score indicating the semantic confidence,while ${n}_{{\mathcal{P}}^{{\mathcal{I}}_{i}}}$ refers to the number of 3D points the view ${\mathcal{I}}_{i}$ contributes to the fused 3D pointcloud $\mathcal{P}$ ,presenting the geometric contribution of the view. Each object is then cropped based on its bounding box $\mathcal{B}$ ,and a caption describing the object crop is obtained using LLAVA v1.6 [46- 48]. Finally, to derive a unified language description for each object candidate, we employ GPT-4 [1] to summarize the multi-view LLAVA captions.
对象候选。为给每个对象候选节点生成自然语言描述，我们首先为每个对象选取前${N}_{v}$个视角，并按${\mathcal{S}}^{{\mathcal{I}}_{i}} \times  \frac{{n}_{{\mathcal{P}}^{{\mathcal{I}}_{i}}}}{{n}_{\mathcal{P}}}$排序，其中${\mathcal{S}}^{{\mathcal{I}}_{i}}$是表示语义置信度的2D置信分数，而${n}_{{\mathcal{P}}^{{\mathcal{I}}_{i}}}$指该视角${\mathcal{I}}_{i}$对融合后的3D点云$\mathcal{P}$贡献的3D点数，体现该视角的几何贡献。随后，基于其边界框$\mathcal{B}$对每个对象进行裁剪，并使用LLAVA v1.6 [46- 48]获取描述该对象裁剪图的字幕。最后，为了为每个对象候选提炼出统一的语言描述，我们采用GPT-4 [1]对多视角LLAVA字幕进行总结。


Interactive element candidates. Captioning small interactive elements poses additional challenges: the bounding box crops are considerably smaller, often containing only a few pixels, which hinders LLAVA's ability to generate accurate captions. To address this, we enlarge the bounding boxes by multiple scales to incorporate richer contextual visual information. Similar multi-scale approaches have been shown to be effective in [39, 73]. To direct the VLM's attention to the interactive element within the expanded crop, we highlight the element with a red outline before passing it to LLAVA, as demonstrated in [71]. Finally, the multi-scale, multiview captions are summarized into a single natural language description using GPT-4.
交互元素候选。为小型交互元素生成字幕带来了额外挑战：边界框裁剪尺寸要小得多，通常只包含少量像素，这阻碍了LLAVA生成准确字幕的能力。为解决这一问题，我们将边界框按多个尺度放大，以纳入更丰富的上下文视觉信息。类似的多尺度方法已被证明在[39, 73]中有效。为将VLM的注意力引导至扩展裁剪中的交互元素，我们在将其输入LLAVA之前先用红色轮廓高亮该元素，如[71]所示。最后，利用GPT-4将多尺度、多视角字幕总结为单一的自然语言描述。


### 4.3. Functional Relationships
### 4.3. 功能关系


To model functional relationships between objects and interactive elements, we employ a sequential reasoning approach. Drawing on the concept of Chain-of-Thought reasoning [82], we decompose the task into a series of simpler steps rather than prompting the LLM to infer all possible element-object connections simultaneously. Initially, we concentrate on identifying direct, local relationships between objects and elements that are rigidly connected (e.g., door - handle). Once these relationships are established, we extend the search to remote relationships, where object-element pairs are functionally related but physically separated (e.g., TV - remote control).
为建模对象与交互元素之间的功能关系，我们采用按步骤推理的方法。借鉴链式思维推理 [82] 的概念，我们将任务拆分为一系列更简单的步骤，而不是让 LLM 同时推断所有可能的元素-对象连接。首先，我们聚焦于识别对象与元素之间直接的、局部的关系，这些关系是刚性连接的（例如：门 - 把手）。一旦这些关系确定，我们再扩展到远程关系：在这种情况下，对象-元素对在功能上有关联，但在物理上彼此分离（例如：电视 - 遥控器）。


Local relationship reasoning. First, we aim to construct the edges of the graph with local functional relationships, e.g., the keypanel of a microwave or the knob of a cabinet. A common characteristic of these cases is that objects and interactive elements are rigidly connected. To identify such cases efficiently, we first perform a spatial filtering process: For each object node ${\mathcal{C}}_{o}^{j}$ ,we assess whether an element node ${\mathcal{C}}_{ie}^{k}$ has a significant spatial overlap. Subsequently,we leverage the LLM's common sense knowledge to reason whether a local functional relationship between these two nodes is feasible. To do this, we prompt the LLM with the language descriptions ${\mathcal{L}}^{j},{\mathcal{L}}^{k}$ and 3D bounding boxes ${\mathcal{B}}^{j},{\mathcal{B}}^{k}$ of ${\mathcal{C}}_{o}^{j}$ and ${\mathcal{C}}_{ie}^{k}$ respectively. It is tasked with reasoning whether a local rigid connection between the interactive element (e.g., handle) and object (e.g., fridge) is feasible, and then generate a language description ${\mathcal{L}}^{k \rightarrow  j}$ of the functional relationship (e.g., "opens"). This step produces the subgraph of local connections ${\widehat{\mathcal{G}}}^{L} = \left( {{\mathcal{O}}^{L},{\mathcal{I}}^{L},{\mathcal{R}}^{L}}\right)$ .
局部关系推理。首先，我们要构建具有局部功能关系的图边，例如微波炉的按键面板或橱柜的旋钮。此类情形的一个共同特点是：对象与交互元素被刚性连接。为高效识别这些情况，我们首先进行空间筛选：对每个对象节点 ${\mathcal{C}}_{o}^{j}$ ，评估是否存在元素节点 ${\mathcal{C}}_{ie}^{k}$ 具有显著的空间重叠。随后，我们利用 LLM 的常识知识推理这两个节点之间是否可能存在局部功能关系。为此，我们向 LLM 提供 ${\mathcal{C}}_{o}^{j}$ 与 ${\mathcal{C}}_{ie}^{k}$ 的语言描述 ${\mathcal{L}}^{j},{\mathcal{L}}^{k}$ 以及 3D 边界框 ${\mathcal{B}}^{j},{\mathcal{B}}^{k}$。让其推理交互元素（例如把手）与对象（例如冰箱）之间的局部刚性连接是否可行，并生成该功能关系的语言描述 ${\mathcal{L}}^{k \rightarrow  j}$（例如“打开”）。该步骤会产出局部连接子图 ${\widehat{\mathcal{G}}}^{L} = \left( {{\mathcal{O}}^{L},{\mathcal{I}}^{L},{\mathcal{R}}^{L}}\right)$ 。


Confidence-aware remote relationship reasoning. In this step, we construct graph edges representing remote functional relationships, such as those between a ceiling light and its switch. Determining these remote relationships is challenging, as visual cues alone often do not fully clarify which interactive element controls which specific object. To address this, we introduce a confidence-aware reasoning strategy that assigns a confidence score to each inferred remote relationship. This approach enhances decision-making in real-world scenarios by enabling the agent to prioritize interactions with higher confidence scores.
考虑置信度的远程关系推理。在该步骤中，我们构建表示远程功能关系的图边，例如天花灯及其开关之间的关系。确定这些远程关系很有挑战，因为仅凭视觉线索往往无法充分说明哪个交互元素控制的是哪个具体对象。为解决这一问题，我们提出一种考虑置信度的推理策略：为每个推断出的远程关系分配置信度分数。该方法通过让智能体优先选择置信度更高的交互来增强其在真实场景中的决策能力。


First, we form an initial set of potential candidates for remote connections, by considering the interactive element nodes that remained unassigned from the previous stage. To construct potential remote connections among the interactive elements and objects in the scene, we utilize the common sense knowledge of the LLM. Specifically, we provide the LLM with natural language descriptions $\mathcal{L}$ of the interactive element and object nodes, so that it can output a list of likely target objects that each interactive element could be functionally linked to. Next, for each element-object pair, we employ the VLM to assess the feasibility of a functional connection. The visual input for this step is prepared by the top-1 views of the interactive element and object. The VLM can exploit useful information in the images of the element and object to generate descriptions for the feasibility assessment. For example, it describes whether the appliance is physically plugged into the electric outlet, or whether the switch is mount on the wall under the ceiling light. The descriptions from all pairs are then provided to the LLM to form a global context, assisting it to assign a relative confidence score to each proposed connection and describe the nature of each relationship. This step outputs the subgraph of remote relations: ${\widehat{\mathcal{G}}}^{R} = \left( {{\mathcal{O}}^{R},{\mathcal{I}}^{R},{\mathcal{R}}^{R}}\right)$ .
首先，我们通过考虑前一阶段仍未被分配的交互元素节点，形成远程连接的初始候选集合。为构建场景中交互元素与对象之间的潜在远程连接，我们利用 LLM 的常识知识。具体而言，我们提供给 LLM 交互元素节点与对象节点的自然语言描述 $\mathcal{L}$ ，使其输出一份可能目标对象列表，即每个交互元素可能在功能上与哪些对象相连。接下来，对于每个元素-对象对，我们使用 VLM 来评估功能连接的可行性。该步骤的视觉输入由交互元素与对象的 top-1 视图组成。VLM 能从元素与对象的图像中提取有用信息，从而生成可行性评估所需的描述。例如，它会描述该电器是否在物理上插接在电源插座中，或开关是否安装在天花灯下方的墙上。然后，将所有配对得到的描述提供给 LLM 以形成全局上下文，帮助其为每个提出的连接分配相对置信度分数，并描述每段关系的性质。该步骤输出远程关系子图：${\widehat{\mathcal{G}}}^{R} = \left( {{\mathcal{O}}^{R},{\mathcal{I}}^{R},{\mathcal{R}}^{R}}\right)$ 。


### 4.4. Final Graph Formation
### 4.4. 最终图形构建


To construct the final graph, we combine the nodes and relationships identified in both the local and remote functional reasoning stages. The resulting predicted graph is formulated as $\widehat{\mathcal{G}} = \left( {{\mathcal{O}}^{L} \cup  {\mathcal{O}}^{R},{\mathcal{I}}^{L} \cup  {\mathcal{I}}^{R},{\mathcal{R}}^{L} \cup  {\mathcal{R}}^{R}}\right)$ .
为构建最终图，我们将本地与远程功能推理阶段中识别到的节点和关系进行组合。得到的预测图被表述为 $\widehat{\mathcal{G}} = \left( {{\mathcal{O}}^{L} \cup  {\mathcal{O}}^{R},{\mathcal{I}}^{L} \cup  {\mathcal{I}}^{R},{\mathcal{R}}^{L} \cup  {\mathcal{R}}^{R}}\right)$ 。


## 5. Data Collection
## 5. 数据采集


Existing datasets of high-fidelity 3D indoor spaces focus primarily on understanding either 3D objects [7, 93] or 3D interactive elements [17]. However, they lack ground-truth annotations of the functional relationships. In many cases, these relationships cannot be inferred from static visual observations alone but instead require video captures of physical interactions with the scene to determine which actions trigger specific responses. For example, a static 3D reconstruction cannot indicate which switch controls a particular light in a room with multiple switches and lights. To systematically evaluate our method, we construct a novel dataset of 3D real-world indoor environments along with multi-sensor data (i.e., high-fidelity 3D reconstructions, consumer-device video captures, egocentric human-scene interaction videos) and functional 3D scene graph annotations. We outline the steps towards building this dataset, which we refer to as FunGraph3D (Figure 4).
现有的高保真3D室内数据集主要关注理解3D物体[7, 93]或3D交互元素[17]。然而，它们缺乏功能关系的真实标注。很多情况下，这些关系仅凭静态视觉观察无法推断，而需要通过对场景的物理交互进行视频采集，以确定哪些动作会触发特定响应。比如，在同一房间有多个开关和灯的情况下，静态3D重建无法判断哪个开关控制哪盏灯。为系统评估我们的方法，我们构建了一个新的真实世界3D室内环境数据集，并配套多传感器数据（即高保真3D重建、消费级设备视频采集、第一人称人-场景交互视频），以及功能3D场景图标注。我们将概述构建该数据集的步骤，并将其命名为FunGraph3D（图4）。


<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/06/2026_06_11__23_42_36_f90f86.jpg"/>



Fig. 3. Modalities of our FunGraph3D dataset. Top: 3D scans from a Faro laser scanner, annotated with 3D object and interactive element masks. Middle: Ground truth functional 3D scene graphs. Bottom: Egocentric video capturing human-scene interactions.
图3. 我们的FunGraph3D数据集的模态。上：来自Faro激光扫描仪的3D扫描，并标注3D物体与交互元素掩码。中：真实的功能3D场景图。下：第一人称视频，用于捕捉人-场景交互。


<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/06/2026_06_11__23_42_36_0081a7.jpg"/>



Fig. 4. Example scenes from our FunGraph3D dataset. The dataset includes typical indoor environments such as living rooms, bedrooms, bathrooms, and kitchens.
图4. 我们的FunGraph3D数据集中的示例场景。该数据集包含典型的室内环境，如客厅、卧室、浴室和厨房。


Laser scans. As illustrated in [17], we highlight that laser scans can capture a higher level of 3D geometry details, such as small interactive elements (i.e., knobs, buttons), which is necessary for fine-grained scene understanding applications. To this end, we use a Leica RTC360 laser scanner to capture a high-resolution (5mm) 3D scan of the scene. To ensure high scene coverage during the capture, we place the scanner in multiple positions in the scene. We subsequently use the supporting software by Leica to fuse the multiple scans into a single one for the scene.
激光扫描。如[17]所示，我们强调激光扫描能够捕获更高层次的3D几何细节，例如小型交互元素（即旋钮、按键），这对于细粒度场景理解任务是必要的。为此，我们使用Leica RTC360激光扫描仪对场景进行高分辨率（5mm）的3D扫描。在采集过程中，为确保覆盖场景的范围，我们将扫描仪放置在场景中的多个位置。随后，我们使用Leica配套的软件将多次扫描融合为场景中的单一结果。


iPad video sequences. To enable scene understanding through multiple sensor data, we accompany the high-fidelity 3D reconstruction with RGB-D image information from a commodity device. Specifically, we capture multiple videos of the static scene with the camera of an iPad 15 Pro.
iPad视频序列。为利用多传感器数据实现场景理解，我们在高保真的3D重建之外，配套使用消费级设备提供的RGB-D图像信息。具体而言，我们使用iPad 15 Pro的相机对静态场景采集多段视频。


Registration and alignment. To register the iPad video frames to the laser scan coordinate system, we build upon the COLMAP-based pipeline in [93]. Specifically, we run the COLMAP SfM pipeline [68, 69] by augmenting the collection of real iPad frames with rendered pseudo images of the laser scan. However, we notice that this pipeline leads to a large number of unregistered frames. To address this limitation, we incorporate the deep learning-based methods Superpoint [19] and Superglue [67] for feature extraction and matching, leading to a more accurate registration result. Afterwards, we utilize the optimized pose for each camera frame to render high-resolution depth maps for accurate back-projection from the iPad frames to the 3D space.
配准与对齐。为将iPad视频帧配准到激光扫描坐标系，我们基于[93]中的基于COLMAP的流程。具体来说，我们通过将真实的iPad帧与激光扫描的渲染伪图像相结合来扩展数据采集，然后运行COLMAP SfM流程[68, 69]。然而，我们发现该流程会产生大量未成功配准的帧。为解决这一限制，我们引入基于深度学习的方法Superpoint[19]和Superglue[67]用于特征提取与匹配，从而得到更准确的配准结果。之后，我们使用每个相机帧的优化位姿，将iPad帧反投到3D空间中，以渲染高分辨率深度图并实现精确回投。


Egocentric videos. We include egocentric videos of property owners interacting with the environment using an Apple Vision Pro headset in our dataset. These videos facilitate accurate relationship labeling as they help clarify ambiguous connections among objects and interactive elements (e.g., which light switch controls the ceiling light).
第一人称视频。我们在数据集中包含物业业主使用Apple Vision Pro头显与环境交互的第一人称视频。这些视频有助于进行准确的关系标注，因为它们能澄清物体与交互元素之间的模糊连接（例如，哪个灯光开关控制顶灯）。


Annotation. For the annotation process, we extend the SceneFun3D annotation tool [17] to construct the ground-truth functional 3D scene graphs. Annotators can navigate the 3D scene and annotate the instances of objects and interactive elements along with a free-form label. Annotators are also asked to connect the interactive element to the corresponding object that it controls and provide a description of their relationship. An example of the collected annotations is displayed in Figure 3.
标注。进行标注时，我们将SceneFun3D标注工具[17]扩展为构建真实的功能3D场景图。标注者可以在3D场景中导航，并对物体实例与交互元素进行标注，同时使用自由形式标签。我们还要求标注者将交互元素连接到其所控制的对应物体，并给出对其关系的描述。所收集标注的示例如图3所示。


Statistics. FunGraph3D contains 14 in-the-wild scenes of various types (6 kitchens, 2 living rooms, 3 bedrooms and 3 bathrooms). In total, the dataset includes 201 interactive elements, 228 functional relationships and 146 objects of interest, along with open-vocabulary labels and relationships.
统计信息。FunGraph3D包含14个“野外”场景，涵盖多种类型（6个厨房、2个客厅、3个卧室和3个浴室）。总体而言，数据集包含201个交互元素、228条功能关系以及146个关注对象，同时提供开放词汇标签与关系。


## 6. Experiments
## 6. 实验


### 6.1. Experimental Setup
### 6.1. 实验设置


Datasets. To evaluate our method, we utilize the developed FunGraph3D dataset, described in Section 5. Additionally, we use the SceneFun3D dataset [17], which provides high-resolution $5\mathrm{\;{mm}}$ laser scans of real-world environments along with iPad video sequences. Specifically, we randomly select 20 scenes (8 from the validation and 12 from the test split) and apply our annotation pipeline to annotate the functional 3D scene graph in each scene. Since we do not have physical access to the 3D environments, we restrict our evaluation to functional relationships that are visually unambiguous. In total, 212 interactive elements, 195 functional relationships, and 105 corresponding objects are annotated for these scenes.
数据集。为评估我们的方法，我们使用了第5节中介绍的 FunGraph3D 数据集。此外，我们还使用 SceneFun3D 数据集 [17]，它提供了真实环境的高分辨率 $5\mathrm{\;{mm}}$ 激光扫描以及 iPad 视频序列。具体而言，我们随机选取20个场景（8个来自验证集，12个来自测试集），并对每个场景应用我们的标注流程来标注功能性3D场景图。由于我们无法实际接触这些3D环境，我们将评估限制在视觉上无歧义的功能关系上。总计为这些场景标注了212个交互元素、195种功能关系以及105个对应对象。


<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/06/2026_06_11__23_42_36_338d8c.jpg"/>



Fig. 5. Qualitative results. Top: input images. Bottom: predicted functional 3D scene graph. Best seen zoomed in on a color screen.
图5. 定性结果。上：输入图像。下：预测的功能性3D场景图。放大后在彩色屏幕上观看效果最佳。


Metrics. To evaluate open-vocabulary functional 3D scene graphs effectively, a new quantitative metric is essential. Existing approaches, such as ConceptGraph [27], rely on subjective human assessments, while Open3DSG [41] approaches evaluation as a label retrieval task, assuming all ground-truth nodes are known, an assumption that diverges from our real-world setting. To address this, we extend the Open3DSG Recall@K metric [41] with a node detection component, using spatial overlap between predicted and ground-truth nodes, inspired by evaluation techniques on 2D scene graph generation [50, 87-89, 106]. More specifically, our evaluation metric comprises two Recall@K scores: one for nodes,i.e., $\mathcal{O}$ and $\mathcal{I}$ ,and one for triplets, i.e., $\left( {\mathcal{O},\mathcal{I},\mathcal{R}}\right)$ . For node evaluation,we preprocess all ground-truth labels to enable top-K retrieval, following Open3DSG [41]. A retrieval is considered successful if a ground-truth node has a non-zero 3D IoU with a predicted node and the ground-truth label ranks within the top-K retrievals based on cosine similarity of CLIP embeddings [60] with the predicted label. We calculate overall node recall as ${R}_{no} = \frac{{n}_{no}^{re}}{{n}_{no}}$ ,where ${n}_{no}^{re}$ is the number of successfully retrieved ground-truth nodes,and ${n}_{no}$ is the total count of ground-truth nodes. Additionally, we assess recall for object and interactive element nodes separately, denoted as ${R}_{o} = \frac{{n}_{o}^{re}}{{n}_{o}}$ and ${R}_{ie} = \frac{{n}_{ie}^{re}}{{n}_{ie}}$ ,where ${n}_{o}^{re}$ and ${n}_{ie}^{re}$ are the counts of correctly retrieved objects and interactive elements and ${n}_{o}$ and ${n}_{ie}$ are their respective totals. For triplet $\left( {\mathcal{O},\mathcal{I},\mathcal{R}}\right)$ evaluation,we apply stricter criteria: a ground-truth triplet is successfully retrieved in the top-K only when all its components $\mathcal{O},\mathcal{I}$ and $\mathcal{R}$ are individually retrieved within the top-K. The retrieval process for $\mathcal{O}$ and $\mathcal{I}$ follows the same approach as above. To handle $\mathcal{R}$ ,we preprocess all relationship annotations by generating BERT em-beddings [20], an approach effective for open-vocabulary predicates [41]. Successful retrieval is based on cosine similarity between ground-truth and predicted BERT embed-dings. Triplet recall is defined as ${R}_{tr} = \frac{{n}_{re}}{{n}_{tr}}$ ,where ${n}_{re}$ is the count of retrieved triplets,and ${n}_{tr}$ is the total count of ground-truth. We decompose triplet evaluation into node association $\left( {{R}_{na} = \frac{{n}_{na}}{{n}_{tr}}}\right.$ ,with ${n}_{na}$ being the number of triplets retrieved only considering $\mathcal{O},\mathcal{I}$ ),indicating node recognition,and edge prediction $\left( {{R}_{ep} = \frac{{n}_{re}}{{n}_{na}}}\right)$ ,showing relationship inference given correct node associations.
指标。为有效评估开放词汇功能性3D场景图，必须采用新的定量指标。现有方法，如 ConceptGraph [27]，依赖主观人工评估，而 Open3DSG [41] 将评估视为标签检索任务，假设所有真实节点已知，这一假设与我们的真实世界设置不同。为此，我们在 Open3DSG 的 Recall@K 指标 [41] 基础上扩展了节点检测组件，使用预测节点与真实节点之间的空间重叠，并借鉴了2D场景图生成中的评估技术 [50, 87-89, 106]。更具体地说，我们的评估指标包含两个 Recall@K 分数：一个用于节点，即 $\mathcal{O}$ 和 $\mathcal{I}$，另一个用于三元组，即 $\left( {\mathcal{O},\mathcal{I},\mathcal{R}}\right)$。对于节点评估，我们按照 Open3DSG [41] 预处理所有真实标签，以支持 top-K 检索。若某个真实节点与一个预测节点的3D IoU 非零，且其真实标签基于与预测标签的 CLIP 嵌入 [60] 余弦相似度在 top-K 检索中排名靠前，则该检索被视为成功。我们将整体节点召回率计算为 ${R}_{no} = \frac{{n}_{no}^{re}}{{n}_{no}}$，其中 ${n}_{no}^{re}$ 为成功检索的真实节点数，${n}_{no}$ 为真实节点总数。此外，我们还分别评估对象节点和交互元素节点的召回率，记为 ${R}_{o} = \frac{{n}_{o}^{re}}{{n}_{o}}$ 和 ${R}_{ie} = \frac{{n}_{ie}^{re}}{{n}_{ie}}$，其中 ${n}_{o}^{re}$ 和 ${n}_{ie}^{re}$ 分别为正确检索的对象和交互元素数量，${n}_{o}$ 和 ${n}_{ie}$ 为其各自总数。对于三元组 $\left( {\mathcal{O},\mathcal{I},\mathcal{R}}\right)$ 评估，我们采用更严格的标准：只有当其所有组成部分 $\mathcal{O},\mathcal{I}$ 和 $\mathcal{R}$ 都分别在 top-K 中成功检索时，真实三元组才算在 top-K 中成功检索。$\mathcal{O}$ 和 $\mathcal{I}$ 的检索过程与上述相同。为处理 $\mathcal{R}$，我们通过生成 BERT 嵌入 [20] 来预处理所有关系标注，这种方法对开放词汇谓词 [41] 有效。成功检索基于真实与预测 BERT 嵌入之间的余弦相似度。三元组召回率定义为 ${R}_{tr} = \frac{{n}_{re}}{{n}_{tr}}$，其中 ${n}_{re}$ 是检索到的三元组数量，${n}_{tr}$ 是真实三元组总数。我们将三元组评估分解为节点关联 $\left( {{R}_{na} = \frac{{n}_{na}}{{n}_{tr}}}\right.$，其中 ${n}_{na}$ 为仅考虑 $\mathcal{O},\mathcal{I}$ 时检索到的三元组数量，表示节点识别；以及边预测 $\left( {{R}_{ep} = \frac{{n}_{re}}{{n}_{na}}}\right)$，表示在节点关联正确的情况下进行关系推断。


<table><tr><td rowspan="3">Methods</td><td colspan="6">SceneFun3D [17]</td><td colspan="6">FunGraph3D (Ours)</td></tr><tr><td colspan="2">Objects</td><td>Inter.</td><td rowspan="2">Elements R@10</td><td colspan="2">Overall 1 Nodes</td><td colspan="2">Objects</td><td colspan="2">Inter. Elements</td><td colspan="2">Overall Nodes</td></tr><tr><td>R@3</td><td>R@10</td><td>R@3</td><td>R@3</td><td>R@10</td><td>R@3</td><td>R@10</td><td>R@3</td><td>R@10</td><td>R@3</td><td>R@10</td></tr><tr><td>Open3DSG* [41]</td><td>61.2</td><td>70.7</td><td>54.4</td><td>61.8</td><td>56.7</td><td>64.7</td><td>50.9</td><td>58.1</td><td>21.8</td><td>33.9</td><td>33.4</td><td>43.6</td></tr><tr><td>Open3DSG*† [41]</td><td>42.9</td><td>50.0</td><td>33.8</td><td>38.3</td><td>37.4</td><td>43.0</td><td>30.9</td><td>44.1</td><td>13.0</td><td>19.6</td><td>20.2</td><td>29.4</td></tr><tr><td>ConceptGraph* [27]</td><td>71.3</td><td>77.1</td><td>6.6</td><td>8.6</td><td>28.3</td><td>31.4</td><td>58.0</td><td>66.3</td><td>2.5</td><td>4.1</td><td>20.1</td><td>25.2</td></tr><tr><td>ConceptGraph* [27] + IED</td><td>71.3</td><td>77.1</td><td>53.1</td><td>59.5</td><td>60.1</td><td>66.0</td><td>58.0</td><td>66.3</td><td>20.5</td><td>33.4</td><td>38.9</td><td>45.0</td></tr><tr><td>OpenFunGraph (Ours)</td><td>81.8</td><td>87.8</td><td>71.0</td><td>79.5</td><td>73.0</td><td>82.8</td><td>70.7</td><td>79.1</td><td>44.4</td><td>57.6</td><td>55.5</td><td>65.8</td></tr></table>
<table><tbody><tr><td rowspan="3">方法</td><td colspan="6">SceneFun3D [17]</td><td colspan="6">FunGraph3D（我们的方法）</td></tr><tr><td colspan="2">对象</td><td>交互</td><td rowspan="2">元素 R@10</td><td colspan="2">整体 1 节点</td><td colspan="2">对象</td><td colspan="2">交互元素</td><td colspan="2">整体节点</td></tr><tr><td>R@3</td><td>R@10</td><td>R@3</td><td>R@3</td><td>R@10</td><td>R@3</td><td>R@10</td><td>R@3</td><td>R@10</td><td>R@3</td><td>R@10</td></tr><tr><td>Open3DSG* [41]</td><td>61.2</td><td>70.7</td><td>54.4</td><td>61.8</td><td>56.7</td><td>64.7</td><td>50.9</td><td>58.1</td><td>21.8</td><td>33.9</td><td>33.4</td><td>43.6</td></tr><tr><td>Open3DSG*† [41]</td><td>42.9</td><td>50.0</td><td>33.8</td><td>38.3</td><td>37.4</td><td>43.0</td><td>30.9</td><td>44.1</td><td>13.0</td><td>19.6</td><td>20.2</td><td>29.4</td></tr><tr><td>ConceptGraph* [27]</td><td>71.3</td><td>77.1</td><td>6.6</td><td>8.6</td><td>28.3</td><td>31.4</td><td>58.0</td><td>66.3</td><td>2.5</td><td>4.1</td><td>20.1</td><td>25.2</td></tr><tr><td>ConceptGraph* [27] + IED</td><td>71.3</td><td>77.1</td><td>53.1</td><td>59.5</td><td>60.1</td><td>66.0</td><td>58.0</td><td>66.3</td><td>20.5</td><td>33.4</td><td>38.9</td><td>45.0</td></tr><tr><td>OpenFunGraph（我们的方法）</td><td>81.8</td><td>87.8</td><td>71.0</td><td>79.5</td><td>73.0</td><td>82.8</td><td>70.7</td><td>79.1</td><td>44.4</td><td>57.6</td><td>55.5</td><td>65.8</td></tr></tbody></table>


Tab. 1. Node evaluation on the SceneFun3D [17] and FunGraph3D datasets. * means to adapt the LLM prompts used for functional relationships inference. IED refers to the interactive element candidate detection in Section 4.1. ${}^{ \dagger  }$ refers to the usage of the OpenFunGraph's fused 3D nodes rather than the ground-truth for fair comparison.
表 1. 在 SceneFun3D [17] 与 FunGraph3D 数据集上的节点评估。* 表示需要适配用于功能关系推断的 LLM 提示。IED 指第 4.1 节中的交互元素候选检测。${}^{ \dagger  }$ 指在进行公平比较时使用 OpenFunGraph 的融合三维节点，而非使用真实标注。


State-of-the-art comparisons. We compare our approach against ConceptGraph [27] and Open3DSG [41]-based baselines. Two ConceptGraph-based baselines are reimplemented: ConceptGraph* modifies the original LLM prompts to infer functional relationships, rather than focusing on spatial relationships such as in or on. Concept-Graph* + IED further incorporates the proposed interactive element candidate detection (IED) from Section 4.1, addressing ConceptGraph's initial limitation in detecting small parts. Both baselines use LLAVA v1.6 and GPT-4 for fair comparison with OpenFunGraph. We also reimplement two Open3DSG-based baselines. Open3DSG* modifies the LLM prompts to output functional relationships instead of spatial relationships. Since Open3DSG baselines rely on ground-truth node instance segmentation for graph neural network inference, we implement Open3DSG*†, which uses OpenFunGraph's fused 3D nodes for fair comparison. We report Recall@3 and Recall@10 for node metrics, and Recall@5 and Recall@10 for triplet metrics.
最新方法对比。我们将所提方法与 ConceptGraph [27] 及基于 Open3DSG [41] 的基线进行比较。我们重实现了两个基于 ConceptGraph 的基线：ConceptGraph* 修改原始 LLM 提示以推断功能关系，而不是关注如 in 或 on 这类空间关系。Concept-Graph* + IED 进一步结合第 4.1 节提出的交互元素候选检测（IED），以解决 ConceptGraph 初始在检测小部件方面的局限。两种基线均使用 LLAVA v1.6 和 GPT-4，以确保与 OpenFunGraph 的公平比较。我们也重实现了两个基于 Open3DSG 的基线。Open3DSG* 修改 LLM 提示，使其输出功能关系而非空间关系。由于 Open3DSG 基线在图神经网络推断中依赖真实标注的节点实例分割，我们实现 Open3DSG*†，使用 OpenFunGraph 的融合三维节点以保证公平比较。我们报告节点指标的 Recall@3 和 Recall@10，以及三元组指标的 Recall@5 和 Recall@10。


### 6.2. Results
### 6.2. 结果


Quantitative results are presented in Table 1 and 2. Overall, the FunGraph3D dataset poses a greater challenge than SceneFun3D [17] due to its more complex scenes, which contain a higher number of objects and interactive elements.
定量结果见表1和表2。总体而言，由于场景更复杂、包含更多物体和交互元素，FunGraph3D数据集比SceneFun3D [17] 更具挑战性。


Node evaluation. As shown in Table 1, OpenFunGraph surpasses ConceptGraph* [27] by 160% on SceneFun3D and by 176% in R@3 on FunGraph3D. ConceptGraph* primarily focuses on object perception, resulting in poor recall scores for interactive elements. With the added interactive element candidate detection (IED), ConceptGraph* + IED improves node recognition, but still falls short of OpenFun-Graph by ${22}\%$ in R@3 on SceneFun3D,and ${43}\%$ in R@3 on FunGraph3D, thanks to the specified node description stage proposed in OpenFunGraph. Our approach also outperforms Open3DSG-based baselines, achieving 95% and 29% higher scores than Open3DSG*† and Open3DSG* in R@3 on SceneFun3D, and 174% and 66% higher on Fun-Graph3D. The limited ability of Open3DSG-based methods to identify interactive elements arises from their focus on object-level features during training, whereas our approach employs a more practical open-vocabulary inference pipeline, free from these training constraints.
节点评估。如表1所示，OpenFunGraph在SceneFun3D上的表现比ConceptGraph* [27]高出160%，在FunGraph3D上的R@3高出176%。ConceptGraph*主要聚焦于物体感知，因此对交互元素的召回较差。加入交互元素候选检测（IED）后，ConceptGraph* + IED提升了节点识别能力，但在SceneFun3D上的R@3仍比OpenFun-Graph低${22}\%$，在FunGraph3D上的R@3低${43}\%$，这得益于OpenFunGraph提出的指定节点描述阶段。我们的方法也优于基于Open3DSG的基线，在SceneFun3D上的R@3分别比Open3DSG*†和Open3DSG*高95%和29%，在Fun-Graph3D上高174%和66%。基于Open3DSG的方法在识别交互元素方面能力有限，源于其训练时侧重于物体级特征，而我们的方法采用了更实用的开放词汇推理流程，不受这些训练约束。


Triplet evaluation. Table 2 shows triplet prediction results. On SceneFun3D and FunGraph3D, benefiting from accurate node recognition and the sequential reasoning strategy for functional inference, OpenFunGraph outperforms ConceptGraph* + IED by 76% and 189% in R@5, and Open3DSG*† by 179% and 308%. Notably, Open3DSG-based baselines struggle with functional relationships, as they rely on spatial edge features from adjacent instances. ConceptGraph-based methods, which prompt the LLM to predict all possible connections, also perform worse when compared to our sequential reasoning strategy due to the increased interpretive complexity imposed on the LLM. Figure 5 visualizes qualitative results for OpenFunGraph. In the left scene, our confidence-aware remote relationship reasoning successfully infers that the light switch is more likely to control the ceiling light rather than the two table light bulbs. In the right scene, the local functional relationship between the handle and the door is accurately identified. Additionally, the fan is most confidently inferred to be powered by the nearby electric outlet.
三元组评估。表2展示了三元组预测结果。在SceneFun3D和FunGraph3D上，得益于准确的节点识别以及用于功能推断的顺序推理策略，OpenFunGraph在R@5上分别比ConceptGraph* + IED高76%和189%，比Open3DSG*†高179%和308%。值得注意的是，基于Open3DSG的基线在功能关系上表现吃力，因为它们依赖相邻实例的空间边特征。基于ConceptGraph的方法通过提示LLM预测所有可能连接，但由于对LLM施加了更高的解释复杂度，与我们的顺序推理策略相比表现也更差。图5展示了OpenFunGraph的定性结果。在左侧场景中，我们具备置信度感知的远程关系推理成功推断出，灯开关更可能控制吸顶灯，而不是两个桌灯灯泡。在右侧场景中，把手与门之间的局部功能关系被准确识别。此外，风扇最有把握地被推断为由附近的电源插座供电。


### 6.3. Ablation studies
### 6.3消融实验


We ablate three key modules in our pipeline, i.e., the GroundingDINO prompts for interactive element candidate detection, sequential reasoning, and confidence-aware remote relationship reasoning, presented in Table 3. The prompting strategy for GroundingDINO, which combines assistive object and element tags, proves effective. Using only element tags reduces node R@3 by 19% and 10%, as well as triplet R@5 by 20% and 22% on the two datasets respectively, due to incomplete detections. Replacing sequential reasoning with a direct approach, where the LLM infers functional relationships across all nodes, significantly reduces triplet reasoning performance (42% and 32% in triplet R@5 on SceneFun3D and FunGraph3D respectively). Sequential reasoning decomposes complex relationships into distinct types, making LLM processing easier. Ablating confidence-aware remote relationship reasoning by randomly selecting connections, instead of using the highest-confident edge (e.g., choosing a random light for the switch instead of the most confident ceiling light), leads to a decrease in triplet R@5 by 7% and 11% on the two datasets respectively. This illustrates more reasonable edges are selected correctly in our mechanism by incorporating the common sense understanding of the foundation models.
我们对管线中的三个关键模块进行消融，即用于交互元素候选检测的 GroundingDINO 提示、顺序推理以及置信度感知的远程关系推理，并在表3中给出结果。采用将辅助物体与元素标签结合的 GroundingDINO 提示策略是有效的。仅使用元素标签会使两种数据集上的节点 R@3 分别降低19%和10%，以及三元组 R@5 分别降低20%和22%，原因在于检测不完整。将顺序推理替换为直接方法——由 LLM 在所有节点之间推断功能关系——会显著降低三元组推理性能（在 SceneFun3D 与 FunGraph3D 上的三元组 R@5 分别为42%和32%）。顺序推理将复杂关系拆分为不同类型，从而使 LLM 的处理更容易。通过随机选择连接来消融置信度感知的远程关系推理（例如为开关随机选择一盏灯，而不是选择置信度最高的顶灯），两种数据集上的三元组 R@5 分别下降7%和11%。这表明，在引入基础模型的常识理解后，我们的机制能够正确选择更合理的边。


<table><tr><td rowspan="3">Methods</td><td colspan="6">SceneFun3D [17]</td><td colspan="6">FunGraph3D (Ours)</td></tr><tr><td colspan="2">Node Assoc.</td><td colspan="2">Edge Pred.</td><td colspan="2">Overall Triplets</td><td colspan="2">Node Assoc.</td><td colspan="2">Edge e Pred.</td><td colspan="2">Overall Triplets</td></tr><tr><td>R@5</td><td>R@10</td><td>R@5</td><td>R@10</td><td>R@5</td><td>R@10</td><td>R@5</td><td>R@10</td><td>R@5</td><td>R@10</td><td>R@5</td><td>R@10</td></tr><tr><td>Open3DSG* [41]</td><td>47.2</td><td>58.0</td><td>69.2</td><td>78.8</td><td>32.7</td><td>45.7</td><td>22.8</td><td>36.7</td><td>47.9</td><td>55.9</td><td>10.5</td><td>20.0</td></tr><tr><td>Open3DSG*† [41]</td><td>33.6</td><td>38.8</td><td>64.4</td><td>72.3</td><td>21.6</td><td>28.1</td><td>15.7</td><td>24.2</td><td>46.6</td><td>55.7</td><td>7.3</td><td>13.5</td></tr><tr><td>ConceptGraph* [27]</td><td>5.6</td><td>6.8</td><td>80.2</td><td>95.0</td><td>4.7</td><td>6.4</td><td>1.9</td><td>2.8</td><td>51.5</td><td>84.6</td><td>1.1</td><td>2.5</td></tr><tr><td>ConceptGraph* [27] + IED</td><td>45.4</td><td>49.3</td><td>75.6</td><td>90.9</td><td>34.3</td><td>44.5</td><td>18.8</td><td>22.8</td><td>46.1</td><td>79.7</td><td>10.3</td><td>18.9</td></tr><tr><td>OpenFunGraph (Ours)</td><td>68.3</td><td>73.0</td><td>88.1</td><td>96.2</td><td>60.4</td><td>70.3</td><td>45.8</td><td>49.3</td><td>65.1</td><td>91.4</td><td>29.8</td><td>45.0</td></tr></table>
<table><tbody><tr><td rowspan="3">方法</td><td colspan="6">SceneFun3D [17]</td><td colspan="6">FunGraph3D（我们）</td></tr><tr><td colspan="2">节点关联</td><td colspan="2">边预测</td><td colspan="2">总体三元组</td><td colspan="2">节点关联</td><td colspan="2">边 e 预测</td><td colspan="2">总体三元组</td></tr><tr><td>R@5</td><td>R@10</td><td>R@5</td><td>R@10</td><td>R@5</td><td>R@10</td><td>R@5</td><td>R@10</td><td>R@5</td><td>R@10</td><td>R@5</td><td>R@10</td></tr><tr><td>Open3DSG* [41]</td><td>47.2</td><td>58.0</td><td>69.2</td><td>78.8</td><td>32.7</td><td>45.7</td><td>22.8</td><td>36.7</td><td>47.9</td><td>55.9</td><td>10.5</td><td>20.0</td></tr><tr><td>Open3DSG*† [41]</td><td>33.6</td><td>38.8</td><td>64.4</td><td>72.3</td><td>21.6</td><td>28.1</td><td>15.7</td><td>24.2</td><td>46.6</td><td>55.7</td><td>7.3</td><td>13.5</td></tr><tr><td>概念图* [27]</td><td>5.6</td><td>6.8</td><td>80.2</td><td>95.0</td><td>4.7</td><td>6.4</td><td>1.9</td><td>2.8</td><td>51.5</td><td>84.6</td><td>1.1</td><td>2.5</td></tr><tr><td>概念图* [27] + IED</td><td>45.4</td><td>49.3</td><td>75.6</td><td>90.9</td><td>34.3</td><td>44.5</td><td>18.8</td><td>22.8</td><td>46.1</td><td>79.7</td><td>10.3</td><td>18.9</td></tr><tr><td>OpenFunGraph（我们）</td><td>68.3</td><td>73.0</td><td>88.1</td><td>96.2</td><td>60.4</td><td>70.3</td><td>45.8</td><td>49.3</td><td>65.1</td><td>91.4</td><td>29.8</td><td>45.0</td></tr></tbody></table>


Tab. 2. Triplet evaluation on the SceneFun3D [17] and FunGraph3D datasets. All marks keep the same meaning with Table 1. Node Assoc. refers to the node association metric while Edge Pred. means the edge prediction metric.
表2。SceneFun3D [17] 和 FunGraph3D 数据集上的三元组评估。所有标记的含义与表1相同。Node Assoc. 指节点关联指标，而 Edge Pred. 指边预测指标。


<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/06/2026_06_11__23_42_36_a83bec.jpg"/>



Fig. 6. Functional 3D Scene Graphs for Robotic Manipulation. Left: 3D scene and functional graph generated after querying 'turning on the light.' Right: Robot interacting with scene elements as guided by the functional scene graph.
图6。用于机器人操作的功能性3D场景图。左：在查询“turning on the light”后生成的3D场景和功能图。右：机器人在功能场景图的引导下与场景元素交互。


### 6.4. Downstream Applications
### 6.4. 下游应用


We showcase the versatility of the proposed functional 3D scene graph representation in downstream applications that require complex reasoning about indoor functionalities and task-oriented interactions.
我们展示了所提出的功能性 3D 场景图表示在下游应用中的多样性，这些应用需要对室内功能和面向任务的交互进行复杂推理。


3D inventory question answering. To enable functional reasoning, we convert the graph structure into a JSON list that the LLM can easily query. With this list, the LLM can answer questions such as "How can I turn on the ceiling light?". Using the functional 3D scene graph's nodes (objects, interactive elements) and edges (functional relationships), the LLM can provide responses such as "You can turn on the ceiling light using the light switch plate located at position [0.611, 0.113, 0.732]. From the provided JSON list, we can see the light switch plate with id 0 has the highest confidence level of 0.8 with the ceiling light fixture."
3D 库存问答。为实现功能推理，我们将图结构转换为 LLM 可轻松查询的 JSON 列表。借助该列表，LLM 可以回答诸如“如何打开天花板灯？”之类的问题。利用功能性 3D 场景图的节点（物体、交互元素）和边（功能关系），LLM 可以给出如下回答：“你可以使用位于 [0.611, 0.113, 0.732] 的灯开关面板来打开天花板灯。从提供的 JSON 列表中，我们可以看到 id 为 0 的灯开关面板与天花板灯具的置信度最高，为 0.8。”


<table><tr><td rowspan="2">Experiments</td><td colspan="2">Overall Nodes</td><td colspan="2">Overall Triplets</td></tr><tr><td>R@3</td><td>R@10</td><td>R@5</td><td>R@10</td></tr><tr><td>w/o prompts for element detection</td><td>59.3</td><td>68.7</td><td>48.3</td><td>59.9</td></tr><tr><td>w/o sequential edge reasoning*</td><td>73.0</td><td>82.8</td><td>34.8</td><td>48.9</td></tr><tr><td>w/o confidence-aware edge reasoning*</td><td>73.0</td><td>82.8</td><td>56.0</td><td>65.1</td></tr><tr><td>Ours</td><td>73.0</td><td>82.8</td><td>60.470.3</td><td></td></tr><tr><td>w/o prompts for element detection</td><td>49.9</td><td>59.1</td><td>23.1</td><td>37.6</td></tr><tr><td>w/o sequential edge reasoning*</td><td>55.5</td><td>65.8</td><td>20.2</td><td>33.8</td></tr><tr><td>w/o confidence-aware edge reasoning*</td><td>55.5</td><td>65.8</td><td>26.8</td><td>40.1</td></tr><tr><td>Ours</td><td>55.5</td><td>65.8</td><td>29.8</td><td>45.0</td></tr></table>
<table><tbody><tr><td rowspan="2">实验</td><td colspan="2">总节点数</td><td colspan="2">总三元组数</td></tr><tr><td>R@3</td><td>R@10</td><td>R@5</td><td>R@10</td></tr><tr><td>不使用元素检测提示</td><td>59.3</td><td>68.7</td><td>48.3</td><td>59.9</td></tr><tr><td>不使用顺序边推理*</td><td>73.0</td><td>82.8</td><td>34.8</td><td>48.9</td></tr><tr><td>不使用置信度感知边推理*</td><td>73.0</td><td>82.8</td><td>56.0</td><td>65.1</td></tr><tr><td>我们的方法</td><td>73.0</td><td>82.8</td><td>60.470.3</td><td></td></tr><tr><td>不使用元素检测提示</td><td>49.9</td><td>59.1</td><td>23.1</td><td>37.6</td></tr><tr><td>不使用顺序边推理*</td><td>55.5</td><td>65.8</td><td>20.2</td><td>33.8</td></tr><tr><td>不使用置信度感知边推理*</td><td>55.5</td><td>65.8</td><td>26.8</td><td>40.1</td></tr><tr><td>我们的方法</td><td>55.5</td><td>65.8</td><td>29.8</td><td>45.0</td></tr></tbody></table>


Tab. 3. Ablation study on SceneFun3D [17] (Top) and our Fun-Graph3D (Bottom). Note that edge reasoning (*) impacts only the triplet metric and does not affect node recognition performance.
表 3。SceneFun3D [17]（上）与我们的 Fun-Graph3D（下）的消融研究。注意，边推理（*）仅影响三元组指标，不影响节点识别性能。


Robotic manipulation. The functional 3D scene graph also supports robotic manipulation [43, 108] for user queries that involve functional reasoning, as illustrated in Figure 6. Similar to inventory question answering, the LLM queries the JSON list to locate the interactive element referenced in the query. The robot then navigates to and interacts with the element using the methods described in [43].
机器人操控。对于包含功能推理的用户查询，功能 3D 场景图也支持机器人操控 [43, 108]，如图 6 所示。与库存问答类似，LLM 会查询 JSON 列表以定位查询中所指的交互元素。随后，机器人按 [43] 中描述的方法导航到该元素并与之交互。


## 7. Conclusion
## 7. 结论


We introduce Functional 3D Scene Graphs, a novel representation that jointly models objects, interactive elements, and their functional relationships in 3D indoor environments. Our open-vocabulary pipeline leverages the common-sense knowledge of foundation models to infer functional 3D scene graphs and enable flexible querying. To support systematic benchmarking, we develop a high-fidelity dataset of real-world 3D indoor environments with multi-modal data and functional annotations. Experiments on this and existing datasets show that our method significantly outperforms baselines. We further demonstrate the versatility of our representation for downstream tasks such as 3D question answering and robotic manipulation.
我们提出了功能性3D场景图，一种新颖的表示方式，可在3D室内环境中联合建模物体、交互元素及其功能关系。我们的开放词汇流程借助基础模型的常识知识来推断功能性3D场景图，并支持灵活查询。为支持系统性基准评测，我们构建了一个高保真的真实世界3D室内环境数据集，包含多模态数据和功能标注。该数据集及现有数据集上的实验表明，我们的方法显著优于基线。我们还展示了该表示在3D问答和机器人操作等下游任务中的多样适用性。


Acknowledgments. We would like to thank colleagues and friends who helped us capture the data of Fun-Graph3D: Christine Engelmann, Dominik Faerber, Elisabetta Fedele, Xudong Jiang, Xin Kong, Aoxue Liu and Houssam Naous. This work was supported by the Swiss National Science Foundation Advanced Grant 216260: "Beyond Frozen Worlds: Capturing Functional 3D Digital Twins from the Real World". AD is supported by the Max Planck ETH Center for Learning Systems (CLS) and FE by an SNSF PostDoc.Mobility Fellowship.
致谢。我们感谢帮助我们采集 Fun-Graph3D 数据的同事和朋友：Christine Engelmann、Dominik Faerber、Elisabetta Fedele、Xudong Jiang、Xin Kong、Aoxue Liu 和 Houssam Naous。本研究得到瑞士国家科学基金会高级资助 216260：“Beyond Frozen Worlds: Capturing Functional 3D Digital Twins from the Real World”的支持。AD 由马克斯·普朗克—苏黎世联邦理工学习系统中心（CLS）资助，FE 由 SNSF PostDoc.Mobility 奖学金资助。


## References
## 参考文献


[1] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023. 3, 4
[1] Josh Achiam，Steven Adler，Sandhini Agarwal，Lama Ahmad，Ilge Akkaya，Florencia Leoni Aleman，Diogo Almeida，Janko Altenschmidt，Sam Altman，Shyamal Anadkat，等。Gpt-4 技术报告。arXiv 预印本 arXiv:2303.08774，2023。3，4


[2] Christopher Agia, Krishna Murthy Jatavallabhula, Mohamed Khodeir, Ondrej Miksik, Vibhav Vineet, Mustafa Mukadam, Liam Paull, and Florian Shkurti. Taskography: Evaluating robot task planning over large $3\mathrm{\;d}$ scene graphs. In Conference on Robot Learning (CoRL), 2022. 1
[2] Christopher Agia，Krishna Murthy Jatavallabhula，Mohamed Khodeir，Ondrej Miksik，Vibhav Vineet，Mustafa Mukadam，Liam Paull，以及 Florian Shkurti。Taskography：在大型 $3\mathrm{\;d}$ 场景图上评估机器人任务规划。在机器人学习会议（CoRL）中，2022。1


[3] Iro Armeni, Ozan Sener, Amir R Zamir, Helen Jiang, Ioan-nis Brilakis, Martin Fischer, and Silvio Savarese. 3d semantic parsing of large-scale indoor spaces. In International Conference on Computer Vision and Pattern Recognition (CVPR), 2016. 2
[3] Iro Armeni，Ozan Sener，Amir R Zamir，Helen Jiang，Ioan-nis Brilakis，Martin Fischer，以及 Silvio Savarese。大规模室内空间的三维语义解析。在计算机视觉与模式识别国际会议（CVPR）中，2016。2


[4] Iro Armeni, Zhi-Yang He, JunYoung Gwak, Amir R Zamir, Martin Fischer, Jitendra Malik, and Silvio Savarese. 3d scene graph: A structure for unified semantics, 3d space, and camera. In International Conference on Computer Vision (ICCV), 2019. 1, 2
[4] Iro Armeni，Zhi-Yang He，JunYoung Gwak，Amir R Zamir，Martin Fischer，Jitendra Malik，以及 Silvio Savarese。3d 场景图：用于统一语义、三维空间和相机的结构。在计算机视觉国际会议（ICCV）中，2019。1，2


[5] Matan Atzmon, Haggai Maron, and Yaron Lipman. Point convolutional neural networks by extension operators. ACM Transactions On Graphics (TOG), 2018. 2
[5] Matan Atzmon，Haggai Maron，以及 Yaron Lipman。通过扩展算子实现点卷积神经网络。ACM 图形学汇刊（TOG），2018。2


[6] Prithviraj Banerjee, Sindi Shkodrani, Pierre Moulon, Shreyas Hampali, Fan Zhang, Jade Fountain, Edward Miller, Selen Basol, Richard Newcombe, Robert Wang, et al. Introducing hot3d: An egocentric dataset for 3d hand and object tracking. arXiv preprint arXiv:2406.09598, 2024. 2
[6] Prithviraj Banerjee，Sindi Shkodrani，Pierre Moulon，Shreyas Hampali，Fan Zhang，Jade Fountain，Edward Miller，Selen Basol，Richard Newcombe，Robert Wang，等。介绍 hot3d：用于三维手部与物体跟踪的以视角为中心的数据集。arXiv 预印本 arXiv:2406.09598，2024。2


[7] Gilad Baruch, Zhuoyuan Chen, Afshin Dehghan, Tal Dimry, Yuri Feigin, Peter Fu, Thomas Gebauer, Brandon Joffe, Daniel Kurz, Arik Schwartz, et al. ARKitScenes: A diverse real-world dataset for 3d indoor scene understanding using mobile RGB-D data. In International Conference on Neural Information Processing Systems (NeurIPS), 2021. 2, 5
[7] Gilad Baruch，Zhuoyuan Chen，Afshin Dehghan，Tal Dimry，Yuri Feigin，Peter Fu，Thomas Gebauer，Brandon Joffe，Daniel Kurz，Arik Schwartz，等。ARKitScenes：利用移动端 RGB-D 数据进行三维室内场景理解的多样化真实世界数据集。在神经信息处理系统国际会议（NeurIPS）中，2021。2，5


[8] Valentin Bieri, Marco Zamboni, Nicolas S. Blumer, Qingx-uan Chen, and Francis Engelmann. OpenCity3D: 3D Urban Scene Understanding with Vision-Language Models. In IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), 2025. 2
[8] Valentin Bieri，Marco Zamboni，Nicolas S. Blumer，Qingx-uan Chen，以及 Francis Engelmann。OpenCity3D：借助视觉-语言模型理解三维城市场景。在 IEEE/CVF 冬季计算机视觉应用会议（WACV）中，2025。2


[9] Rishi Bommasani, Drew A Hudson, Ehsan Adeli, Russ Altman, Simran Arora, Sydney von Arx, Michael S Bernstein, Jeannette Bohg, Antoine Bosselut, Emma Brunskill, et al. On the opportunities and risks of foundation models. arXiv preprint arXiv:2108.07258, 2021. 3
[9] Rishi Bommasani，Drew A Hudson，Ehsan Adeli，Russ Altman，Simran Arora，Sydney von Arx，Michael S Bernstein，Jeannette Bohg，Antoine Bosselut，Emma Brunskill，等。关于基础模型的机遇与风险。arXiv 预印本 arXiv:2108.07258，2021。3


[10] Angel Chang, Angela Dai, Thomas Funkhouser, Maciej Halber, Matthias Niessner, Manolis Savva, Shuran Song, Andy Zeng, and Yinda Zhang. Matterport3d: Learning from rgb-d data in indoor environments. International Conference on 3d Vision (3dV), 2017. 2
[10] Angel Chang，Angela Dai，Thomas Funkhouser，Maciej Halber，Matthias Niessner，Manolis Savva，Shuran Song，Andy Zeng，以及 Yinda Zhang。Matterport3d：从 rgb-d 数据中学习室内环境。在三维视觉国际会议（3dV）中，2017。2


[11] Lianggangxu Chen, Xuejiao Wang, Jiale Lu, Shaohui Lin, Changbo Wang, and Gaoqi He. Clip-driven open-vocabulary 3d scene graph generation via cross-modality contrastive learning. In International Conference on Computer Vision and Pattern Recognition (CVPR), 2024. 1
[11] Lianggangxu Chen，Xuejiao Wang，Jiale Lu，Shaohui Lin，Changbo Wang，以及 Gaoqi He。通过跨模态对比学习实现基于剪辑驱动的开放词汇三维场景图生成。在计算机视觉与模式识别国际会议（CVPR）中，2024。1


[12] Zerui Chen, Yana Hasson, Cordelia Schmid, and Ivan Laptev. Alignsdf: Pose-aligned signed distance fields for hand-object reconstruction. In European Conference on Computer Vision (ECCV), 2022. 2
[12] Zerui Chen，Yana Hasson，Cordelia Schmid，以及 Ivan Laptev。Alignsdf：用于手-物重建的姿态对齐有符号距离场。在计算机视觉欧洲会议（ECCV）中，2022。2


[13] Woojin Cho, Jihyun Lee, Minjae Yi, Minje Kim, Taeyun Woo, Donghwan Kim, Taewook Ha, Hyokeun Lee, Je-Hwan Ryu, Woontack Woo, et al. Dense hand-object (ho) graspnet with full grasping taxonomy and dynamics. European Conference on Computer Vision (ECCV), 2024. 2
[13] Woojin Cho，Jihyun Lee，Minjae Yi，Minje Kim，Taeyun Woo，Donghwan Kim，Taewook Ha，Hyokeun Lee，Je-Hwan Ryu，Woontack Woo，等。具有完整抓取分类与动力学的密集手-物抓取（ho）抓取网络。在计算机视觉欧洲会议（ECCV）中，2024。2


[14] Christopher Choy, JunYoung Gwak, and Silvio Savarese. 4d spatio-temporal convnets: Minkowski convolutional neural networks. In International Conference on Computer Vision and Pattern Recognition (CVPR), 2019. 1, 2
[14] Christopher Choy, JunYoung Gwak, and Silvio Savarese. 4d时空卷积网络：Minkowski卷积神经网络。发表于计算机视觉与模式识别国际会议（CVPR），2019年。1，2


[15] Angela Dai, Angel X Chang, Manolis Savva, Maciej Hal-ber, Thomas Funkhouser, and Matthias Nießner. Scannet: Richly-annotated 3d reconstructions of indoor scenes. In International Conference on Computer Vision and Pattern Recognition (CVPR), 2017. 2
[15] Angela Dai, Angel X Chang, Manolis Savva, Maciej Hal-ber, Thomas Funkhouser, and Matthias Nießner. Scannet：对室内场景进行丰富标注的3d重建。发表于计算机视觉与模式识别国际会议（CVPR），2017年。2


[16] Alexandros Delitzas, Maria Parelli, Nikolas Hars, Geor-gios Vlassis, Sotirios-Konstantinos Anagnostidis, Gregor Bachmann, and Thomas Hofmann. Multi-clip: Contrastive vision-language pre-training for question answering tasks in 3d scenes. In British Machine Vision Conference (BMVC), 2023. 2
[16] Alexandros Delitzas, Maria Parelli, Nikolas Hars, Geor-gios Vlassis, Sotirios-Konstantinos Anagnostidis, Gregor Bachmann, and Thomas Hofmann. Multi-clip：用于3d场景问答任务的对比视觉-语言预训练。发表于英国机器视觉会议（BMVC），2023年。2


[17] Alexandros Delitzas, Ayca Takmaz, Federico Tombari, Robert Sumner, Marc Pollefeys, and Francis Engelmann. Scenefun3d: Fine-grained functionality and affordance understanding in 3d scenes. In International Conference on Computer Vision and Pattern Recognition (CVPR), 2024. 1,2,3,5,6,7,8
[17] Alexandros Delitzas, Ayca Takmaz, Federico Tombari, Robert Sumner, Marc Pollefeys, and Francis Engelmann. Scenefun3d：3d场景中的精细功能性与可用性理解。发表于计算机视觉与模式识别国际会议（CVPR），2024年。1，2，3，5，6，7，8


[18] Shengheng Deng, Xun Xu, Chaozheng Wu, Ke Chen, and Kui Jia. 3d affordancenet: A benchmark for visual object affordance understanding. In International Conference on Computer Vision and Pattern Recognition (CVPR), 2021. 2
[18] Shengheng Deng, Xun Xu, Chaozheng Wu, Ke Chen, and Kui Jia. 3d affordancenet：视觉物体可用性理解的基准。发表于计算机视觉与模式识别国际会议（CVPR），2021年。2


[19] Daniel DeTone, Tomasz Malisiewicz, and Andrew Rabi-novich. Superpoint: Self-supervised interest point detection and description. In International Conference on Computer Vision and Pattern Recognition (CVPR) Workshops, 2018. 6
[19] Daniel DeTone, Tomasz Malisiewicz, and Andrew Rabi-novich. Superpoint：自监督兴趣点检测与描述。发表于计算机视觉与模式识别国际会议（CVPR）工作坊，2018年。6


[20] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. In Proceedings of naacL-HLT, 2019. 6
[20] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert：用于语言理解的深度双向Transformer预训练。发表于naacL-HLT会议论文集，2019年。6


[21] Helisa Dhamo, Fabian Manhardt, Nassir Navab, and Federico Tombari. Graph-to-3d: End-to-end generation and manipulation of $3\mathrm{\;d}$ scenes using scene graphs. In International Conference on Computer Vision (ICCV), 2021. 1, 2
[21] Helisa Dhamo, Fabian Manhardt, Nassir Navab, and Federico Tombari. Graph-to-3d：使用场景图生成并操控$3\mathrm{\;d}$场景的端到端方法。发表于计算机视觉国际会议（ICCV），2021年。1，2


[22] Thanh-Toan Do, Anh Nguyen, and Ian Reid. Affor-dancenet: An end-to-end deep learning approach for object affordance detection. In International Conference on Robotics and Automation (ICRA), 2018. 2
[22] Thanh-Toan Do, Anh Nguyen, and Ian Reid. Affor-dancenet：一种用于物体可用性检测的端到端深度学习方法。发表于机器人与自动化国际会议（ICRA），2018年。2


[23] Francis Engelmann, Martin Bokeloh, Alireza Fathi, Bastian Leibe, and Matthias Nießner. 3d-mpa: Multi-proposal aggregation for $3\mathrm{\;d}$ semantic instance segmentation. In International Conference on Computer Vision and Pattern Recognition (CVPR), 2020. 2
[23] Francis Engelmann, Martin Bokeloh, Alireza Fathi, Bastian Leibe, and Matthias Nießner. 3d-mpa：用于$3\mathrm{\;d}$语义实例分割的多提案聚合。发表于计算机视觉与模式识别国际会议（CVPR），2020年。2


[24] Francis Engelmann, Fabian Manhardt, Michael Niemeyer, Keisuke Tateno, Marc Pollefeys, and Federico Tombari. Opennerf: Open set 3d neural scene segmentation with pixel-wise features and rendered novel views. International Conference on Learning Representations (ICLR), 2024. 2
[24] Francis Engelmann, Fabian Manhardt, Michael Niemeyer, Keisuke Tateno, Marc Pollefeys, and Federico Tombari. Opennerf：带像素级特征并渲染新视角的开放集3d神经场景分割。发表于国际学习表征会议（ICLR），2024年。2


[25] Zicong Fan, Maria Parelli, Maria Eleni Kadoglou, Xu Chen, Muhammed Kocabas, Michael J Black, and Otmar Hilliges. Hold: Category-agnostic 3d reconstruction of interacting hands and objects from video. In International Conference on Computer Vision and Pattern Recognition (CVPR), 2024. 2
[25] Zicong Fan, Maria Parelli, Maria Eleni Kadoglou, Xu Chen, Muhammed Kocabas, Michael J Black, and Otmar Hilliges. Hold：从视频中交互的手与物体进行类别无关的3d重建。发表于计算机视觉与模式识别国际会议（CVPR），2024年。2


[26] Kuan Fang, Te-Lin Wu, Daniel Yang, Silvio Savarese, and Joseph J Lim. Demo2vec: Reasoning object affordances from online videos. In International Conference on Computer Vision and Pattern Recognition (CVPR), 2018. 2
[26] Kuan Fang, Te-Lin Wu, Daniel Yang, Silvio Savarese, and Joseph J Lim. Demo2vec：从在线视频推理物体可用性。发表于计算机视觉与模式识别国际会议（CVPR），2018年。2


[27] Qiao Gu, Ali Kuwajerwala, Sacha Morin, Krishna Murthy Jatavallabhula, Bipasha Sen, Aditya Agarwal, Corban Rivera, William Paul, Kirsty Ellis, Rama Chellappa, et al. ConceptGraphs: Open-vocabulary 3d scene graphs for perception and planning. In International Conference on Robotics and Automation (ICRA), 2024. 1, 3, 4, 6, 7, 8
[27] Qiao Gu, Ali Kuwajerwala, Sacha Morin, Krishna Murthy Jatavallabhula, Bipasha Sen, Aditya Agarwal, Corban Rivera, William Paul, Kirsty Ellis, Rama Chellappa, 等。ConceptGraphs：用于感知与规划的开放词汇3d场景图。发表于机器人与自动化国际会议（ICRA），2024年。1，3，4，6，7，8


[28] Lei Han, Tian Zheng, Lan Xu, and Lu Fang. Occuseg: Occupancy-aware 3d instance segmentation. In International Conference on Computer Vision and Pattern Recognition (CVPR), 2020. 2
[28] Lei Han, Tian Zheng, Lan Xu, and Lu Fang. Occuseg：面向占据感知的3D实例分割。发表于国际计算机视觉与模式识别会议（CVPR），2020年。2


[29] Ji Hou, Angela Dai, and Matthias Nießner. 3d-sis: 3d semantic instance segmentation of rgb-d scans. In International Conference on Computer Vision and Pattern Recognition (CVPR), 2019. 2
[29] Ji Hou, Angela Dai, and Matthias Nießner. 3d-sis：RGB-D扫描的3D语义实例分割。发表于国际计算机视觉与模式识别会议（CVPR），2019年。2


[30] Joy Hsu, Jiayuan Mao, and Jiajun Wu. Ns3d: Neuro-symbolic grounding of 3d objects and relations. In International Conference on Computer Vision and Pattern Recognition (CVPR), 2023. 2
[30] Joy Hsu, Jiayuan Mao, and Jiajun Wu. Ns3d：3D对象与关系的神经符号式 grounding。发表于国际计算机视觉与模式识别会议（CVPR），2023年。2


[31] Zeyu Hu, Xuyang Bai, Jiaxiang Shang, Runze Zhang, Jiayu Dong, Xin Wang, Guangyuan Sun, Hongbo Fu, and Chiew-Lan Tai. Vmnet: Voxel-mesh network for geodesic-aware 3d semantic segmentation. In International Conference on Computer Vision (ICCV), 2021. 2
[31] Zeyu Hu, Xuyang Bai, Jiaxiang Shang, Runze Zhang, Jiayu Dong, Xin Wang, Guangyuan Sun, Hongbo Fu, and Chiew-Lan Tai. Vmnet：面向地理距离感知的3D语义分割的体素-网格网络。发表于国际计算机视觉会议（ICCV），2021年。2


[32] Binh-Son Hua, Minh-Khoi Tran, and Sai-Kit Yeung. Pointwise convolutional neural networks. In International Conference on Computer Vision and Pattern Recognition (CVPR), 2018.
[32] Binh-Son Hua, Minh-Khoi Tran, and Sai-Kit Yeung. 点式卷积神经网络。发表于国际计算机视觉与模式识别会议（CVPR），2018年。


[33] Rui Huang, Songyou Peng, Ayca Takmaz, Federico Tombari, Marc Pollefeys, Shiji Song, Gao Huang, and Francis Engelmann. Segment3d: Learning fine-grained class-agnostic 3d segmentation without manual labels. European Conference on Computer Vision (ECCV), 2024. 2
[33] Rui Huang, Songyou Peng, Ayca Takmaz, Federico Tombari, Marc Pollefeys, Shiji Song, Gao Huang, and Francis Engelmann. Segment3d：无需人工标注的细粒度类别无关3D分割学习。欧洲计算机视觉会议（ECCV），2024年。2


[34] Shijia Huang, Yilun Chen, Jiaya Jia, and Liwei Wang. Multi-view transformer for 3d visual grounding. In International Conference on Computer Vision and Pattern Recognition (CVPR), 2022. 2
[34] Shijia Huang, Yilun Chen, Jiaya Jia, and Liwei Wang. 用于3D视觉定位的多视图Transformer。发表于国际计算机视觉与模式识别会议（CVPR），2022年。2


[35] Xinyu Huang, Yi-Jie Huang, Youcai Zhang, Weiwei Tian, Rui Feng, Yuejie Zhang, Yanchun Xie, Yaqian Li, and Lei Zhang. Open-set image tagging with multi-grained text supervision. arXiv e-prints, 2023. 4
[35] Xinyu Huang, Yi-Jie Huang, Youcai Zhang, Weiwei Tian, Rui Feng, Yuejie Zhang, Yanchun Xie, Yaqian Li, and Lei Zhang. 利用多粒度文本监督进行开放集图像标注。arXiv预印本，2023年。4


[36] Krishna Murthy Jatavallabhula, Alihusein Kuwajerwala, Qiao Gu, Mohd Omama, Tao Chen, Alaa Maalouf, Shuang Li, Ganesh Iyer, Soroush Saryazdi, Nikhil Keetha, et al. Conceptfusion: Open-set multimodal 3d mapping. ICRA2023 Workshop on Pretraining for Robotics (PT4R), 2023. 2
[36] Krishna Murthy Jatavallabhula, Alihusein Kuwajerwala, Qiao Gu, Mohd Omama, Tao Chen, Alaa Maalouf, Shuang Li, Ganesh Iyer, Soroush Saryazdi, Nikhil Keetha, et al. Conceptfusion：开放集多模态3D建图。ICRA2023机器人预训练研讨会（PT4R），2023年。2


[37] Guangda Ji, Silvan Weder, Francis Engelmann, Marc Polle-feys, and Hermann Blum. Arkit labelmaker: A new scale for indoor 3d scene understanding. International Conference on Computer Vision and Pattern Recognition (CVPR), 2025. 2
[37] Guangda Ji, Silvan Weder, Francis Engelmann, Marc Polle-feys, and Hermann Blum. Arkit labelmaker：室内3D场景理解的新规模。国际计算机视觉与模式识别会议（CVPR），2025年。2


[38] Li Jiang, Hengshuang Zhao, Shaoshuai Shi, Shu Liu, Chi-Wing Fu, and Jiaya Jia. Pointgroup: Dual-set point grouping for 3d instance segmentation. In International Conference on Computer Vision and Pattern Recognition (CVPR), 2020. 2
[38] Li Jiang, Hengshuang Zhao, Shaoshuai Shi, Shu Liu, Chi-Wing Fu, and Jiaya Jia. Pointgroup：用于3D实例分割的双集合点分组。发表于国际计算机视觉与模式识别会议（CVPR），2020年。2


[39] Justin Kerr, Chung Min Kim, Ken Goldberg, Angjoo Kanazawa, and Matthew Tancik. Lerf: Language embedded radiance fields. In International Conference on Computer Vision (ICCV), 2023. 2, 4
[39] Justin Kerr, Chung Min Kim, Ken Goldberg, Angjoo Kanazawa, and Matthew Tancik. Lerf：语言嵌入辐射场。发表于国际计算机视觉会议（ICCV），2023年。2, 4


[40] Sebastian Koch, Pedro Hermosilla, Narunas Vaskevicius, Mirco Colosi, and Timo Ropinski. Lang3dsg: Language-based contrastive pre-training for 3d scene graph prediction. In International Conference on 3d Vision (3dV), 2024. 1,2
[40] Sebastian Koch, Pedro Hermosilla, Narunas Vaskevicius, Mirco Colosi, and Timo Ropinski. Lang3dsg：用于3D场景图预测的基于语言的对比式预训练。发表于3D视觉国际会议（3dV），2024年。1,2


[41] Sebastian Koch, Narunas Vaskevicius, Mirco Colosi, Pedro Hermosilla, and Timo Ropinski. Open3dsg: Open-vocabulary 3d scene graphs from point clouds with queryable objects and open-set relationships. In International Conference on Computer Vision and Pattern Recognition (CVPR), 2024. 1, 2, 3, 6, 7, 8
[41] Sebastian Koch, Narunas Vaskevicius, Mirco Colosi, Pedro Hermosilla, and Timo Ropinski. Open3dsg：来自点云的开放词汇3D场景图，具有可查询对象和开放集关系。发表于国际计算机视觉与模式识别会议（CVPR），2024年。1, 2, 3, 6, 7, 8


[42] Loic Landrieu and Martin Simonovsky. Large-scale point cloud semantic segmentation with superpoint graphs. In International Conference on Computer Vision and Pattern Recognition (CVPR), 2018. 2
[42] Loic Landrieu and Martin Simonovsky. 利用超级点图进行大规模点云语义分割。发表于国际计算机视觉与模式识别会议（CVPR），2018年。2


[43] Oliver Lemke, Zuria Bauer, René Zurbrügg, Marc Polle-feys, Francis Engelmann, and Hermann Blum. Spot-Compose: A framework for open-vocabulary object retrieval and drawer manipulation in point clouds. In International Conference on Robotics and Automation (ICRA), 2024. 8
[43] Oliver Lemke, Zuria Bauer, René Zurbrügg, Marc Polle-feys, Francis Engelmann, 和 Hermann Blum。Spot-Compose：用于点云中开放词汇对象检索与抽屉操控的框架。发表于机器人与自动化国际会议（ICRA），2024。8


[44] Qi Li, Kaichun Mo, Yanchao Yang, Hang Zhao, and Leonidas Guibas. IFR-Explore: Learning inter-object functional relationships in 3d indoor scenes. International Conference on Learning Representations (ICLR), 2022. 2
[44] Qi Li, Kaichun Mo, Yanchao Yang, Hang Zhao, 和 Leonidas Guibas。IFR-Explore：在3D室内场景中学习对象间的功能关系。发表于国际学习表征会议（ICLR），2022。2


[45] Yangyan Li, Rui Bu, Mingchao Sun, Wei Wu, Xinhan Di, and Baoquan Chen. Pointcnn: Convolution on x-transformed points. International Conference on Neural Information Processing Systems (NeurIPS), 2018. 2
[45] Yangyan Li, Rui Bu, Mingchao Sun, Wei Wu, Xinhan Di, 和 Baoquan Chen。Pointcnn：对x变换点的卷积。发表于神经信息处理系统国际会议（NeurIPS），2018。2


[46] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. In International Conference on Neural Information Processing Systems (NeurIPS), 2023. 4
[46] Haotian Liu, Chunyuan Li, Qingyang Wu, 和 Yong Jae Lee。视觉指令微调。发表于神经信息处理系统国际会议（NeurIPS），2023。4


[47] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. In International Conference on Computer Vision and Pattern Recognition (CVPR), 2024.
[47] Haotian Liu, Chunyuan Li, Yuheng Li, 和 Yong Jae Lee。结合视觉指令微调的改进基线。发表于计算机视觉与模式识别国际会议（CVPR），2024。


[48] Haotian Liu, Chunyuan Li, Yuheng Li, Bo Li, Yuanhan Zhang, Sheng Shen, and Yong Jae Lee. Llava-next: Improved reasoning, ocr, and world knowledge, 2024. 3, 4
[48] Haotian Liu, Chunyuan Li, Yuheng Li, Bo Li, Yuanhan Zhang, Sheng Shen, 和 Yong Jae Lee。Llava-next：改进推理、OCR与世界知识，2024。3, 4


[49] Shilong Liu, Zhaoyang Zeng, Tianhe Ren, Feng Li, Hao Zhang, Jie Yang, Chunyuan Li, Jianwei Yang, Hang Su, Jun Zhu, et al. Grounding dino: Marrying dino with grounded pre-training for open-set object detection. European Conference on Computer Vision (ECCV), 2024. 3, 4
[49] Shilong Liu, Zhaoyang Zeng, Tianhe Ren, Feng Li, Hao Zhang, Jie Yang, Chunyuan Li, Jianwei Yang, Hang Su, Jun Zhu 等。Grounding dino：将dino与基于标注的预训练结合，用于开放集目标检测。发表于欧洲计算机视觉会议（ECCV），2024。3, 4


[50] Cewu Lu, Ranjay Krishna, Michael Bernstein, and Li Fei-Fei. Visual relationship detection with language priors. In European Conference on Computer Vision (ECCV), 2016. 6
[50] Cewu Lu, Ranjay Krishna, Michael Bernstein, 和 Li Fei-Fei。带语言先验的视觉关系检测。发表于欧洲计算机视觉会议（ECCV），2016。6


[51] Yang Miao, Francis Engelmann, Olga Vysotska, Federico Tombari, Marc Pollefeys, and Dániel Béla Baráth. Scene-GraphLoc: Cross-modal coarse visual localization on 3d scene graphs. In European Conference on Computer Vision (ECCV), 2024. 1
[51] Yang Miao, Francis Engelmann, Olga Vysotska, Federico Tombari, Marc Pollefeys, 和 Dániel Béla Baráth。Scene-GraphLoc：在3D场景图上的跨模态粗粒度视觉定位。发表于欧洲计算机视觉会议（ECCV），2024。1


[52] Kaichun Mo, Yuzhe Qin, Fanbo Xiang, Hao Su, and Leonidas Guibas. O2o-afford: Annotation-free large-scale object-object affordance learning. In Conference on Robot Learning (CoRL), 2022. 2
[52] Kaichun Mo, Yuzhe Qin, Fanbo Xiang, Hao Su, 和 Leonidas Guibas。O2o-afford：无标注的大规模对象-对象可用性学习。发表于机器人学习会议（CoRL），2022。2


[53] Tushar Nagarajan and Kristen Grauman. Learning affor-dance landscapes for interaction exploration in 3d environments. International Conference on Neural Information Processing Systems (NeurIPS), 2020. 2
[53] Tushar Nagarajan 和 Kristen Grauman。为3D环境中的交互探索学习affor-dance地形。发表于神经信息处理系统国际会议（NeurIPS），2020。2


[54] Tushar Nagarajan, Christoph Feichtenhofer, and Kristen Grauman. Grounded human-object interaction hotspots from video. In International Conference on Computer Vision (ICCV), 2019. 2
[54] Tushar Nagarajan, Christoph Feichtenhofer, 和 Kristen Grauman。从视频中获取有依据的真人-物体交互热点。发表于计算机视觉国际会议（ICCV），2019。2


[55] Maria Parelli, Alexandros Delitzas, Nikolas Hars, Geor-gios Vlassis, Sotirios Anagnostidis, Gregor Bachmann, and Thomas Hofmann. CLIP-Guided Vision-Language PreTraining for Question Answering in 3D Scenes. In International Conference on Computer Vision and Pattern Recognition (CVPR) Workshops, 2023. 2
[55] Maria Parelli, Alexandros Delitzas, Nikolas Hars, Geor-gios Vlassis, Sotirios Anagnostidis, Gregor Bachmann, 和 Thomas Hofmann。用于3D场景问答的CLIP引导视觉-语言预训练。发表于计算机视觉与模式识别会议（CVPR）工作坊，2023。2


[56] Songyou Peng, Kyle Genova, Chiyu Jiang, Andrea Tagliasacchi, Marc Pollefeys, Thomas Funkhouser, et al. Openscene: 3d scene understanding with open vocabularies. In International Conference on Computer Vision and Pattern Recognition (CVPR), 2023. 2
[56] Songyou Peng, Kyle Genova, Chiyu Jiang, Andrea Tagliasacchi, Marc Pollefeys, Thomas Funkhouser 等。Openscene：使用开放词汇进行3D场景理解。发表于计算机视觉与模式识别国际会议（CVPR），2023。2


[57] Charles R Qi, Hao Su, Kaichun Mo, and Leonidas J Guibas. Pointnet: Deep learning on point sets for $3\mathrm{\;d}$ classification and segmentation. In International Conference on Computer Vision and Pattern Recognition (CVPR), 2017. 1, 2
[57] Charles R Qi, Hao Su, Kaichun Mo, 和 Leonidas J Guibas。Pointnet：用于$3\mathrm{\;d}$分类与分割的点集深度学习。发表于计算机视觉与模式识别国际会议（CVPR），2017。1, 2


[58] Charles Ruizhongtai Qi, Li Yi, Hao Su, and Leonidas J Guibas. Pointnet++: Deep hierarchical feature learning on point sets in a metric space. International Conference on Neural Information Processing Systems (NeurIPS), 2017. 2
[58] Charles Ruizhongtai Qi, Li Yi, Hao Su, 和 Leonidas J Guibas。Pointnet++：在度量空间中的点集上进行深层次的分层特征学习。发表于神经信息处理系统国际会议（NeurIPS），2017。2


[59] Minghan Qin, Wanhua Li, Jiawei Zhou, Haoqian Wang, and Hanspeter Pfister. Langsplat: 3d language gaussian splat-ting. In International Conference on Computer Vision and Pattern Recognition (CVPR), 2024. 2
[59] Minghan Qin，Wanhua Li，Jiawei Zhou，Haoqian Wang 和 Hanspeter Pfister。Langsplat：3d 语言高斯光栅。发表于计算机视觉与模式识别国际会议（CVPR），2024。2


[60] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International Conference on Machine Learning (ICML), 2021. 2, 6
[60] Alec Radford，Jong Wook Kim，Chris Hallacy，Aditya Ramesh，Gabriel Goh，Sandhini Agarwal，Girish Sastry，Amanda Askell，Pamela Mishkin，Jack Clark 等。基于自然语言监督学习可迁移的视觉模型。发表于机器学习国际会议（ICML），2021。2，6


[61] Krishan Rana, Jesse Haviland, Sourav Garg, Jad Abou-Chakra, Ian Reid, and Niko Suenderhauf. SayPlan: Grounding large language models using 3d scene graphs for scalable robot task planning. In Conference on Robot Learning (CoRL), 2023. 1
[61] Krishan Rana，Jesse Haviland，Sourav Garg，Jad Abou-Chakra，Ian Reid 和 Niko Suenderhauf。SayPlan：使用 3d 场景图来支撑大语言模型，以实现可扩展的机器人任务规划。发表于机器人学习会议（CoRL），2023。1


[62] Junha Roh, Karthik Desingh, Ali Farhadi, and Dieter Fox. Languagerefer: Spatial-language model for 3d visual grounding. In Conference on Robot Learning (CoRL), 2022. 2
[62] Junha Roh，Karthik Desingh，Ali Farhadi 和 Dieter Fox。Languagerefer：用于 3d 视觉定位的空间语言模型。发表于机器人学习会议（CoRL），2022。2


[63] Antoni Rosinol, Arjun Gupta, Marcus Abate, Jingnan Shi, and Luca Carlone. 3d dynamic scene graphs: Actionable spatial perception with places, objects, and humans. Robotics, Science and Systems, 2020. 1, 2
[63] Antoni Rosinol，Arjun Gupta，Marcus Abate，Jingnan Shi 和 Luca Carlone。3d 动态场景图：借助位置、物体和人实现可操作的空间感知。Robotics, Science and Systems，2020。1，2


[64] Antoni Rosinol, Andrew Violette, Marcus Abate, Nathan Hughes, Yun Chang, Jingnan Shi, Arjun Gupta, and Luca Carlone. Kimera: From slam to spatial perception with 3d dynamic scene graphs. International Journal on Robotics Research (IJRR), 2021. 1, 2
[64] Antoni Rosinol，Andrew Violette，Marcus Abate，Nathan Hughes，Yun Chang，Jingnan Shi，Arjun Gupta 和 Luca Carlone。Kimera：从 slam 到借助 3d 动态场景图的空间感知。国际机器人研究期刊（IJRR），2021。1，2


[65] David Rozenberszki, Or Litany, and Angela Dai. Language-grounded indoor 3d semantic segmentation in the wild. In European Conference on Computer Vision (ECCV), 2022. 2
[65] David Rozenberszki，Or Litany 和 Angela Dai。野外的语言引导室内 3d 语义分割。发表于欧洲计算机视觉会议（ECCV），2022。2


[66] Sayan Deb Sarkar, Ondrej Miksik, Marc Pollefeys, Daniel Barath, and Iro Armeni. SGAligner: 3d scene alignment with scene graphs. In International Conference on Computer Vision (ICCV), 2023. 1
[66] Sayan Deb Sarkar，Ondrej Miksik，Marc Pollefeys，Daniel Barath 和 Iro Armeni。SGAligner：带场景图的 3d 场景对齐。发表于计算机视觉国际会议（ICCV），2023。1


[67] Paul-Edouard Sarlin, Daniel DeTone, Tomasz Malisiewicz, and Andrew Rabinovich. SuperGlue: Learning feature matching with graph neural networks. In International Conference on Computer Vision and Pattern Recognition (CVPR), 2020. 6
[67] Paul-Edouard Sarlin，Daniel DeTone，Tomasz Malisiewicz 和 Andrew Rabinovich。SuperGlue：用图神经网络学习特征匹配。发表于计算机视觉与模式识别国际会议（CVPR），2020。6


[68] Johannes Lutz Schönberger and Jan-Michael Frahm. Structure-from-Motion Revisited. In International Conference on Computer Vision and Pattern Recognition (CVPR), 2016. 6
[68] Johannes Lutz Schönberger 和 Jan-Michael Frahm。重新审视结构光运动（Structure-from-Motion）。发表于计算机视觉与模式识别国际会议（CVPR），2016。6


[69] Johannes Lutz Schönberger, Enliang Zheng, Marc Polle-feys, and Jan-Michael Frahm. Pixelwise view selection for unstructured multi-view stereo. In European Conference on Computer Vision (ECCV), 2016. 6
[69] Johannes Lutz Schönberger，Enliang Zheng，Marc Polle-feys 和 Jan-Michael Frahm。用于非结构化多视图立体的逐像素视角选择。发表于欧洲计算机视觉会议（ECCV），2016。6


[70] Jonas Schult, Francis Engelmann, Alexander Hermans, Or Litany, Siyu Tang, and Bastian Leibe. Mask3d: Mask transformer for 3d semantic instance segmentation. In International Conference on Robotics and Automation (ICRA), 2023. 1,2
[70] Jonas Schult，Francis Engelmann，Alexander Hermans，Or Litany，Siyu Tang 和 Bastian Leibe。Mask3d：用于 3d 语义实例分割的 mask transformer。发表于机器人与自动化国际会议（ICRA），2023。1，2


[71] Aleksandar Shtedritski, Christian Rupprecht, and Andrea Vedaldi. What does Clip Know About a Red Circle? Visual Prompt Engineering for VLMs. In International Conference on Computer Vision (ICCV), 2023. 4
[71] Aleksandar Shtedritski，Christian Rupprecht 和 Andrea Vedaldi。Clip 知道一个红色圆圈是什么吗？面向 VLM 的视觉提示工程。发表于计算机视觉国际会议（ICCV），2023。4


[72] Tao Sun, Yan Hao, Shengyu Huang, Silvio Savarese, Konrad Schindler, Marc Pollefeys, and Iro Armeni. Nothing Stands Still: A Spatiotemporal Benchmark on 3D Point Cloud Registration Under Large Geometric and Temporal Change. ISPRS Journal of Photogrammetry and Remote Sensing, 2025. 2
[72] Tao Sun，Yan Hao，Shengyu Huang，Silvio Savarese，Konrad Schindler，Marc Pollefeys 和 Iro Armeni。一切都不静止：在大几何变化与时间变化下的 3D 点云配准时空基准。ISPRS 光学测量与遥感期刊，2025。2


[73] Ayça Takmaz, Elisabetta Fedele, Robert W Sumner, Marc Pollefeys, Federico Tombari, and Francis Engelmann. Openmask3d: Open-vocabulary 3d instance segmentation. International Conference on Neural Information Processing Systems (NeurIPS), 2023. 2, 4
[73] Ayça Takmaz，Elisabetta Fedele，Robert W Sumner，Marc Pollefeys，Federico Tombari 和 Francis Engelmann。Openmask3d：开放词汇 3d 实例分割。发表于神经信息处理系统国际会议（NeurIPS），2023。2，4


[74] Ayça Takmaz, Jonas Schult, Irem Kaftan, Mertcan Akçay, Bastian Leibe, Robert Sumner, Francis Engelmann, and Siyu Tang. 3D Segmentation of Humans in Point Clouds with Synthetic Data. In International Conference on Computer Vision (ICCV), 2023. 2
[74] Ayça Takmaz，Jonas Schult，Irem Kaftan，Mertcan Akçay，Bastian Leibe，Robert Sumner，Francis Engelmann 和 Siyu Tang。借助合成数据的点云中人体 3D 分割。发表于计算机视觉国际会议（ICCV），2023。2


[75] Ayca Takmaz, Alexandros Delitzas, Robert W. Sumner, Francis Engelmann, Johanna Wald, and Federico Tombari. Search3D: Hierarchical Open-Vocabulary 3D Segmentation. IEEE Robotics and Automation Letters (RA-L), 2025. 2
[75] Ayca Takmaz, Alexandros Delitzas, Robert W. Sumner, Francis Engelmann, Johanna Wald 和 Federico Tombari。Search3D：用于三维分割的分层开放词汇语义。IEEE Robotics and Automation Letters（RA-L），2025。2


[76] Hugues Thomas, Charles R Qi, Jean-Emmanuel Deschaud, Beatriz Marcotegui, François Goulette, and Leonidas J Guibas. Kpconv: Flexible and deformable convolution for point clouds. In International Conference on Computer Vision (ICCV), 2019. 2
[76] Hugues Thomas, Charles R Qi, Jean-Emmanuel Deschaud, Beatriz Marcotegui, François Goulette 和 Leonidas J Guibas。Kpconv：用于点云的灵活且可变形卷积。在计算机视觉国际会议（ICCV）中，2019。2


[77] Thang Vu, Kookhoi Kim, Tung M Luu, Thanh Nguyen, and Chang D Yoo. Softgroup for 3d instance segmentation on point clouds. In International Conference on Computer Vision and Pattern Recognition (CVPR), 2022. 2
[77] Thang Vu, Kookhoi Kim, Tung M Luu, Thanh Nguyen 和 Chang D Yoo。用于点云三维实例分割的 Softgroup。在计算机视觉与模式识别国际会议（CVPR）中，2022。2


[78] Johanna Wald, Helisa Dhamo, Nassir Navab, and Federico Tombari. Learning 3d semantic scene graphs from 3d indoor reconstructions. In International Conference on Computer Vision and Pattern Recognition (CVPR), 2020. 1, 2, 3
[78] Johanna Wald, Helisa Dhamo, Nassir Navab 和 Federico Tombari。从三维室内重建中学习三维语义场景图。在计算机视觉与模式识别国际会议（CVPR）中，2020。1, 2, 3


[79] Ziqin Wang, Bowen Cheng, Lichen Zhao, Dong Xu, Yang Tang, and Lu Sheng. Vl-sat: Visual-linguistic semantics assisted training for 3d semantic scene graph prediction in point cloud. In International Conference on Computer Vision and Pattern Recognition (CVPR), 2023. 2
[79] Ziqin Wang, Bowen Cheng, Lichen Zhao, Dong Xu, Yang Tang 和 Lu Sheng。Vl-sat：面向点云三维语义场景图预测的视觉-语言语义辅助训练。在计算机视觉与模式识别国际会议（CVPR）中，2023。2


[80] Silvan Weder, Francis Engelmann, Johannes L Schönberger, Akihito Seki, Marc Pollefeys, and Martin R Oswald. Alster: A Local Spatio-temporal Expert for Online 3D Semantic Reconstruction. IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), 2023. 2
[80] Silvan Weder, Francis Engelmann, Johannes L Schönberger, Akihito Seki, Marc Pollefeys 和 Martin R Oswald。Alster：面向在线三维语义重建的局部时空专家。在计算机视觉应用冬季会议（WACV，IEEE/CVF）中，2023。2


[81] Silvan Weder, Hermann Blum, Francis Engelmann, and Marc Pollefeys. Labelmaker: Automatic semantic label generation from rgb-d trajectories. In International Conference on 3d Vision (3dV), 2024. 2
[81] Silvan Weder, Hermann Blum, Francis Engelmann 和 Marc Pollefeys。Labelmaker：从 rgb-d 轨迹自动生成语义标签。在三维视觉国际会议（3dV）中，2024。2


[82] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. International Conference on Neural Information Processing Systems (NeurIPS), 2022. 4
[82] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou 等。链式思维提示会在大型语言模型中引发推理。在神经信息处理系统国际会议（NeurIPS）中，2022。4


[83] Abdelrhman Werby, Chenguang Huang, Martin Büchner, Abhinav Valada, and Wolfram Burgard. Hierarchical open-vocabulary $3\mathrm{\;d}$ scene graphs for language-grounded robot navigation. In First Workshop on Vision-Language Models for Navigation and Manipulation at ICRA 2024, 2024. 1
[83] Abdelrhman Werby, Chenguang Huang, Martin Büchner, Abhinav Valada 和 Wolfram Burgard。面向语言引导机器人导航的分层开放词汇 $3\mathrm{\;d}$ 场景图。在 ICRA 2024 会议上“导航与操作的视觉-语言模型”第一届工作坊中，2024。1


[84] Shun-Cheng Wu, Johanna Wald, Keisuke Tateno, Nassir Navab, and Federico Tombari. SceneGraphFusion: Incremental 3d scene graph prediction from rgb-d sequences. In International Conference on Computer Vision and Pattern Recognition (CVPR), 2021. 1, 2
[84] Shun-Cheng Wu, Johanna Wald, Keisuke Tateno, Nassir Navab 和 Federico Tombari。SceneGraphFusion：来自 rgb-d 序列的增量三维场景图预测。在计算机视觉与模式识别国际会议（CVPR）中，2021。1, 2


[85] Shun-Cheng Wu, Keisuke Tateno, Nassir Navab, and Federico Tombari. Incremental 3d semantic scene graph prediction from rgb sequences. In International Conference on Computer Vision and Pattern Recognition (CVPR), 2023. 2
[85] Shun-Cheng Wu, Keisuke Tateno, Nassir Navab 和 Federico Tombari。来自 rgb 序列的增量三维语义场景图预测。在计算机视觉与模式识别国际会议（CVPR）中，2023。2


[86] Chao Xu, Yixin Chen, He Wang, Song-Chun Zhu, Yixin Zhu, and Siyuan Huang. Partafford: Part-level affordance discovery from 3d objects. European Conference on Computer Vision (ECCV) Workshops, 2022. 2
[86] Chao Xu, Yixin Chen, He Wang, Song-Chun Zhu, Yixin Zhu 和 Siyuan Huang。Partafford：从三维物体中发现部件级可操作性。在欧洲计算机视觉会议（ECCV）工作坊中，2022。2


[87] Danfei Xu, Yuke Zhu, Christopher B Choy, and Li Fei-Fei. Scene graph generation by iterative message passing. In International Conference on Computer Vision and Pattern Recognition (CVPR), 2017. 6
[87] Danfei Xu, Yuke Zhu, Christopher B Choy 和 Li Fei-Fei。通过迭代消息传递生成场景图。在计算机视觉与模式识别国际会议（CVPR）中，2017。6


[88] Jianwei Yang, Jiasen Lu, Stefan Lee, Dhruv Batra, and Devi Parikh. Graph r-cnn for scene graph generation. In European Conference on Computer Vision (ECCV), 2018.
[88] Jianwei Yang, Jiasen Lu, Stefan Lee, Dhruv Batra 和 Devi Parikh。用于场景图生成的图 r-cnn。在欧洲计算机视觉会议（ECCV）中，2018。


[89] Jingkang Yang, Yi Zhe Ang, Zujin Guo, Kaiyang Zhou, Wayne Zhang, and Ziwei Liu. Panoptic scene graph generation. In European Conference on Computer Vision (ECCV), 2022. 6
[89] Jingkang Yang, Yi Zhe Ang, Zujin Guo, Kaiyang Zhou, Wayne Zhang 和 Ziwei Liu。全景场景图生成。在欧洲计算机视觉会议（ECCV）中，2022。6


[90] Zhengyuan Yang, Songyang Zhang, Liwei Wang, and Jiebo Luo. Sat: 2d semantics assisted training for 3d visual grounding. In International Conference on Computer Vision (ICCV), 2021. 2
[90] Zhengyuan Yang, Songyang Zhang, Liwei Wang 和 Jiebo Luo。Sat：用于三维视觉定位的二维语义辅助训练。在计算机视觉国际会议（ICCV）中，2021。2


[91] Yufei Ye, Abhinav Gupta, and Shubham Tulsiani. What's in your hands? 3d reconstruction of generic objects in hands. In International Conference on Computer Vision and Pattern Recognition (CVPR), 2022. 2
[91] Yufei Ye, Abhinav Gupta, and Shubham Tulsiani. 你的手里有什么？手中通用物体的3D重建。发表于国际计算机视觉与模式识别会议（CVPR），2022。2


[92] Yufei Ye, Abhinav Gupta, Kris Kitani, and Shubham Tul-siani. G-hop: Generative hand-object prior for interaction reconstruction and grasp synthesis. In International Conference on Computer Vision and Pattern Recognition (CVPR), 2024. 2
[92] Yufei Ye, Abhinav Gupta, Kris Kitani, and Shubham Tul-siani. G-hop：用于交互重建和抓取合成的生成式手-物先验。发表于国际计算机视觉与模式识别会议（CVPR），2024。2


[93] Chandan Yeshwanth, Yueh-Cheng Liu, Matthias Nießner, and Angela Dai. Scannet++: A high-fidelity dataset of 3d indoor scenes. In International Conference on Computer Vision (ICCV), 2023. 2, 5, 6
[93] Chandan Yeshwanth, Yueh-Cheng Liu, Matthias Nießner, and Angela Dai. Scannet++：高保真3D室内场景数据集。发表于国际计算机视觉会议（ICCV），2023。2, 5, 6


[94] Gonca Yilmaz, Songyou Peng, Marc Pollefeys, Francis Engelmann, and Hermann Blum. OpenDAS: Open-Vocabulary Domain Adaptation for 2D and 3D Segmentation. arXiv preprint arXiv:2405.20141, 2024. 2
[94] Gonca Yilmaz, Songyou Peng, Marc Pollefeys, Francis Engelmann, and Hermann Blum. OpenDAS：2D和3D分割的开放词汇领域自适应。arXiv预印本 arXiv:2405.20141，2024。2


[95] Tomoya Yoshida, Shuhei Kurita, Taichi Nishimura, and Shinsuke Mori. Text-driven affordance learning from egocentric vision. arXiv preprint arXiv:2404.02523, 2024. 2
[95] Tomoya Yoshida, Shuhei Kurita, Taichi Nishimura, and Shinsuke Mori. 基于自我中心视觉的文本驱动可供性学习。arXiv预印本 arXiv:2404.02523，2024。2


[96] Yuanwen Yue, Sabarinath Mahadevan, Jonas Schult, Francis Engelmann, Bastian Leibe, Konrad Schindler, and Theodora Kontogianni. Agile3d: Attention guided interactive multi-object 3d segmentation. International Conference on Learning Representations (ICLR), 2024. 2
[96] 袁文·岳，沙巴里纳斯·马哈德万，乔纳斯·舒尔特，弗朗西斯·恩格尔曼，巴斯蒂安·莱贝，康拉德·施林德勒，以及西奥多拉·孔托吉安尼。Agile3d：注意力引导的交互式多物体三维分割。国际学习表征会议（ICLR），2024。2


[97] Guangyao Zhai, Evin Pinar Örnek, Shun-Cheng Wu, Yan Di, Federico Tombari, Nassir Navab, and Benjamin Busam. Commonscenes: Generating commonsense 3d indoor scenes with scene graphs. International Conference on Neural Information Processing Systems (NeurIPS), 2023. 1,2
[97] Guangyao Zhai，Evin Pinar Örnek，沈成·吴，Yan Di，Federico Tombari，Nassir Navab，以及 Benjamin Busam。Commonscenes：借助场景图生成具有常识的三维室内场景。神经信息处理系统国际会议（NeurIPS），2023。1,2


[98] Wei Zhai, Hongchen Luo, Jing Zhang, Yang Cao, and Dacheng Tao. One-shot object affordance detection in the wild. International Journal on Computer Vision (IJCV), 2022. 2
[98] Wei Zhai，Hongchen Luo，Jing Zhang，Yang Cao，以及 Dacheng Tao。野外的一次性物体可用性检测。国际计算机视觉期刊（IJCV），2022。2


[99] Chaoyi Zhang, Jianhui Yu, Yang Song, and Weidong Cai. Exploiting edge-oriented reasoning for 3d point-based scene graph analysis. In International Conference on Computer Vision and Pattern Recognition (CVPR), 2021. 2
[99] Chaoyi Zhang，Jianhui Yu，Yang Song，以及 Weidong Cai。利用面向边缘的推理来分析基于三维点的场景图。国际计算机视觉与模式识别会议（CVPR），2021。2


[100] Chenyangguang Zhang, Yan Di, Ruida Zhang, Guangyao Zhai, Fabian Manhardt, Federico Tombari, and Xiangyang Ji. Ddf-ho: Hand-held object reconstruction via conditional directed distance field. International Conference on Neural Information Processing Systems (NeurIPS), 2023. 2
[100] Chenyangguang Zhang，Yan Di，Ruida Zhang，Guangyao Zhai，Fabian Manhardt，Federico Tombari，以及 Xiangyang Ji。Ddf-ho：基于条件定向距离场的手持物体重建。神经信息处理系统国际会议（NeurIPS），2023。2


[101] Chenyangguang Zhang, Guanlong Jiao, Yan Di, Gu Wang, Ziqin Huang, Ruida Zhang, Fabian Manhardt, Bowen Fu, Federico Tombari, and Xiangyang Ji. Moho: Learning single-view hand-held object reconstruction with multiview occlusion-aware supervision. In International Conference on Computer Vision and Pattern Recognition (CVPR), 2024. 2
[101] Chenyangguang Zhang，Guanlong Jiao，Yan Di，Gu Wang，Ziqin Huang，Ruida Zhang，Fabian Manhardt，Bowen Fu，Federico Tombari，以及 Xiangyang Ji。Moho：通过多视角遮挡感知监督学习单视角手持物体重建。国际计算机视觉与模式识别会议（CVPR），2024。2


[102] Shoulong Zhang, Aimin Hao, Hong Qin, et al. Knowledge-inspired 3d scene graph prediction in point cloud. International Conference on Neural Information Processing Systems (NeurIPS), 2021. 2
[102] Shoulong Zhang，Aimin Hao，Hong Qin 等。基于知识启发的点云三维场景图预测。神经信息处理系统国际会议（NeurIPS），2021。2


[103] Yiming Zhang, ZeMing Gong, and Angel X Chang. Multi3drefer: Grounding text description to multiple 3d objects. In International Conference on Computer Vision (ICCV), 2023. 2
[103] Yiming Zhang，ZeMing Gong，以及 Angel X Chang。Multi3drefer：将文本描述落到多个三维物体上。国际计算机视觉会议（ICCV），2023。2


[104] Youcai Zhang, Xinyu Huang, Jinyu Ma, Zhaoyang Li, Zhaochuan Luo, Yanchun Xie, Yuzhuo Qin, Tong Luo, Yaqian Li, Shilong Liu, et al. Recognize anything: A strong image tagging model. In International Conference on Computer Vision and Pattern Recognition (CVPR), 2024. 3, 4
[104] Youcai Zhang，Xinyu Huang，Jinyu Ma，Zhaoyang Li，Zhaochuan Luo，Yanchun Xie，Yuzhuo Qin，Tong Luo，Yaqian Li，Shilong Liu 等。识别任意事物：一种强大的图像标注模型。国际计算机视觉与模式识别会议（CVPR），2024。3,4


[105] Shijie Zhou, Haoran Chang, Sicheng Jiang, Zhiwen Fan, Zehao Zhu, Dejia Xu, Pradyumna Chari, Suya You, Zhangyang Wang, and Achuta Kadambi. Feature 3dgs: Supercharging 3d gaussian splatting to enable distilled feature fields. In International Conference on Computer Vision and Pattern Recognition (CVPR), 2024. 2
[105] Shijie Zhou，Haoran Chang，Sicheng Jiang，Zhiwen Fan，Zehao Zhu，Dejia Xu，Pradyumna Chari，Suya You，Zhangyang Wang，以及 Achuta Kadambi。Feature 3dgs：用来提升三维高斯溅射能力，从而实现蒸馏特征场。国际计算机视觉与模式识别会议（CVPR），2024。2


[106] Zijian Zhou, Zheng Zhu, Holger Caesar, and Miaojing Shi. Openpsg: Open-set panoptic scene graph generation via large multimodal models. European Conference on Computer Vision (ECCV), 2024. 6
[106] Zijian Zhou，Zheng Zhu，Holger Caesar，以及 Miaojing Shi。Openpsg：借助大规模多模态模型实现开放集全景场景图生成。欧洲计算机视觉会议（ECCV），2024。6


[107] Xingxing Zuo, Pouya Samangouei, Yunwen Zhou, Yan Di, and Mingyang Li. Fmgs: Foundation model embedded 3d gaussian splatting for holistic $3\mathrm{\;d}$ scene understanding. International Journal on Computer Vision (IJCV), 2024. 2
[107] Xingxing Zuo，Pouya Samangouei，Yunwen Zhou，Yan Di，以及 Mingyang Li。Fmgs：用于整体 $3\mathrm{\;d}$ 场景理解的基于基础模型嵌入的三维高斯溅射。国际计算机视觉期刊（IJCV），2024。2


[108] René Zurbrügg, Yifan Liu, Francis Engelmann, Suryansh Kumar, Marco Hutter, Vaishakh Patil, and Fisher Yu. ICGNet: A Unified Approach for Instance-centric Grasping. In International Conference on Robotics and Automation (ICRA), 2024. 8
[108] René Zurbrügg，Yifan Liu，Francis Engelmann，Suryansh Kumar，Marco Hutter，Vaishakh Patil，以及 Fisher Yu。ICGNet：一种面向以实例为中心的抓取的统一方法。机器人与自动化国际会议（ICRA），2024。8