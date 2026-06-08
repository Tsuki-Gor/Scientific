# Hierarchical Open-Vocabulary 3D Scene Graphs for Language-Grounded Robot Navigation
# 面向语言引导机器人导航的分层开放词汇三维场景图


Abdelrhman Werby ${}^{1 * }$ , Chenguang Huang ${}^{1 * }$ , Martin Büchner ${}^{1 * }$ , Abhinav Valada ${}^{1}$ , and Wolfram Burgard ${}^{2}$
Abdelrhman Werby ${}^{1 * }$ ，Chenguang Huang ${}^{1 * }$ ，Martin Büchner ${}^{1 * }$ ，Abhinav Valada ${}^{1}$ ，以及 Wolfram Burgard ${}^{2}$


Abstract- Typically, robotic mapping relies on highly accurate dense representations obtained via approaches to simultaneous localization and mapping. While these maps allow for point/voxel-level features, they do not provide language grounding within large-scale environments due to the sheer number of points. In this work, we present HOV-SG, a hierarchical open-vocabulary 3D scene graph mapping approach for robot navigation. Using open-vocabulary vision foundation models, we first obtain state-of-the-art open-vocabulary maps in 3D. We then perform floor as well as room segmentation and identify room names. Finally, we construct a 3D scene graph hierarchy. Our approach is able to represent multi-story buildings and allows robots to traverse them by providing feasible links among floors. We demonstrate long-horizon robotic navigation in large-scale indoor environments from long queries using large language models based on the obtained scene graph tokens and outperform previous baselines.
摘要——通常，机器人建图依赖通过同时定位与建图方法获得的高精度稠密表示。尽管这些地图支持点/体素级特征，但由于点的数量巨大，它们无法在大规模环境中实现语言落地。本文提出 HOV-SG，一种用于机器人导航的分层开放词汇三维场景图建图方法。我们利用开放词汇视觉基础模型，首先在三维中获得最先进的开放词汇地图。随后进行地面与房间分割，并识别房间名称。最后构建三维场景图层级。该方法能够表示多层建筑，并通过楼层之间可行的连接关系使机器人可在其中穿行。我们在大规模室内环境中展示了基于大型语言模型、从长查询出发的长时域机器人导航，并利用获得的场景图标记超过了先前的基线方法。


## I. INTRODUCTION
## I. INTRODUCTION


Humans acquire conceptual knowledge through multimodal experiences. These experiences are paramount to object recognition and language as well as reasoning and planning [1], [2]. Cognitive maps store this information based on sensor fusion, fragmentation, and hierarchical structure. This is central to the human ability to navigate the physical world [3]-[5]. Recently, language proved to be an effective link between intelligent systems and humans and can enable robot autonomy in complex human-centered environments [6]-[9].
人类通过多模态经验获得概念知识。这些经验对目标识别与语言，以及推理与规划至关重要[1]，[2]。认知地图通过传感器融合、碎片化与层次结构来存储这些信息。这是人类在物理世界中导航能力的核心[3]-[5]。最近，语言被证明是智能系统与人类之间的一种有效纽带，并能在以人为中心的复杂环境中实现机器人自主性[6]-[9]。


Classical methods for robot navigation build dense spatial maps of high accuracy using approaches to simultaneous localization and mapping (SLAM) [10]-[12]. Those give rise to fine-grained navigation and manipulation based on geometric goal specifications. Recent advances have combined dense maps with pre-trained zero-shot vision-language models, which facilitates open-vocabulary indexing of observed environments [7], [13]-[18]. While these approaches marry the area of classical robotics with modern open-vocabulary semantics, representing larger scenes while abstracting still poses a considerable hurdle. A number of works approach this using 3D scene graph structures [19]-[21] that excel at representing larger environments efficiently. At the same time, they constitute a useful interface to semantic tokens used for prompting large language models (LLM). Nonetheless, most approaches rely on closed-set semantics with the exception of ConceptGraphs [22] that focuses on smaller scenes.
传统的机器人导航方法通过同时定位与建图（SLAM）[10]-[12]来构建高精度的稠密空间地图。它们基于几何目标表述，从而实现精细的导航与操作。近期进展将稠密地图与预训练的零样本视觉-语言模型结合，从而便于对已观测环境进行开放词汇索引[7]，[13]-[18]。尽管这些方法将经典机器人领域与现代开放词汇语义结合，但在抽象的同时表示更大的场景仍是一个相当大的挑战。一些工作通过3D场景图结构来应对这一问题[19]-[21]，它们能够高效表示更大的环境。与此同时，它们也是用于提示大语言模型（LLM）的语义标记的一个有用接口。然而，大多数方法依赖封闭集合语义，除ConceptGraphs[22]外，它关注的是较小的场景。


<img src="https://cdn.noedgeai.com/bo_d8j3o4491nqc738ucjg0_0.jpg?x=915&y=454&w=733&h=569&r=0"/>



Fig. 1. HOV-SG allows the construction of accurate, open-vocabulary 3D scene graphs for large-scale and multi-story environments and enables robots to effectively navigate in them.
图1. HOV-SG能够为大规模、多楼层环境构建精确的开放词汇3D场景图，并使机器人能够在其中有效导航。


With HOV-SG, we demonstrate the construction of hierarchical 3D scene graphs with open-vocabulary vision-language features [23], [24] across multi-story environments. By abstracting from dense maps and indexing floors, rooms as well as objects, our actionable 3D scene graph hierarchies are promptable using LLMs. This enables object retrieval as well as long-horizon robotic navigation in large-scale indoor environments from long queries as shown in Fig. 1. By doing so, we present state-of-the-art results in open-set 3D semantic segmentation, object retrieval from long queries as well as room identification from perception inputs.
通过HOV-SG，我们展示了在多楼层环境中，如何利用开放词汇的视觉-语言特征[23]，[24]构建分层3D场景图。我们通过从稠密地图中抽象并索引楼层，将“房间”以及“物体”纳入层次结构，因此这种可操作的3D场景图层级可以由LLM进行提示。由此，机器人能够在大规模室内环境中实现物体检索以及从长查询出发的长时域导航，如图1所示。通过上述方式，我们在开放式3D语义分割、从长查询检索物体以及从感知输入识别房间方面取得了最新的成果。


In summary, we make the following main contributions:
总之，我们做出以下主要贡献：


- We introduce a pipeline for constructing hierarchical, open-vocabulary 3D scene graphs from a multi-story environment that enables LLM-based prompting, planning as well as robotic navigation.
- 我们提出了一种流程，用于从多楼层环境构建分层的开放词汇3D场景图，使得能够进行基于LLM的提示、规划以及机器人导航。


- We extensively evaluate our approach across three diverse datasets as well as a real-world environment. We achieve state-of-the-art performance in 3D open-set semantic segmentation and demonstrate successful robotic navigation from abstract language queries.
- 我们在三个多样化数据集以及一个真实世界环境上对该方法进行了广泛评估。在3D开放式语义分割上达到了最先进性能，并证明了机器人能够从抽象语言查询中成功实现导航。


- We introduce a novel evaluation metric for measuring open-vocabulary semantics termed ${\mathrm{{AUC}}}_{\text{ top-k }}$ and publish code at http://hovsg.github.io.
- 我们提出了一种用于度量开放词汇语义的新评估指标，称为${\mathrm{{AUC}}}_{\text{ top-k }}$，并在http://hovsg.github.io发布代码。


---



* Equal contribution.
* 贡献相同。


1 Department of Computer Science, University of Freiburg, Germany.
1 德国弗赖堡大学计算机科学系。


2 Department of Eng., University of Technology Nuremberg, Germany. Supplementary material can be found at http://hovsg.github.io This work was funded by the German Research Foundation (DFG) Emmy Noether Program grant number 468878300, the BrainLinks-BrainTools Center of the University of Freiburg, and an academic grant from NVIDIA.
2 德国纽伦堡应用工科大学工程学院。补充材料可在http://hovsg.github.io找到。本研究由德国研究基金会（DFG）Emmy Noether计划（资助编号468878300）、弗赖堡大学的BrainLinks-BrainTools中心，以及来自NVIDIA的学术资助支持。


---



<img src="https://cdn.noedgeai.com/bo_d8j3o4491nqc738ucjg0_1.jpg?x=155&y=141&w=1491&h=366&r=0"/>



Fig. 2. HOV-SG builds hierarchical open-vocabulary 3D scene graphs of indoor household scenes. We first use SAM to extract object masks per frame while obtaining vision-language features via CLIP. In the next step, we aggregate these features on a point level in the map. Secondly, we segment the full point cloud based on merged 3D masks. To produce more meaningful semantic object features, we employ a DBSCAN-based filtering approach to obtain a majority vote feature for each object. To construct an actionable 3D scene graph, we segment the obtained panoptic map into multiple floors, segment and classify distinct regions using several view embeddings, and identify object names via querying. As a result, HOV-SG allows hierarchical querying and navigation using mobile robots even in complex multi-floor environments.
图2. HOV-SG构建室内家庭场景的分层开放词汇3D场景图。首先，我们使用SAM为每一帧提取物体掩码，同时通过CLIP获取视觉-语言特征。下一步，我们在地图的点级别对这些特征进行聚合。其次，我们基于融合后的3D掩码对完整点云进行分割。为了得到更有意义的语义物体特征，我们采用基于DBSCAN的过滤方法，为每个物体获取“多数投票”的特征。为了构建可操作的3D场景图，我们将得到的全景图（panoptic map）分割为多个楼层，并使用多个视角嵌入对不同区域进行分割与分类，同时通过查询来识别物体名称。结果表明，即使在复杂的多楼层环境中，HOV-SG也能让移动机器人进行分层查询与导航。


## II. TECHNICAL APPROACH
## II. 技术方法


This work aims to develop a concise and efficient visual-language graph representation for large-scale multi-floor indoor environments given RGB-D observations and odometry. The graph should facilitate the indexing of semantic concepts through natural language queries and enable robotic navigation in multi-floor environments. We address this by proposing Hierarchical Open-Vocabulary Scene Graphs, in short HOV-SG. In the following, we describe (i) the construction of a 3D segment-level open-vocabulary map (Sec. II-A), (ii) the generation of the hierarchical open-vocabulary scene graphs (Sec. II-B), and (iii) language-conditioned navigation across large-scale environments (Sec. II-C). Fig. 2 presents an overview of our method.
本工作旨在基于 RGB-D 观测和里程计，开发一种适用于大型多楼层室内环境的简洁高效视觉-语言图表示。该图应便于通过自然语言查询对语义概念进行索引，并支持机器人在多楼层环境中的导航。我们通过提出分层开放词汇场景图（Hierarchical Open-Vocabulary Scene Graphs），简称 HOV-SG 来实现。下文将依次介绍（i）3D 分段级开放词汇地图的构建（Sec. II-A）、（ii）分层开放词汇场景图的生成（Sec. II-B）以及（iii）在大型环境中的语言条件导航（Sec. II-C）。图 2 给出了我们方法的概览。


### A.3D Segment-Level Open-Vocabulary Mapping
### A. 3D分段级开放词汇映射


Frame-Wise 3D Segment Merging: Given a sequence of RGB-D observations, we utilize Segment Anything [24] to obtain a list of class-agnostic 2D binary masks at each timestep. The pixels in each mask are then backprojected into 3D using the depth information, resulting in a list of point clouds, or 3D segments. Based on accurate odometry estimates, we transform all 3D segments into the global coordinate frame. These frame-wise segments are either initialized as new global segments or merged with existing ones based on an overlap metric detailed in Sec. S.1-A.
逐帧3D分段合并：给定一系列RGB-D观测，我们利用Segment Anything [24]在每个时间步获取一组类别无关的2D二值掩码。随后，利用深度信息将每个掩码中的像素反投影到3D中，得到一组点云，即3D分段。基于准确的里程计估计，我们将所有3D分段变换到全局坐标系中。这些逐帧分段要么初始化为新的全局分段，要么根据S.1-A节中详述的重叠度量与已有分段合并。


Open-Vocabulary Segment Features: For each obtained 2D SAM mask per frame, we obtain an image crop based on its bounding box as well as an image of the isolated mask without background. We encode the RGB observation and the two mask-wise images with CLIP [23] and fuse them in a weighted-sum manner. Assuming constant CLIP features across each mask, we transform the 2D mask into global 3D coordinates and associate the obtained fused CLIP feature with the nearest 3D points in a pre-computed reference point cloud. Based on this association, we register the obtained segment features on a global point-wise feature map. The final point-wise features are then determined by averaging each reference point's associated features. Based on the 3D segments obtained in the independent merging step, we can finally infer open-vocabulary vision-language features for all 3D segments as outlined in Fig.1. In the subsequent step, we match point-wise features with previously computed segments. For each point within a segment, we identify the nearest points in the reference point cloud and collect their CLIP features. To mitigate potential high variance, instead of directly averaging these features, we employ DBSCAN clustering to handle cases like under-segmentation or imperfect views, ensuring a more representative feature by selecting the feature closest to the mean of the majority cluster.
开放词汇分段特征：对于每帧获得的每个2D SAM掩码，我们根据其边界框裁剪出图像，并生成一张去除背景的独立掩码图像。我们使用CLIP [23]对RGB观测以及这两种掩码图像进行编码，并以加权求和的方式融合。假设每个掩码内的CLIP特征保持不变，我们将2D掩码变换到全局3D坐标中，并将得到的融合CLIP特征关联到预计算参考点云中最近的3D点。基于这一关联，我们将得到的分段特征注册到全局逐点特征图上。最终的逐点特征通过对每个参考点关联的特征取平均得到。基于独立合并步骤得到的3D分段，我们最终可以如图1所示，为所有3D分段推断开放词汇的视觉-语言特征。在后续步骤中，我们将逐点特征与先前计算的分段进行匹配。对于分段中的每个点，我们识别参考点云中的最近点并收集其CLIP特征。为缓解潜在的高方差，我们不直接对这些特征求平均，而是采用DBSCAN聚类来处理欠分割或视角不完美等情况，通过选择最接近多数簇均值的特征，确保得到更具代表性的特征。


<img src="https://cdn.noedgeai.com/bo_d8j3o4491nqc738ucjg0_1.jpg?x=952&y=705&w=663&h=254&r=0"/>



Fig. 3. By associating 10 camera views, i.e., its CLIP embeddings to each room we obtain a set of 10 open-vocabulary embeddings per segmented room. This serves as the attributed room feature in the scene graph.
图3. 通过关联10个摄像机视角，即每个房间对应的CLIP嵌入，我们为每个分割后的房间获得一组10个开放词汇嵌入。这作为场景图中的属性化房间特征。


### B.3D Scene Graph Construction
### B.3D 场景图构建


In this section, we describe how to build the hierarchical open-vocabulary scene graph given a global reference point cloud of the scene, a list of global 3D segments, and their associated CLIP features as described in Sec. II-A. The HOV-SG representation comprises a root node, multiple floors, and room nodes as well as object nodes. The edges among nodes represent the hierarchy of the obtained graph. For more details, we refer to Sec. S.1-B.
在本节中，我们将介绍如何基于场景的全局参考点云、全局 3D 分段列表以及它们在 Sec. II-A 中所述的对应 CLIP 特征，构建分层开放词汇场景图。HOV-SG 表示由一个根节点、多层楼层、房间节点以及对象节点组成。节点之间的边表示所获得图的层级关系。更多细节见 Sec. S.1-B。


Floor Segmentation: In order to separate floors, we identify peaks within a height histogram over all points contained in the point cloud. We apply adaptive thresholding and DBSCAN clustering to obtain potential floors and ceilings. We select the top-2 levels in each cluster. Taken pairwise, these represent individual floors (floor and ceiling) in the building as in Fig. S.1. We equip each floor node with a CLIP text embedding using the template "floor \{#\}". An edge between the root node and the respective floor node is established.
楼层分割：为区分不同楼层，我们在点云中所有点的高度直方图上识别峰值。我们采用自适应阈值与 DBSCAN 聚类来得到潜在的楼层与天花板。每个聚类中选择最高的 2 个层级。两两配对后，它们对应建筑中的单个楼层（楼层与天花板），如图 S.1 所示。我们使用模板“floor \{#\}”为每个楼层节点配备一个 CLIP 文本嵌入。并在根节点与对应的楼层节点之间建立一条边。


Room Segmentation: Based on each obtained floor point cloud, we construct a 2D bird's-eye-view (BEV) histogram as outlined in Fig. S.1. Next, we obtain walls and apply the Watershed algorithm to obtain a list of region masks. We extract the 3D points that fall into the floor's height interval as well as the BEV room segment to form room point clouds that are used to associate objects to rooms later. Each room constitutes a node and is connected to its corresponding floor.
房间分割：在获得每个楼层点云后，我们按图 S.1 的方法构建 2D 顶视图（BEV）直方图。接着，我们提取墙体，并使用 Watershed 算法得到一组区域掩膜。我们提取落在楼层高度区间内的 3D 点，以及 BEV 房间分段，以形成房间点云，用于后续将对象关联到房间。每个房间构成一个节点，并与其对应的楼层相连。


In order to attribute room nodes, we associate RGB-D observations whose camera poses reside within a BEV room segment to those rooms, see Fig. 3. The CLIP embeddings of these images are filtered by extracting $k$ representative view embeddings using the k-means algorithm. During the query, we compute the cosine similarity between the CLIP text embeddings of a room categories list and each representative feature, resulting in a similarity matrix. With the argmax operation, we obtain opinions from all representatives, allowing retrieval of the room type voted by the majority. These K representative embeddings and the room point cloud are jointly stored in the root node in the graph. An edge between the floor node and each room node and its parent floor node is established. The construction and querying of room features are illustrated in Fig. 3.
为确定房间节点，我们将相机位姿位于某个 BEV 房间分段内的 RGB-D 观测关联到对应房间，见图 3。通过使用 k-means 算法从这些图像中提取 $k$ 个代表视角嵌入来筛选 CLIP 嵌入。在查询阶段，我们计算房间类别列表的 CLIP 文本嵌入与每个代表特征之间的余弦相似度，从而得到相似度矩阵。借助 argmax 操作，我们从所有代表中获得投票，使得多数投票确定房间类型。随后，这 K 个代表嵌入以及房间点云将被共同存储在图的根节点中。在每个楼层节点与其各自的房间节点及父楼层节点之间建立边。房间特征的构建与查询如图 3 所示。


Object Identification: We associate global object segments to rooms in the bird's-eye-view based on point cloud overlaps, which is further described in Appendix S.1-C. To reduce the number of nodes, we merge 3D segments of partial overlap that produce equal object labels when queried against a chosen label set. Finally, each obtained 3D segment translates to an object node, and an edge is established between the object and its parent room node.
对象识别：我们在鸟瞰图（BEV）中基于点云重叠度，将全局对象分段关联到房间，详见附录 S.1-C。为减少节点数量，我们将查询到所选标签集时会产生相同对象标签的部分重叠的 3D 分段进行合并。最终，每个得到的 3D 分段对应一个对象节点，并在对象节点与其父房间节点之间建立一条边。


Actionable Navigational Graph: In addition to the open-vocabulary hierarchy, the scene graph also contains a navigational Voronoi graph that serves robotic traversability of the mapped surroundings [25] spanning multiple floors. This enables high-level planning and low-level execution based on the Voronoi graph. The details of the navigation graph creation are provided in Sec. S.1-D.
可执行导航图：除了开放词汇层级结构，场景图还包含一个用于机器人通行性的导航 Voronoi 图，用于跨多个楼层的已映射环境 [25]。这使得能够基于 Voronoi 图进行高层规划与低层执行。导航图的创建细节见 Sec. S.1-D。


## C. Object Navigation from Long Queries
## C. 从长查询进行目标导航


HOV-SG extends the scope of potential navigation goals to more specific spatial concepts like regions and floors compared to simple object goals [7], [15], [16], [22]. Language-guided navigation with HOV-SG involves processing complex queries such as find the toilet in the bathroom on floor 2 using a large language model (GPT-3.5). We break down such lengthy instructions into three separate queries: one for the floor level, one for the room level, and one for the object level. Leveraging the explicit hierarchical structure of HOV-SG, we sequentially query against each hierarchy to progressively narrow down the solution space. Once a target node is identified, we utilize the navigational graph mentioned above to plan a path from the starting pose to the target destination, which is demonstrated in Fig. S.4.
与仅关注简单目标的做法相比，HOV-SG 将可能的导航目标范围扩展到更具体的空间概念，如区域和楼层[7]，[15]，[16]，[22]。借助 HOV-SG 的语言引导导航，需要处理诸如使用大语言模型（GPT-3.5）查找“二楼浴室里的马桶”之类的复杂查询。我们将这类冗长指令拆分为三个独立的查询：一个用于楼层层级，一个用于房间层级，一个用于目标层级。利用 HOV-SG 的明确层级结构，我们依次在各个层级上进行查询，从而逐步缩小解空间。确定目标节点后，我们使用上文提到的导航图，从起始位姿规划到目标目的地的路径，如图 S.4 所示。


## III. EXPERIMENTAL EVALUATION
## III. 实验评估


In the following, we first evaluate HOV-SG against recent open-vocabulary map representations on the task of 3D semantic segmentation. Secondly, we investigate the accuracy of the constructed hierarchical, open-vocabulary 3D scene graphs from scenes of the Habitat Matterport 3D Semantics Dataset [26]. Finally, we study open-vocabulary object retrieval and demonstrate large-scale language-grounded robotic navigation in the real world.
下面，我们首先在3D语义分割任务上，将HOV-SG与近期的开放词汇地图表示进行评估。其次，我们考察基于Habitat Matterport 3D Semantics Dataset [26]场景构建的分层开放词汇3D场景图的准确性。最后，我们研究开放词汇目标检索，并展示现实世界中的大规模语言引导机器人导航。


<img src="https://cdn.noedgeai.com/bo_d8j3o4491nqc738ucjg0_2.jpg?x=916&y=130&w=732&h=297&r=0"/>



Fig. 4. Qualitative results for 3D semantic segmentation on Replica
图4. Replica上3D语义分割的定性结果


TABLE I
表I


Open-Vocabulary 3D SEMANTIC SEGMENTATION
开放词汇3D语义分割


<table><tr><td rowspan="2">Method</td><td rowspan="2">CLIP Backbone</td><td colspan="3">Replica</td><td colspan="3">ScanNet</td></tr><tr><td>mIOU</td><td>F-mIOU</td><td>mAcc</td><td>mIOU</td><td>F-mIOU</td><td>mAcc</td></tr><tr><td rowspan="2">ConceptFusion [16]</td><td>OVSeg</td><td>0.10</td><td>0.21</td><td>0.16</td><td>0.08</td><td>0.11</td><td>0.15</td></tr><tr><td>Vit-H-14</td><td>0.10</td><td>0.18</td><td>0.17</td><td>0.11</td><td>0.12</td><td>0.21</td></tr><tr><td rowspan="2">ConceptGraph [22]</td><td>OVSeg</td><td>0.13</td><td>0.27</td><td>0.21</td><td>0.15</td><td>0.18</td><td>0.23</td></tr><tr><td>Vit-H-14</td><td>0.18</td><td>0.23</td><td>0.30</td><td>0.16</td><td>0.20</td><td>0.28</td></tr><tr><td rowspan="2">HOV-SG (ours)</td><td>OVseg</td><td>0.144</td><td>0.255</td><td>0.212</td><td>0.214</td><td>0.258</td><td>0.420</td></tr><tr><td>Vit-H-14</td><td>0.231</td><td>0.386</td><td>0.304</td><td>0.222</td><td>0.303</td><td>0.431</td></tr></table>
<table><tbody><tr><td rowspan="2">方法</td><td rowspan="2">CLIP骨干网络</td><td colspan="3">Replica</td><td colspan="3">ScanNet</td></tr><tr><td>mIOU</td><td>F-mIOU</td><td>mAcc</td><td>mIOU</td><td>F-mIOU</td><td>mAcc</td></tr><tr><td rowspan="2">ConceptFusion [16]</td><td>OVSeg</td><td>0.10</td><td>0.21</td><td>0.16</td><td>0.08</td><td>0.11</td><td>0.15</td></tr><tr><td>Vit-H-14</td><td>0.10</td><td>0.18</td><td>0.17</td><td>0.11</td><td>0.12</td><td>0.21</td></tr><tr><td rowspan="2">ConceptGraph [22]</td><td>OVSeg</td><td>0.13</td><td>0.27</td><td>0.21</td><td>0.15</td><td>0.18</td><td>0.23</td></tr><tr><td>Vit-H-14</td><td>0.18</td><td>0.23</td><td>0.30</td><td>0.16</td><td>0.20</td><td>0.28</td></tr><tr><td rowspan="2">HOV-SG（我们的）</td><td>OVseg</td><td>0.144</td><td>0.255</td><td>0.212</td><td>0.214</td><td>0.258</td><td>0.420</td></tr><tr><td>Vit-H-14</td><td>0.231</td><td>0.386</td><td>0.304</td><td>0.222</td><td>0.303</td><td>0.431</td></tr></tbody></table>


Higher values are better. The ConceptFusion pipeline evaluated against made use of instance masks predicted by SAM [24]. We consider ViT-H-14 and a fine-tuned backbone ViT-L-14 released with the work OVSeg [29].
数值越高越好。所评估的 ConceptFusion 流水线使用了由 SAM [24] 预测的实例掩码。我们考虑了 ViT-H-14 以及随 OVSeg [29] 工作发布的、经过微调的骨干网络 ViT-L-14。


### A.3D Semantic Segmentation on ScanNet and Replica
### A.ScanNet和Replica上的3D语义分割


We evaluate the open-vocabulary 3D semantic segmentation performance on the ScanNet [27] and Replica [28] datasets. We compare our method with two competitive vision-and-language representations, namely ConceptFusion [16] and ConceptGraphs [22], and ablate over two CLIP backbones, see Table I. In terms of mIOU and F-mIOU, HOV-SG outperforms the open-vocabulary baselines by a large margin. This is primarily due to the following improvements we made: First, when we merge segment features, we consider all pointwise features that each segment covers and use DBSCAN to obtain the dominant feature, which increases the robustness compared to taking the mean as done by ConceptGraphs. Second, when we generate the point-wise features, we use the mask feature which is the weighted sum of the sub-image and its contextless counterpart, to some extent mitigate the impact of salient background objects. Further qualitative results are shown in Fig. 4.
我们在ScanNet [27]和Replica [28]数据集上评估开放词汇3D语义分割性能。我们将我们的方法与两种有竞争力的视觉-语言表示进行比较，即ConceptFusion [16]和ConceptGraphs [22]，并在两个CLIP骨干网络上做消融实验，见表I。就mIOU和F-mIOU而言，HOV-SG大幅优于开放词汇基线。这主要得益于我们做出的以下改进：首先，在合并分段特征时，我们考虑每个分段所覆盖的所有逐点特征，并使用DBSCAN提取主导特征，相比ConceptGraphs采用均值的做法，这提高了鲁棒性。其次，在生成逐点特征时，我们使用掩码特征，即子图像及其无上下文对应项的加权和，在一定程度上减轻显著背景物体的影响。更多定性结果见图4。


## B. Scene Graph Evaluation on Habitat 3D Semantics
## B. 在 Habitat 3D Semantics 上的场景图评估


Object-Level Semantics: Existing open-vocabulary evaluations usually circumvent the problem of measuring true open-vocabulary semantic accuracy. This is due to arbitrary sizes of the investigated label sets, a potentially enormous amount of object categories [26], and the ease of use of existing evaluation protocols [7], [16]. While human-level evaluations solve this problem partly, robust replication of results remains challenging [22].
对象级语义：现有的开放词汇评测通常会回避衡量真正开放词汇语义准确性的问题。这是因为所考察的标签集合大小任意、潜在需要海量对象类别 [26]，以及现有评测协议使用起来很方便 [7]、[16]。尽管人类级评测能在一定程度上解决该问题，但结果的可靠复现仍然很困难 [22]。


TABLE II
表 II


Object-Level Semantics Evaluation on HM3DSEM
在 HM3DSEM 上的对象级语义评估


<table><tr><td>Method</td><td>top5</td><td>${to}{p}_{10}$</td><td>top25</td><td>top100</td><td>top250</td><td>top500</td><td>${\mathrm{{AUC}}}_{k}^{\text{ top }}$</td></tr><tr><td>VLMaps [7]</td><td>0.05</td><td>0.17</td><td>0.54</td><td>15.32</td><td>26.01</td><td>40.02</td><td>56.20</td></tr><tr><td>ConceptGraphs [22]</td><td>18.11</td><td>24.01</td><td>33.00</td><td>55.17</td><td>70.85</td><td>81.55</td><td>84.07</td></tr><tr><td>HOV-SG (ours)</td><td>18.43</td><td>25.73</td><td>36.41</td><td>56.46</td><td>69.95</td><td>80.86</td><td>84.88</td></tr></table>
<table><tbody><tr><td>方法</td><td>top5</td><td>${to}{p}_{10}$</td><td>top25</td><td>top100</td><td>top250</td><td>top500</td><td>${\mathrm{{AUC}}}_{k}^{\text{ top }}$</td></tr><tr><td>VLMaps [7]</td><td>0.05</td><td>0.17</td><td>0.54</td><td>15.32</td><td>26.01</td><td>40.02</td><td>56.20</td></tr><tr><td>ConceptGraphs [22]</td><td>18.11</td><td>24.01</td><td>33.00</td><td>55.17</td><td>70.85</td><td>81.55</td><td>84.07</td></tr><tr><td>HOV-SG（我们的方法）</td><td>18.43</td><td>25.73</td><td>36.41</td><td>56.46</td><td>69.95</td><td>80.86</td><td>84.88</td></tr></tbody></table>


We provide object-level semantic accuracies across all 8 considered scenes within HM3DSem [26] using both the overall ${\mathrm{{AUC}}}_{k}^{\text{ top }}$ metric across 1624 categories as well as accuracies at a few selected thresholds $k$ .
我们在 HM3DSem [26] 的全部 8 个考虑场景中，提供了基于整体 ${\mathrm{{AUC}}}_{k}^{\text{ top }}$ 指标（覆盖 1624 个类别）以及若干选定阈值 $k$ 下的目标级语义准确率。


In this work,we propose the novel ${\mathrm{{AUC}}}_{k}^{\text{ top }}$ metric that quantifies the area under the top- $k$ accuracy curve between the predicted and the actual ground-truth object category. This means computing the ranking of all cosine similarities between the predicted object feature and all possible category text features, which are in turn encoded using a vision-language model (CLIP). Thus, the metric encodes how many erroneous shots are necessary on average before the ground-truth label is predicted correctly. Based on this, the metric encodes the actual open-set similarity while scaling to large, variably-sized label sets. We envision a future use of this metric in various open-vocabulary tasks.
在这项工作中，我们提出了新的 ${\mathrm{{AUC}}}_{k}^{\text{ top }}$ 指标，用于量化预测对象类别与实际真实类别之间的“前- $k$ 准确率曲线”的曲线下面积。这意味着需要计算：将预测对象特征与所有可能类别文本特征之间的余弦相似度的排序；而这些类别文本特征又通过视觉-语言模型（CLIP）进行编码。因此，该指标刻画了在平均意义下，要在多次候选中达到正确预测真实标签之前，需要多少次错误尝试。基于此，该指标在扩展到大规模、变长的标签集合时，仍能反映真实的开放集相似度。我们设想该指标未来可用于各类开放词汇任务。


In order to show the applicability of the ${\mathrm{{AUC}}}_{k}^{\text{ top }}$ metric, we compare HOV-SG against two strong baselines on the Habitat-Semantics dataset [26] in Tab. II, which comprises an enormous label set of 1624 object categories. We observe that VLMaps [7] performs inferior, which is presumably due to its dense feature aggregation. In comparison, ConceptGraphs [22] obtains a competitive score of 84.07% while HOV-SG achieves 84.88%. Additional top- $k$ values shed light on how probable it is to score the correct class within a few tries.
为展示 ${\mathrm{{AUC}}}_{k}^{\text{ top }}$ 指标的适用性，我们在表 II 中的 Habitat-Semantics 数据集 [26] 上，将 HOV-SG 与两个强基线进行比较；该数据集包含规模巨大的 1624 个目标类别标签集。我们观察到 VLMaps [7] 表现较差，推测原因在于其密集特征聚合。相比之下，ConceptGraphs [22] 的得分具有竞争力，为 84.07%，而 HOV-SG 达到 84.88%。额外的前- $k$ 指标进一步说明，在少量尝试之内命中正确类别的可能性有多大。


Hierarchical Concept Retrieval: To take advantage of the hierarchical character of our proposed representation, we evaluate to what extent we can retrieve objects from hierarchical queries of the form: pillow in the living room on the second floor or bottle in the kitchen. To do so, we decompose the query using GPT-3.5 into its sought-after concepts and compute the corresponding CLIP embeddings. In the next step, we hierarchically query against the most suitable floor, the most appropriate room, and lastly, the most suitable object given the query at hand (see Table III). While floor prompting is done naively, we select the room producing the highest maximum cosine similarity to the query room across its ten embeddings. On average, this produces higher success rates compared with mean- or median-based schemes. In the following, we compare HOV-SG against an augmented variant of ConceptGraphs [22]. We equip it with privileged floor information while it scores objects against the requested room and object, which allows it to draw answers at the floor and room level. As shown in Tab. III, HOV-SG shows a significant performance increase of 11.69% on object-room-floor queries and a 2.2% advantage on object-room queries when compared with ConceptGraphs. While ConceptGraphs struggles on larger scenes and under more detailed queries, HOV-SG outperforms it by a significant margin even though it suffers from erroneous room segmentations by design. For more information, we refer to Sec. S.2-D.
分层概念检索：为了利用我们提出的表示的分层特性，我们评估在分层查询（如：二楼客厅里的枕头，或厨房里的瓶子）中，我们能在多大程度上检索到目标。为此，我们使用 GPT-3.5 将查询分解为所要寻找的概念，并计算对应的 CLIP 嵌入。下一步，我们基于查询层级地依次在最合适的楼层、最合适的房间，最后在给定查询下最合适的目标进行检索（见表 III）。虽然楼层的提示是直接进行的，但我们选择能在其十个嵌入中与查询房间产生最高最大余弦相似度的房间。平均而言，这会带来比基于均值或中位数方案更高的成功率。接下来，我们将 HOV-SG 与 ConceptGraphs [22] 的一个增强变体进行比较。我们为其提供了特权的楼层信息：它在对目标进行评分时，基于所请求的房间与目标；这使其能够在楼层与房间层级作答。如表 III 所示，与 ConceptGraphs 相比，HOV-SG 在“目标-房间-楼层”查询上提升了显著的 11.69%，而在“目标-房间”查询上也有 2.2% 的优势。尽管 ConceptGraphs 在更大场景和更细致的查询下表现吃力，但 HOV-SG 仍以显著优势胜出，即便它在设计上会遭遇错误的房间分割。更多信息见 Sec. S.2-D。


TABLE III
表 III


OBJECT RETRIEVAL FROM LANGUAGE QUERIES (HM3DSEM)
来自语言查询的目标检索（HM3DSEM）


<table><tr><td>Query Type</td><td>Method</td><td>#Floors</td><td>#Regions</td><td>#Trials</td><td>SR10[%]</td></tr><tr><td>(obj, room, floor)</td><td>ConceptGraphs HOV-SG (ours)</td><td>1.88</td><td>15.63</td><td>40.63</td><td>16.31 28.00</td></tr><tr><td>(obj, room)</td><td>ConceptGraphs HOV-SG (ours)</td><td>1.88</td><td>15.63</td><td>34.87</td><td>29.26 31.48</td></tr></table>
<table><tbody><tr><td>查询类型</td><td>方法</td><td>#楼层</td><td>#区域</td><td>#试验次数</td><td>SR10[%]</td></tr><tr><td>(物体, 房间, 楼层)</td><td>ConceptGraphs HOV-SG（我们的方法）</td><td>1.88</td><td>15.63</td><td>40.63</td><td>16.31 28.00</td></tr><tr><td>(物体, 房间)</td><td>ConceptGraphs HOV-SG（我们的方法）</td><td>1.88</td><td>15.63</td><td>34.87</td><td>29.26 31.48</td></tr></tbody></table>


Evaluation of 20 frequent distinct object categories across 8 scenes. Success rate criterium: IoU $> {0.1}$ . The floor and room counts refer to the ground-truth labels.
在 8 个场景中评估 20 个常见且彼此独立的目标类别。成功率判据：IoU $> {0.1}$ 。地面与房间的数量指的是真实标签。


TABLE IV
表 IV


REAL-WORLD OBJECT RETRIEVAL FROM LANGUAGE QUERIES
从语言查询中进行真实世界目标检索


<table><tr><td rowspan="2">Query Type</td><td rowspan="2"># Trials</td><td colspan="2">Graph Querying</td><td colspan="2">Goal Navigation</td></tr><tr><td>#Successes</td><td>SR [%]</td><td>Success</td><td>SR [%]</td></tr><tr><td>Object</td><td>41</td><td>29</td><td>70.7</td><td>23</td><td>56.1</td></tr><tr><td>Room</td><td>9</td><td>5</td><td>55.6</td><td>5</td><td>55.6</td></tr><tr><td>Floor</td><td>2</td><td>2</td><td>100</td><td>2</td><td>100</td></tr></table>
<table><tbody><tr><td rowspan="2">查询类型</td><td rowspan="2"># 次尝试</td><td colspan="2">图查询</td><td colspan="2">目标导航</td></tr><tr><td># 成功</td><td>SR [%]</td><td>成功</td><td>SR [%]</td></tr><tr><td>物体</td><td>41</td><td>29</td><td>70.7</td><td>23</td><td>56.1</td></tr><tr><td>房间</td><td>9</td><td>5</td><td>55.6</td><td>5</td><td>55.6</td></tr><tr><td>地板</td><td>2</td><td>2</td><td>100</td><td>2</td><td>100</td></tr></tbody></table>


We count a retrieval as successful whenever the robot is in close vicinity to the object sought after $\left( { \sim  1\mathrm{\;m}}\right)$ .
只要机器人靠近所要查找的 $\left( { \sim  1\mathrm{\;m}}\right)$ 对象，我们就将该检索视为成功。


## C. Real-World Experiments
## C. 现实世界实验


To validate the system in the real world, we conduct multiple navigation trials using a Boston Dynamics Spot quadruped. First, we collect an RGB-D sequence of a two-storage office building comprising a variety of room types as well as objects. Using this data, we create the hierarchical 3D scene graph presentation as introduced in Sec. II and Fig.1. Robot Navigation from Long Queries: We select 41 distinct object goals, 9 room goals, and 2 floor goals and use natural language to query the HOV-SG representation as detailed above. Some examples of the queries are go to floor 0 , navigate to the kitchen on floor 1, or find the plant in the office on floor 0. Similar to the evaluation on Habitat-Semantics, we first evaluate general object retrieval given the scene graph. HOV-SG achieves a 100% success rate on floor retrieval, a 55.6% for room retrieval, and a 70.7% success rate for object retrieval. The major failure cases for room retrieval stem from the visual ambiguity among "meeting room", "seminar room", and "dining room". Based on this, we evaluated the object navigation capabilities from abstract, hierarchical queries in the real world using the Spot quadruped. We observe a 56.1% success rate in object navigation while traversing multiple rooms as well as floors given an abstract long query. These results prove the efficacy of HOV-SG in enabling real-world agents to navigate to language-conditioned goals across multiple floors.
为在真实世界中验证该系统，我们使用 Boston Dynamics Spot 四足机器人进行了多次导航试验。首先，我们采集了一段 RGB-D 序列，场景为一栋两层办公楼，包含多种房间类型以及各类物体。利用这些数据，我们按照第 II 节和图 1 中介绍的方法构建层次化 3D 场景图表示。基于长查询的机器人导航：我们选择了 41 个不同的物体目标、9 个房间目标和 2 个楼层目标，并按上文所述使用自然语言查询 HOV-SG 表示。例如，查询可以是前往 0 层，导航到 1 层的厨房，或查找 0 层办公室里的植物。与在 Habitat-Semantics 上的评估类似，我们首先评估给定场景图时的一般目标检索。HOV-SG 在楼层检索上达到 100% 的成功率，在房间检索上为 55.6%，在物体检索上为 70.7%。房间检索的主要失败案例源于“会议室”“研讨室”和“餐厅”之间的视觉相似性。基于此，我们使用 Spot 四足机器人在真实世界中评估了来自抽象层次查询的物体导航能力。对于一个抽象长查询，我们观察到在跨越多个房间以及楼层时，物体导航的成功率为 56.1%。这些结果证明了 HOV-SG 在支持现实世界智能体跨越多层楼到达语言条件目标方面的有效性。


## IV. CONCLUSION
## IV. 结论


We presented a novel pipeline for constructing hierarchical open-vocabulary 3D scene graphs for robot navigation. Through the semantic decomposition of environments into floors and rooms, we demonstrate effective object retrieval from abstract queries and perform long-horizon navigation in the real world. By doing so, we outperform previous baselines in open-set mapping.
我们提出了一种用于构建面向机器人导航的分层开放词汇3D场景图的新型流程。通过将环境进行语义分解为楼层与房间，我们展示了从抽象查询中进行有效的目标检索，并在真实世界中实现长时域导航。由此，我们在开放集建图任务上优于以往基线。


## REFERENCES
## 参考文献


[1] E. Jefferies and X. Wang, "Semantic cognition: semantic memory and semantic control," in Oxford Research Encyclopedia of Psychology, 2021.
[1] E. Jefferies 和 X. Wang，“语义认知：语义记忆与语义控制”，见《牛津心理学研究百科全书》，2021年。


[2] A. A. Kumar, "Semantic memory: A review of methods, models, and current challenges," Psychonomic Bulletin & Review, vol. 28, pp. 40-80, 2021.
[2] A. A. Kumar，“语义记忆：方法、模型与当前挑战综述”，《心理学通报与评论》，第28卷，pp. 40-80，2021年。


[3] S. C. Hirtle and J. Jonides, "Evidence of hierarchies in cognitive maps," Memory & cognition, vol. 13, no. 3, pp. 208-217, 1985.
[3] S. C. Hirtle 和 J. Jonides，“认知地图中的层级证据”，《记忆与认知》，第13卷，第3期，pp. 208-217，1985年。


[4] B. Kuipers, "The spatial semantic hierarchy," Artificial Intelligence, vol. 119, no. 1, pp. 191-233, 2000.
[4] B. Kuipers，“空间语义层级”，《人工智能》，第119卷，第1期，pp. 191-233，2000年。


[5] H. Voicu, "Hierarchical cognitive maps," Neural Networks, vol. 16, no. 5-6, pp. 569-576, 2003.
[5] H. Voicu，“层级认知地图”，《神经网络》，第16卷，第5-6期，pp. 569-576，2003年。


[6] M. Ahn, A. Brohan, N. Brown, Y. Chebotar, O. Cortes, B. David, C. Finn, C. Fu, K. Gopalakrishnan, K. Hausman, et al., "Do as i can, not as i say: Grounding language in robotic affordances," arXiv preprint arXiv:2204.01691, 2022.
[6] M. Ahn, A. Brohan, N. Brown, Y. Chebotar, O. Cortes, B. David, C. Finn, C. Fu, K. Gopalakrishnan, K. Hausman, et al.，“照我能做的做，不要照我说的做：将语言锚定于机器人可供性”，arXiv 预印本 arXiv:2204.01691，2022年。


[7] C. Huang, O. Mees, A. Zeng, and W. Burgard, "Visual language maps for robot navigation," in Proceedings of the IEEE International Conference on Robotics and Automation (ICRA), London, UK, 2023.
[7] C. Huang, O. Mees, A. Zeng 和 W. Burgard，“用于机器人导航的视觉语言地图”，见 IEEE 国际机器人与自动化会议（ICRA）论文集，英国伦敦，2023年。


[8] O. Mees, J. Borja-Diaz, and W. Burgard, "Grounding language with visual affordances over unstructured data," in 2023 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2023, pp. 11576-11582.
[8] O. Mees, J. Borja-Diaz 和 W. Burgard，“在非结构化数据上用视觉可供性实现语言锚定”，见 2023 IEEE 国际机器人与自动化会议（ICRA）。IEEE，2023年，pp. 11576-11582。


[9] J. Wu, R. Antonova, A. Kan, M. Lepert, A. Zeng, S. Song, J. Bohg, S. Rusinkiewicz, and T. Funkhouser, "Tidybot: Personalized robot assistance with large language models," Autonomous Robots, 2023.
[9] J. Wu, R. Antonova, A. Kan, M. Lepert, A. Zeng, S. Song, J. Bohg, S. Rusinkiewicz 和 T. Funkhouser，“Tidybot：借助大型语言模型的个性化机器人辅助”，《自主机器人》，2023年。


[10] S. Thrun, W. Burgard, and D. Fox, Probabilistic Robotics. MIT Press, 2005.
[10] S. Thrun, W. Burgard 和 D. Fox，《概率机器人学》。MIT出版社，2005年。


[11] N. Vödisch, D. Cattaneo, W. Burgard, and A. Valada, "Continual slam: Beyond lifelong simultaneous localization and mapping through continual learning," in The International Symposium of Robotics Research, 2022, pp. 19-35.
[11] N. Vödisch, D. Cattaneo, W. Burgard 和 A. Valada，“持续性slam：通过持续学习超越终身同步定位与地图构建”，见机器人研究国际研讨会，2022年，pp. 19-35。


[12] J. Arce, N. Vödisch, D. Cattaneo, W. Burgard, and A. Valada, "Padloc: Lidar-based deep loop closure detection and registration using panoptic attention," IEEE Robotics and Automation Letters, vol. 8, no. 3, pp. 1319-1326, 2023.
[12] J. Arce, N. Vödisch, D. Cattaneo, W. Burgard 和 A. Valada，“Padloc：基于激光雷达的深度回环检测与配准，使用全景注意力”，《IEEE机器人与自动化快报》，第8卷，第3期，pp. 1319-1326，2023年。


[13] D. Shah, B. Osiński, S. Levine, et al., "Lm-nav: Robotic navigation with large pre-trained models of language, vision, and action," in Conference on Robot Learning. PMLR, 2023, pp. 492-504.
[13] D. Shah, B. Osiński, S. Levine, et al.，“Lm-nav：结合语言、视觉与动作的大型预训练模型进行机器人导航”，见机器人学习会议。PMLR，2023年，pp. 492-504。


[14] B. Chen, F. Xia, B. Ichter, K. Rao, K. Gopalakrishnan, M. S. Ryoo, A. Stone, and D. Kappler, "Open-vocabulary queryable scene representations for real world planning," in 2023 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2023, pp. 11509-11522.
[14] B. Chen, F. Xia, B. Ichter, K. Rao, K. Gopalakrishnan, M. S. Ryoo, A. Stone 和 D. Kappler，“面向现实世界规划的可查询开放词汇场景表示”，见 2023 IEEE 国际机器人与自动化会议（ICRA）。IEEE，2023年，pp. 11509-11522。


[15] C. Huang, O. Mees, A. Zeng, and W. Burgard, "Audio visual language maps for robot navigation," in Proceedings of the International Symposium on Experimental Robotics (ISER), Chiang Mai, Thailand, 2023.
[15] C. Huang, O. Mees, A. Zeng 和 W. Burgard，“用于机器人导航的音视觉语言地图”，见实验机器人国际研讨会论文集，泰国清迈，2023年。


[16] K. M. Jatavallabhula, A. Kuwajerwala, Q. Gu, M. Omama, T. Chen, S. Li, G. Iyer, S. Saryazdi, N. Keetha, A. Tewari, et al., "Conceptfusion: Open-set multimodal 3d mapping," Robotics: Science And Systems, 2023.
[16] K. M. Jatavallabhula, A. Kuwajerwala, Q. Gu, M. Omama, T. Chen, S. Li, G. Iyer, S. Saryazdi, N. Keetha, A. Tewari, 等，《Conceptfusion：开放集多模态3D建图》，Robotics: Science And Systems，2023。


[17] S. Peng, K. Genova, C. M. Jiang, A. Tagliasacchi, M. Pollefeys, and T. Funkhouser, "Openscene: 3d scene understanding with open vocabularies," in CVPR, 2023.
[17] S. Peng, K. Genova, C. M. Jiang, A. Tagliasacchi, M. Pollefeys, 以及 T. Funkhouser，《Openscene：使用开放词汇的3D场景理解》，发表于CVPR，2023。


[18] N. M. M. Shafiullah, C. Paxton, L. Pinto, S. Chintala, and A. Szlam, "Clip-fields: Weakly supervised semantic fields for robotic memory," arXiv preprint arXiv: Arxiv-2210.05663, 2022.
[18] N. M. M. Shafiullah, C. Paxton, L. Pinto, S. Chintala, 以及 A. Szlam，《Clip-fields：用于机器人记忆的弱监督语义场》，arXiv预印本arXiv:Arxiv-2210.05663，2022。


[19] E. Greve, M. Büchner, N. Vödisch, W. Burgard, and A. Valada, "Collaborative dynamic 3d scene graphs for automated driving," arXiv preprint arXiv:2309.06635, 2023.
[19] E. Greve, M. Büchner, N. Vödisch, W. Burgard, 以及 A. Valada，《用于自动驾驶的协作动态3D场景图》，arXiv预印本arXiv:2309.06635，2023。


[20] N. Hughes, Y. Chang, and L. Carlone, "Hydra: A real-time spatial perception system for 3D scene graph construction and optimization," in Robotics: Science And Systems, 2022.
[20] N. Hughes, Y. Chang, 以及 L. Carlone，《Hydra：用于3D场景图构建与优化的实时空间感知系统》，发表于Robotics: Science And Systems，2022。


[21] A. Rosinol, A. Gupta, M. Abate, J. Shi, and L. Carlone, "3D dynamic scene graphs: Actionable spatial perception with places, objects, and humans," Robotics: Science And Systems, 2020.
[21] A. Rosinol, A. Gupta, M. Abate, J. Shi, 以及 L. Carlone，《3D动态场景图：结合地点、物体与人进行可执行的空间感知》，Robotics: Science And Systems，2020。


[22] Q. Gu, A. Kuwajerwala, S. Morin, K. Jatavallabhula, B. Sen, A. Agarwal, C. Rivera, W. Paul, K. Ellis, R. Chellappa, C. Gan, C. de Melo, J. Tenenbaum, A. Torralba, F. Shkurti, and L. Paull, "Conceptgraphs: Open-vocabulary 3d scene graphs for perception and planning," arXiv, 2023.
[22] Q. Gu, A. Kuwajerwala, S. Morin, K. Jatavallabhula, B. Sen, A. Agarwal, C. Rivera, W. Paul, K. Ellis, R. Chellappa, C. Gan, C. de Melo, J. Tenenbaum, A. Torralba, F. Shkurti, 以及 L. Paull，《Conceptgraphs：用于感知与规划的开放词汇3D场景图》，arXiv，2023。


[23] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark, et al., "Learning transferable visual models from natural language supervision," in International conference on machine learning. PMLR, 2021, pp. 8748-8763.
[23] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark，等，《通过自然语言监督学习可迁移的视觉模型》，发表于机器学习国际会议。PMLR，2021，pp. 8748-8763。


[24] A. Kirillov, E. Mintun, N. Ravi, H. Mao, C. Rolland, L. Gustafson, T. Xiao, S. Whitehead, A. C. Berg, W.-Y. Lo, P. Dollár, and R. Girshick, "Segment anything," arXiv:2304.02643, 2023.
[24] A. Kirillov, E. Mintun, N. Ravi, H. Mao, C. Rolland, L. Gustafson, T. Xiao, S. Whitehead, A. C. Berg, W.-Y. Lo, P. Dollár, 以及 R. Girshick，《Segment anything》，arXiv:2304.02643，2023。


[25] S. Thrun and A. Bücken, "Integrating grid-based and topological maps for mobile robot navigation," in AAAI, 1996.
[25] S. Thrun 与 A. Bücken，《为移动机器人导航融合基于栅格的地图与拓扑地图》，发表于AAAI，1996。


[26] K. Yadav, R. Ramrakhya, S. K. Ramakrishnan, T. Gervet, J. Turner, A. Gokaslan, N. Maestre, A. X. Chang, D. Batra, M. Savva, et al., "Habitat-matterport 3d semantics dataset," in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 4927-4936.
[26] K. Yadav, R. Ramrakhya, S. K. Ramakrishnan, T. Gervet, J. Turner, A. Gokaslan, N. Maestre, A. X. Chang, D. Batra, M. Savva，等，《Habitat-matterport 3D语义数据集》，载于IEEE/CVF计算机视觉与模式识别会议论文集，2023，pp. 4927-4936。


[27] A. Dai, A. X. Chang, M. Savva, M. Halber, T. Funkhouser, and M. Nießner, "Scannet: Richly-annotated 3d reconstructions of indoor scenes," 2017, pp. 5828-5839.
[27] A. Dai, A. X. Chang, M. Savva, M. Halber, T. Funkhouser, 以及 M. Nießner，《Scannet：对室内场景的高质量标注3D重建》，2017，pp. 5828-5839。


[28] J. Straub, T. Whelan, L. Ma, Y. Chen, E. Wijmans, S. Green, J. J. Engel, R. Mur-Artal, C. Ren, S. Verma, et al., "The replica dataset: A digital replica of indoor spaces," arXiv preprint arXiv:1906.05797, 2019.
[28] J. Straub, T. Whelan, L. Ma, Y. Chen, E. Wijmans, S. Green, J. J. Engel, R. Mur-Artal, C. Ren, S. Verma，等，《Replica数据集：室内空间的数字孪生》，arXiv预印本arXiv:1906.05797，2019。


[29] F. Liang, B. Wu, X. Dai, K. Li, Y. Zhao, H. Zhang, P. Zhang, P. Vajda, and D. Marculescu, "Open-vocabulary semantic segmentation with mask-adapted clip," in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 7061-7070.
[29] F. Liang, B. Wu, X. Dai, K. Li, Y. Zhao, H. Zhang, P. Zhang, P. Vajda 和 D. Marculescu，“采用 mask-adapted CLIP 的开放词汇语义分割”，载于 IEEE/CVF 计算机视觉与模式识别会议论文集，2023 年，7061-7070 页。


# Hierarchical Open-Vocabulary 3D Scene Graphs for Language-Grounded Robot Navigation
# 面向语言引导机器人导航的分层开放词汇三维场景图


- Supplementary Material -
- 补充材料 -


Abdelrhman Werby ${}^{1 * }$ , Chenguang Huang ${}^{1 * }$ , Martin Büchner ${}^{1 * }$ , Abhinav Valada ${}^{1}$ , and Wolfram Burgard ${}^{2}$
Abdelrhman Werby ${}^{1 * }$ , Chenguang Huang ${}^{1 * }$ , Martin Büchner ${}^{1 * }$ , Abhinav Valada ${}^{1}$ ，以及 Wolfram Burgard ${}^{2}$


In this supplementary material, we expand upon multiple aspects of our main paper. In Sec. S.1, we detail several design choices regarding the proposed segment merging, the hierarchical scene graph construction, the navigational graph as well as a semantic localization scheme. In Sec. S.2, we additionally present experimental results that support the claims introduced in the manuscript. This includes a more detailed discussion of the proposed open-vocabulary metric, an analysis regarding identified semantic room categories, and additional baselines regarding object retrieval from language queries. Moreover, we provide insightful visualizations of the produced multi-story scene graphs, representing scenes from both Habitat Semantics as well as our real-world environment.
在本补充材料中，我们将对主论文的多个方面进行扩展。在第 S.1 节中，我们详细说明了所提出的分割合并、分层场景图构建、导航图以及语义定位方案方面的若干设计选择。在第 S.2 节中，我们进一步给出支撑手稿中所提出结论的实验结果。这包括对所提出的开放词汇度量进行更细致的讨论、对已识别语义房间类别的分析，以及关于基于语言查询的目标检索的额外基线。此外，我们还提供对生成的多层场景图的直观可视化，展示了来自 Habitat Semantics 以及我们的真实世界环境的场景。


### S.1. METHOD DETAILS
### S.1. 方法细节


## A. Merging Frame-Wise Segments
## A. 合并逐帧分割


Given the frame-wise 3D segments created at all timesteps ${}^{W}{P}_{ik}$ where $i = 1,\ldots ,N,k = 1,\ldots ,{K}_{i}$ ,and ${}^{W}$ indicates world coordinates, we merge the overlapping point clouds across frames based on the geometric distances between point clouds, following a similar merging scheme as Gu et al. [22]. We maintain a list of global 3D segments that are incrementally constructed $\mathcal{S} = {\left\{  {S}_{j}\right\}  }_{j = 1,\ldots ,J}$ ,where $J$ is the total segment number. For each new frame, we add all segments to the global segments list and compute the pair-wise overlapping ratio between all segment pairs in the global segments list. The overlapping ratio $R\left( {m,n}\right)$ between segment $m$ and $n$ is computed as:
给定在所有时间步 ${}^{W}{P}_{ik}$ 创建的逐帧 3D 分割，其中 $i = 1,\ldots ,N,k = 1,\ldots ,{K}_{i}$，且 ${}^{W}$ 表示世界坐标，我们根据点云之间的几何距离合并跨帧重叠点云，采用与 Gu 等人[22]类似的合并方案。我们维护一个全局 3D 分割列表，并以增量方式构建 $\mathcal{S} = {\left\{  {S}_{j}\right\}  }_{j = 1,\ldots ,J}$，其中 $J$ 是分割总数。对于每个新帧，我们将所有分割加入全局分割列表，并计算全局分割列表中所有分割对之间的成对重叠比例。分割 $m$ 与 $n$ 之间的重叠比例 $R\left( {m,n}\right)$ 计算如下：


$$
R\left( {m,n}\right)  = \max \left( {\operatorname{overlap}\left( {{S}_{m},{S}_{n}}\right) ,\operatorname{overlap}\left( {{S}_{n},{S}_{m}}\right) }\right) , \tag{1}
$$



where overlap(;) is computed by taking the number of points in $a$ that can find a neighboring point in $b$ within a certain distance threshold divided by the total number of points in a. Different from Gu et al. [22], who merges new segments with one global segment that has the largest overlapping ratio, we construct a graph based on the overlapping ratios among the segments. When the ratio is above a certain threshold, we establish an edge between the two segments and merge all connected segments. In this way, one segment can merge with more segments, which is useful in situations in which an incoming segment is filling, e.g., a gap between two already registered global segments.
其中 overlap(;) 的计算方式是：统计 $a$ 中能在一定距离阈值内找到 $b$ 中邻近点的点数，再除以 a 的总点数。不同于 Gu 等人[22]将新分割与重叠比例最大的一个全局分割合并，我们基于分割之间的重叠比例构建图。当该比例高于某个阈值时，我们在两个分割之间建立边，并合并所有连通分割。这样，一个分割可以与更多分割合并，这在输入分割正在填补例如两个已注册全局分割之间的空隙时很有用。


## B. Scene Graph Formalization
## B. 场景图形式化


We formalize our graph as $\mathcal{G} = \left( {\mathcal{N},\mathcal{E}}\right)$ where $\mathcal{N}$ denotes the nodes and $\mathcal{E}$ denotes the edges. The nodes can be expressed as $\mathcal{N} = {\mathcal{N}}_{\text{ root }} \cup  {\mathcal{N}}_{F} \cup  {\mathcal{N}}_{R} \cup  {\mathcal{N}}_{O}$ ,consisting of a root node ${\mathcal{N}}_{\text{ root }}$ ,floor nodes ${\mathcal{N}}_{F}$ ,room nodes ${\mathcal{N}}_{R}$ ,and object nodes ${\mathcal{N}}_{O}$ . Each node in the graph except the root node ${\mathcal{N}}_{\text{ root }}$ contains the point cloud of the concept it refers to and the open-vocabulary features associated with it. The edges can be written as $\mathcal{E} = {\mathcal{E}}_{0F} \cup  {\mathcal{E}}_{FR} \cup  {\mathcal{E}}_{RO}$ . Here, ${\mathcal{E}}_{0F}$ represents the edges between the root node and the floor nodes, ${\mathcal{E}}_{FR}$ represents the edges between the floor nodes and the room nodes,and lastly, ${\mathcal{E}}_{RO}$ denotes the edges between the room and object nodes.
我们将图形式化为 $\mathcal{G} = \left( {\mathcal{N},\mathcal{E}}\right)$ ，其中 $\mathcal{N}$ 表示节点，$\mathcal{E}$ 表示边。节点可表示为 $\mathcal{N} = {\mathcal{N}}_{\text{ root }} \cup  {\mathcal{N}}_{F} \cup  {\mathcal{N}}_{R} \cup  {\mathcal{N}}_{O}$ ，由根节点 ${\mathcal{N}}_{\text{ root }}$ 、地板节点 ${\mathcal{N}}_{F}$ 、房间节点 ${\mathcal{N}}_{R}$ 和物体节点 ${\mathcal{N}}_{O}$ 构成。图中的每个节点（除根节点 ${\mathcal{N}}_{\text{ root }}$ 外）都包含其所指概念的点云以及与之关联的开放词汇特征。边可写为 $\mathcal{E} = {\mathcal{E}}_{0F} \cup  {\mathcal{E}}_{FR} \cup  {\mathcal{E}}_{RO}$ 。其中，${\mathcal{E}}_{0F}$ 表示根节点与地板节点之间的边，${\mathcal{E}}_{FR}$ 表示地板节点与房间节点之间的边，最后，${\mathcal{E}}_{RO}$ 表示房间节点与物体节点之间的边。


## C. Hierarchical 3D Scene Graph
## C. 层次化 3D 场景图


Floor Segmentation: Given the point cloud of the whole environment, we plot the histogram of all points along the axis indicating the height (bin size ${0.01}\mathrm{\;m}$ ). Next,we extract local peaks in this histogram (within a local range of ${0.2}\mathrm{\;m}$ ) and select only peaks that are exceed at least ${90}\%$ of the highest peak. We apply DBSCAN and select the top-2 heights in each cluster. After that, every 2 consecutive values in the sorted height vector represents a single floor (floor and ceiling) in the building. The floor segmentation process is shown in Fig. S.1. Using the heights above, we can extract floor point clouds for each floor ${\mathcal{P}}_{Fl}$ where $l$ is the floor number. We compute the CLIP text embedding of "floor \{#\}" and pack it with the floor point cloud as a floor node ${N}_{Fl}$ in the graph. An edge between the root node and this floor node $E\left( {{N}_{\text{ root }},{N}_{Fl}}\right)  \in  {\mathcal{E}}_{0F}$ is also established.
楼层分割：给定整个环境的点云，我们沿高度轴绘制所有点的直方图（箱宽为 ${0.01}\mathrm{\;m}$ ）。接着，我们在该直方图上提取局部峰值（局部范围为 ${0.2}\mathrm{\;m}$ ），并仅保留至少达到最高峰值 ${90}\%$ 的峰。我们应用 DBSCAN，并在每个簇中选择前两个高度。之后，排序后的高度向量中每两个相邻值共同表示建筑中的一个楼层（楼板与顶棚）。楼层分割流程见图 S.1。利用上述高度，我们可以提取每个楼层的楼层点云 ${\mathcal{P}}_{Fl}$，其中 $l$ 为楼层编号。我们计算文本“floor \{#\}”的 CLIP 文本嵌入，并将其与该楼层点云打包，作为图中的楼层节点 ${N}_{Fl}$。同时，我们也在根节点与该楼层节点之间建立一条边 $E\left( {{N}_{\text{ root }},{N}_{Fl}}\right)  \in  {\mathcal{E}}_{0F}$。


Room Segmentation: After segmenting the floors, we use each floor point cloud to further segment room regions. We first compute the 2D histogram of the floor point in the bird's-eye-view. We extract the binary wall skeleton mask ${M}_{w} \in  \{ 0,1{\} }^{\bar{H} \times  \bar{W}}$ in the top-down map by selecting locations where the histogram density is higher than a certain threshold. We apply dilation to the wall mask to enhance the skeleton and compute a Euclidean Distance Field based on this. We extract a list of isolated regions by taking locations that have distance values higher than a certain threshold. These regions are later used as the seeds for the watershed algorithm to obtain a list of region masks ${\left\{  {M}_{r}\right\}  }_{r} = 1\ldots R$ . The room segmentation process is shown in Fig. S.1.
房间分割：完成楼层分割后，我们利用每个楼层点云进一步分割房间区域。首先，我们在鸟瞰视角下计算楼层点的二维直方图。通过选择直方图密度高于某个阈值的位置，我们在俯视图中提取二值墙体骨架掩码 ${M}_{w} \in  \{ 0,1{\} }^{\bar{H} \times  \bar{W}}$。我们对墙体掩码进行膨胀以增强骨架，并基于此计算欧氏距离场。通过选取距离值高于某个阈值的位置，我们提取一组孤立区域。这些区域随后作为分水岭算法的种子，用于得到一系列区域掩码 ${\left\{  {M}_{r}\right\}  }_{r} = 1\ldots R$。房间分割流程见图 S.1。


Using each top-down region mask ${M}_{r}$ ,we extract all points falling into both the region mask as well as the respective floor to form room point clouds. Simultaneously, we collect camera poses within each room, and encode their images with a CLIP image encoder, generating CLIP features for all room views. Instead of assigning only one feature to each room, we propose applying the k-means algorithm to extract $k$ representative features for each room,covering diverse aspects of each region. During the query, given a list of room categories, we encode them with a CLIP text encoder and compute the cosine similarity between each representative feature and each category, resulting in a similarity matrix. By performing the argmax operation, we obtain scores from all representatives, which allows the retrieval of the room type voted by the majority. These $k$ representative embeddings and the room point cloud are jointly stored in the room node ${N}_{FlRr}$ in the graph,representing room $r$ on floor $l$ . An edge between the floor node and each room node $E\left( {{N}_{Fl},{N}_{Rr}}\right)  \in  {\mathcal{E}}_{FR}$ is established. The room feature construction and querying routines are illustrated in Fig. 3. Object Identification: Given the room point cloud, we associate object-level 3D segments that show a point cloud overlap with a potential candidate room in the bird's-eye-view. Whenever a segment shows zero overlap with any room, we associate it to the room with the smallest Euclidean distance. We prompt a list of categories to the segment features to classify the segments. Then we compute the pairwise overlapping ratio for all segments as in Sec. S.1-A. We then merge segments that are in the same category and with overlapping ratios above a certain threshold. Each merged point cloud is an object node ${N}_{FlRrOo}$ ,denoting object $o$ in room $r$ on floor $l$ ,and an edge $E\left( {{N}_{FlRr},{N}_{FlRrOo}}\right)  \in  {\mathcal{E}}_{RO}$ is established between the object ${N}_{FlRrOm}$ and the room node ${N}_{FlRr}$ .
利用每个俯视区域掩码 ${M}_{r}$ ，我们提取同时落在该区域掩码及对应楼层中的所有点，形成房间点云。与此同时，我们在每个房间内收集相机位姿，并使用 CLIP 图像编码器对其图像进行编码，得到所有房间视角的 CLIP 特征。我们不再为每个房间只分配一个特征，而是提出使用 k-means 算法为每个房间提取 $k$ 个代表性特征，以覆盖每个区域的多样方面。在查询阶段，给定一组房间类别，我们用 CLIP 文本编码器对其编码，并计算每个代表性特征与各类别之间的余弦相似度，从而得到相似度矩阵。通过执行 argmax 操作，我们获得来自所有代表者的得分，使得可以通过多数投票来检索房间类型。将这些 $k$ 个代表性嵌入与房间点云共同存储在图中的房间节点 ${N}_{FlRr}$ 中，表示位于楼层 $l$ 上的房间 $r$。在楼层节点与每个房间节点之间建立一条边 $E\left( {{N}_{Fl},{N}_{Rr}}\right)  \in  {\mathcal{E}}_{FR}$。房间特征构建与查询流程如图 3 所示。目标识别：给定房间点云，我们将表现为在鸟瞰视角下与候选房间点云存在重叠的物体级 3D 分段进行关联。每当某个分段与任何房间均无重叠时，我们将其关联到欧氏距离最小的房间。我们将一组类别提示给分段特征以对分段进行分类。然后，我们像在 Sec. S.1-A 中那样，对所有分段计算两两重叠比。接着，我们将属于同一类别且重叠比高于某个阈值的分段进行合并。每个合并后的点云都是一个物体节点 ${N}_{FlRrOo}$，表示位于楼层 $l$、房间 $r$ 中的物体 $o$，并在该物体 ${N}_{FlRrOm}$ 与房间节点 ${N}_{FlRr}$ 之间建立一条边 $E\left( {{N}_{FlRr},{N}_{FlRrOo}}\right)  \in  {\mathcal{E}}_{RO}$。


---



*These authors contributed equally.
*这些作者贡献相同。


${}^{1}$ Department of Computer Science,University of Freiburg,Germany.
${}^{1}$ 计算机科学系，德国弗赖堡大学。


${}^{2}$ Department of Eng.,University of Technology Nuremberg,Germany
${}^{2}$ 德国纽伦堡应用科技大学（Department of Eng.）


---



<img src="https://cdn.noedgeai.com/bo_d8j3o4491nqc738ucjg0_7.jpg?x=149&y=137&w=1504&h=333&r=0"/>



Fig. S.1. Floor and Room Segmentation. Given the point cloud of the whole environment, floor and room nodes are subsequently derived based on geometric heuristics. Floor boundaries are computed by finding peaks of the pixel density along the height direction followed by filtering while room segment masks are extracted with the Watershed algorithm.
图 S.1. 楼层与房间分割。给定整个环境的点云，基于几何启发式方法依次得到楼层节点与房间节点。通过在高度方向上寻找像素密度的峰值，并随后进行筛选来计算楼层边界；房间分割掩码则采用 Watershed 算法提取。


## D. Actionable Navigational Graph
## D. 可行动导航图


The creation of actionable graphs involves constructing per-floor and cross-floor action graphs. For the floor-level action graph, the approach entails computing the free space map of the floor and creating a Voronoi graph [25] based on it. To obtain the free space map, we first project all camera poses on the floor to the top-down plane and consider areas within a certain radius of each pose as navigable. Subsequently, the entire floor's region is obtained by projecting all points on the floor to the top-down plane, and an obstacle map is generated based on points within a predefined height range $\left\lbrack  {{y}_{\min } + {\delta }_{1},{y}_{\min } + {\delta }_{2}}\right\rbrack$ ,where ${y}_{\min }$ is the minimal height of the floor points and ${\delta }_{1},{\delta }_{2}$ are two thresholds we define. In this paper,we use ${\delta }_{1} = {0.2}$ and ${\delta }_{2} = {1.5}$ . By combining the pose region map with the floor region map and subtracting the obstacle region map, the free space map for the floor is derived. The Voronoi graph of this free map yields the floor action graph. (See Fig. S.2).
可行动图的构建涉及生成逐层和跨层动作图。对于楼层级动作图，该方法包括计算该楼层的自由空间地图，并据此创建 Voronoi 图 [25]。为获得自由空间地图，我们首先将楼层上的所有相机位姿投影到俯视平面，并将每个位姿一定半径范围内的区域视为可通行区域。随后，将楼层中的所有点投影到俯视平面以获得整个楼层区域，并基于位于预定义高度范围 $\left\lbrack  {{y}_{\min } + {\delta }_{1},{y}_{\min } + {\delta }_{2}}\right\rbrack$ 内的点生成障碍物地图，其中 ${y}_{\min }$ 是楼层点的最小高度，${\delta }_{1},{\delta }_{2}$ 是我们定义的两个阈值。本文中，我们使用 ${\delta }_{1} = {0.2}$ 和 ${\delta }_{2} = {1.5}$。通过将位姿区域图与楼层区域图结合，并减去障碍区域图，得到该楼层的自由空间地图。该自由地图的 Voronoi 图即为楼层动作图。（见图 S.2）。


<img src="https://cdn.noedgeai.com/bo_d8j3o4491nqc738ucjg0_7.jpg?x=920&y=629&w=730&h=319&r=0"/>



Fig. S.2. Single-floor Navigational Graphs.
图 S.2. 单层导航图。


<img src="https://cdn.noedgeai.com/bo_d8j3o4491nqc738ucjg0_7.jpg?x=917&y=1028&w=732&h=342&r=0"/>



Fig. S.3. Cross-floor Navigational Graphs.
图 S.3. 跨层导航图。


To enable navigation across floors, camera poses on stairs are connected to form a stairs graph. Then, the closest nodes between the stairs graph and the floor graph are selected and connected, see Fig. S.3.
为实现跨楼层导航，将楼梯上的相机位姿连接起来形成楼梯图。随后，选取楼梯图与楼层图之间最近的节点并将其连接，见图 S.3。


## E. Semantic Localization
## E. 语义定位


HOV-SG achieves agent localization within the graph using only RGB images and local odometry using a simple particle filter. The process involves randomly initializing K particles within the free space map, estimated from each floor's point cloud. Subsequently, the global CLIP feature of the RGB image and the CLIP feature of objects within the image are extracted using the same pipeline as used for the graph creation. In the prediction step, the particle poses are updated based on robot odometry. Thus, we assign each particle a floor and room based on its updated coordinates. In the update step, we calculate cosine similarity scores between the current RGB image's global CLIP feature and the graph's room features for each particle. Additionally, scores are computed between object features in the RGB image and observed objects in front of each particle. Then, particle weights are adjusted based on these similarity scores. This integrated approach allows HOV-SG to semantically localize the agent within the graph at the floor and room level within a short span of 10 observed frames.
HOV-SG 仅使用 RGB 图像和简单粒子滤波实现代理在图中的定位。该过程包括在可通行空间地图中随机初始化 K 个粒子，地图由各楼层的点云估计得到。随后，使用与生成图谱相同的流程提取 RGB 图像的全局 CLIP 特征，以及图像中物体的 CLIP 特征。在预测步骤中，根据机器人的里程计更新粒子的位姿。因此，我们根据更新后的坐标为每个粒子分配其所在楼层和房间。在更新步骤中，我们对每个粒子计算当前 RGB 图像全局 CLIP 特征与图谱中房间特征之间的余弦相似度分数。此外，还会在每个粒子前方的可观测物体之间，计算 RGB 图像中物体特征与其对应观测物体之间的分数。然后，依据这些相似度分数调整粒子权重。这种集成方法使得 HOV-SG 能在短短 10 帧观测内，将代理在楼层与房间层级内语义化地定位到图谱中。


<img src="https://cdn.noedgeai.com/bo_d8j3o4491nqc738ucjg0_8.jpg?x=150&y=155&w=749&h=317&r=0"/>



Fig. S.4. The language-grounded navigation module of HOV-SG allows to parse complex queries such as "find the toilet in the bathroom on floor 2" into three queries using a large language model (GPT-3.5) - one each for the floor, room, and object levels. Leveraging HOV-SG's hierarchical structure, we progressively narrow down the search space by querying at each level. Once the target location is identified, the action graph in HOV-SG is used to plan a path from the starting pose to the target.
图 S.4. HOV-SG 的语言引导导航模块允许借助大型语言模型（GPT-3.5）将“在第 2 层楼的浴室里找到厕所”这类复杂查询解析为三个查询——分别对应楼层、房间和物体层级。利用 HOV-SG 的层级结构，我们通过在每个层级逐步提问来不断缩小搜索空间。一旦目标位置被确定，便使用 HOV-SG 中的行动图从起始位姿规划到目标。


### S.2. EXPERIMENTAL EVALUATION
### S.2. 实验评估


## A. Open-Vocabulary Similarity Metric $\left( {{AU}{C}_{k}^{\text{ top }}}\right)$
## A. 开放词汇相似度度量 $\left( {{AU}{C}_{k}^{\text{ top }}}\right)$


In this section, we present a visualization of the open-vocabulary similarity metric ${\mathrm{{AUC}}}_{k}^{\text{ top }}$ introduced in the paper. As shown in Fig. S.5,the ${\mathrm{{AUC}}}_{k}^{\text{ top }}$ metric represents the area under the top-k accuracy curve. The closer this curve is to the upper left point, the higher the open-vocabulary similarity. Instead of showing the accuracy at distinct values of $k$ as in the main paper,we normalize $k$ over the extent of the label category set, which contains 1624 categories for HM3DSem. This also shows visually how the ${\mathrm{{AUC}}}_{k}^{\text{ top }}$ metric provides a dependable measure for large but variably sized label sets. We envision the future use of this metric in a number of open-vocabulary tasks.
在本节中，我们展示论文中提出的开放词汇相似度度量 ${\mathrm{{AUC}}}_{k}^{\text{ top }}$ 的可视化。如图 S.5 所示，${\mathrm{{AUC}}}_{k}^{\text{ top }}$ 度量表示位于 top-k 准确率曲线下方的面积。该曲线越接近左上角点，开放词汇相似度越高。我们不再像主论文那样在不同的 $k$ 取值处展示准确率，而是将 $k$ 在标签类别集合的范围内进行归一化；该集合包含 HM3DSem 的 1624 个类别。这样也能直观地表明，${\mathrm{{AUC}}}_{k}^{\text{ top }}$ 度量能够为规模较大但大小不一的标签集合提供可靠的度量。我们设想该度量在多种开放词汇任务中的未来应用。


## B. Open-Vocabulary Semantics Evaluation
## B. 开放词汇语义评估


To allow for a fair comparison, we perform a linear assignment among predicted and GT objects and only consider predicted objects that show an IoU $> {50}\%$ with the ground truth. Since VLMaps [7] does not predict masks by design, it takes the masks predicted by HOV-SG and evaluates wrt. those.
为确保公平比较，我们在预测对象与真实对象之间进行线性匹配，并且只考虑与真实对象的 IoU $> {50}\%$ 的预测对象。由于 VLMaps [7] 在设计上不预测掩码，它采用 HOV-SG 预测的掩码并据此进行评估。


## C. Room Classification
## C. 房间分类


We quantitatively support our proposed view embedding-based room category labeling method by comparing it against two strong baselines across the set of 8 scenes on HM3DSem. Both baselines rely on object labels to classify room categories. To draw a fair comparison, all methods rely on ground-truth room segmentation. Thus, objects are assigned to rooms based on ground-truth room layouts. This is different from the general HOV-SG method, which estimates room segments. Please refer to the main manuscript for room segmentation results.
我们通过将所提基于视图嵌入的房间类别标注方法与 HM3DSem 上 8 个场景中的两个强基线进行比较，定量支持该方法。这两个基线都依赖物体标签来分类房间类别。为进行公平比较，所有方法都依赖真实房间分割。因此，物体会根据真实房间布局分配到各个房间。这不同于通用的 HOV-SG 方法，后者会估计房间分割。房间分割结果请参见主文稿。


<img src="https://cdn.noedgeai.com/bo_d8j3o4491nqc738ucjg0_8.jpg?x=952&y=141&w=654&h=375&r=0"/>



Fig. S.5. We visualize the ${\mathrm{{AUC}}}_{k}^{\text{ top }}$ curve for different evaluation thresholds $k$ ,which this plot measures in terms of percent out of the total number of categories (HM3DSem: 1624). The shown curve represents the results of our method HOV-SG on the HM3DSem scene 00824.
图 S.5。我们可视化了针对不同评估阈值 $k$ 的 ${\mathrm{{AUC}}}_{k}^{\text{ top }}$ 曲线，该图以总类别数的百分比进行度量（HM3DSem：1624）。所示曲线代表我们的 HOV-SG 方法在 HM3DSem 场景 00824 上的结果。


In this evaluation, we utilize a closed set of room categories. To do so, we manually labeled the regions of the 8 scenes as given in Tab. S.1. The HM3DSem dataset does not provide annotated room categories but merely educated votes, which are often not sufficient. We will make the used room labels available as part of this work. The first and privileged baseline operates on ground-truth maps. This means that ground-truth rooms, objects (masks), and object categories are taken. In the next step, the baseline prompts GPT3.5 to provide a room category guess (out of the closed set of room categories) based on the objects contained in each room. The second and unprivileged baseline applies the same principle of prompting GPT3.5 but relies on the solutions obtained by HOV-SG. This means that object masks are not perfect and the top-1 predicted object category is taken to label objects. In general, we expect that the number of objects will be different from the privileged baseline because of under- and over-segmentation of the produced solutions. In comparison, our view embedding method relies on 10 distinct view embeddings which are scored against the defined set of room categories. The final predicted room category is defined by the room category that showed the highest similarity across all view embeddings, which is further described in the main manuscript.
在此评估中，我们使用一个封闭集的房间类别。为此，我们按照表 S.1 手动标注了 8 个场景的区域。HM3DSem 数据集并未提供标注的房间类别，仅提供一些经验性投票，这通常并不充分。我们将把所用房间标签作为本工作的一部分公开。第一个且具特权的基线基于真实地图运行。这意味着使用真实房间、物体（掩码）和物体类别。接下来，该基线提示 GPT3.5 基于每个房间所包含的物体，给出一个房间类别猜测（从封闭集房间类别中选择）。第二个且无特权的基线采用相同的 GPT3.5 提示原则，但依赖 HOV-SG 得到的结果。这意味着物体掩码并不完美，并采用 Top-1 预测的物体类别来标注物体。一般来说，由于生成结果存在欠分割和过分割，我们预期物体数量会与具特权基线不同。相比之下，我们的视图嵌入方法依赖 10 个不同的视图嵌入，这些嵌入会与定义的房间类别集合进行评分。最终预测的房间类别由在所有视图嵌入中相似度最高的房间类别确定，这在主文稿中有进一步描述。


We apply two different evaluation criteria: The first accuracy called ${\mathrm{{Acc}}}_{\mathrm{{GT}}}$ fosters replicability by evaluating whether the predicted and the ground-truth room category are text-wise equal. Different from that, the performance regarding the ${\mathrm{{Acc}}}_{\text{ valid }}$ metric is produced via human evaluation. This is crucial as room categories are not always fully determinable when labeling such as combined kitchen and living room areas. On top of that, the answers provided by GPT3.5 do not always state definitive categories because of frequent hallucinations. This is exacerbated by a high number of objects per room. This particularly applies to the unprivileged baselines when facing over-segmentation. In order to circumvent this, we manually filter all outputs across the set of 8 scenes and check whether the LLM leaned towards the correct answer, which boosts results in favor of the LLM-based methods.
我们采用两种不同的评估标准：第一种准确率称为 ${\mathrm{{Acc}}}_{\mathrm{{GT}}}$，通过评估预测房间类别与真实房间类别在文本上是否相同来促进可复现性。不同于此，关于 ${\mathrm{{Acc}}}_{\text{ valid }}$ 指标的性能则通过人工评估得出。这一点至关重要，因为在标注诸如厨房与客厅组合区域时，房间类别并不总能完全确定。此外，GPT3.5 给出的答案也并不总能明确给出类别，因为常会出现幻觉。这在每个房间包含大量物体时尤为严重。尤其是在面对过分割时，这一点对无特权基线影响更大。为规避这一问题，我们手动筛查该 8 个场景中的所有输出，并检查 LLM 是否倾向于正确答案，从而使结果更有利于基于 LLM 的方法。


TABLE S.1
表 S.1


SEMANTIC ROOM CLASSIFICATION RESULTS (HM3DSEM).
语义房间分类结果（HM3DSEM）。


<table><tr><td>Room Identification Method</td><td>Scene</td><td>Acc ${}_{\mathrm{{GT}}}\left\lbrack  \% \right\rbrack$</td><td>Accualid [%]</td></tr><tr><td rowspan="9">GPT3.5 w/ ground-truth object categories (privileged)</td><td>00824</td><td>70.00</td><td>90.00</td></tr><tr><td>00829</td><td>71.43</td><td>100.0</td></tr><tr><td>00842</td><td>61.54</td><td>69.23</td></tr><tr><td>00861</td><td>58.33</td><td>70.83</td></tr><tr><td>00862</td><td>50.00</td><td>72.22</td></tr><tr><td>00873</td><td>81.82</td><td>90.91</td></tr><tr><td>00877</td><td>69.23</td><td>76.92</td></tr><tr><td>00877</td><td>72.73</td><td>81.82</td></tr><tr><td>Overall</td><td>66.89</td><td>81.49</td></tr><tr><td rowspan="9">GPT3.5 w/ predicted object categories (unprivileged)</td><td>00824</td><td>30.00</td><td>40.00</td></tr><tr><td>00829</td><td>42.86</td><td>57.14</td></tr><tr><td>00842</td><td>38.46</td><td>38.46</td></tr><tr><td>00861</td><td>16.67</td><td>25.00</td></tr><tr><td>00862</td><td>19.44</td><td>25.00</td></tr><tr><td>00873</td><td>45.45</td><td>63.64</td></tr><tr><td>00877</td><td>07.69</td><td>30.77</td></tr><tr><td>00877</td><td>27.27</td><td>63.64</td></tr><tr><td>Overall</td><td>28.48</td><td>42.95</td></tr><tr><td rowspan="9">View embeddings (ours)</td><td>00824</td><td>80.00</td><td>90.00</td></tr><tr><td>00829</td><td>85.71</td><td>100.0</td></tr><tr><td>00842</td><td>69.23</td><td>76.92</td></tr><tr><td>00861</td><td>54.17</td><td>79.17</td></tr><tr><td>00862</td><td>63.89</td><td>83.33</td></tr><tr><td>00873</td><td>90.91</td><td>90.91</td></tr><tr><td>00877</td><td>61.54</td><td>61.54</td></tr><tr><td>00877</td><td>81.82</td><td>90.91</td></tr><tr><td>Overall</td><td>73.93</td><td>84.10</td></tr></table>
<table><tbody><tr><td>房间识别方法</td><td>场景</td><td>精度 ${}_{\mathrm{{GT}}}\left\lbrack  \% \right\rbrack$</td><td>精度 [%]</td></tr><tr><td rowspan="9">GPT3.5 搭配真实物体类别（特权）</td><td>00824</td><td>70.00</td><td>90.00</td></tr><tr><td>00829</td><td>71.43</td><td>100.0</td></tr><tr><td>00842</td><td>61.54</td><td>69.23</td></tr><tr><td>00861</td><td>58.33</td><td>70.83</td></tr><tr><td>00862</td><td>50.00</td><td>72.22</td></tr><tr><td>00873</td><td>81.82</td><td>90.91</td></tr><tr><td>00877</td><td>69.23</td><td>76.92</td></tr><tr><td>00877</td><td>72.73</td><td>81.82</td></tr><tr><td>整体</td><td>66.89</td><td>81.49</td></tr><tr><td rowspan="9">GPT3.5 搭配预测物体类别（非特权）</td><td>00824</td><td>30.00</td><td>40.00</td></tr><tr><td>00829</td><td>42.86</td><td>57.14</td></tr><tr><td>00842</td><td>38.46</td><td>38.46</td></tr><tr><td>00861</td><td>16.67</td><td>25.00</td></tr><tr><td>00862</td><td>19.44</td><td>25.00</td></tr><tr><td>00873</td><td>45.45</td><td>63.64</td></tr><tr><td>00877</td><td>07.69</td><td>30.77</td></tr><tr><td>00877</td><td>27.27</td><td>63.64</td></tr><tr><td>整体</td><td>28.48</td><td>42.95</td></tr><tr><td rowspan="9">视图嵌入（我们的）</td><td>00824</td><td>80.00</td><td>90.00</td></tr><tr><td>00829</td><td>85.71</td><td>100.0</td></tr><tr><td>00842</td><td>69.23</td><td>76.92</td></tr><tr><td>00861</td><td>54.17</td><td>79.17</td></tr><tr><td>00862</td><td>63.89</td><td>83.33</td></tr><tr><td>00873</td><td>90.91</td><td>90.91</td></tr><tr><td>00877</td><td>61.54</td><td>61.54</td></tr><tr><td>00877</td><td>81.82</td><td>90.91</td></tr><tr><td>整体</td><td>73.93</td><td>84.10</td></tr></tbody></table>


The table shows the room classification performance of our method (view embeddings) and two baselines (at the top) on HM3DSem. The baselines utilize GPT3.5 for labeling the rooms based on either ground-truth objects (masks) and categories or on predicted masks and categories. We consider two different evaluation criteria: ${\mathrm{{Acc}}}_{\mathrm{{GT}}}$ measures whether the exact text-wise room category was predicted while ${\mathrm{{Acc}}}_{\text{ valid }}$ measures correct room labels based on qualitative human evaluation.
表格展示了我们方法（视图嵌入）以及两个基线（位于顶部）在 HM3DSem 上的房间分类性能。基线使用 GPT3.5 依据两种方式为房间标注：一种是基于真实对象（mask）和类别，另一种是基于预测的 mask 和类别。我们考虑两种不同的评估标准：${\mathrm{{Acc}}}_{\mathrm{{GT}}}$ 用于衡量是否预测出了精确的逐文本房间类别；${\mathrm{{Acc}}}_{\text{ valid }}$ 则基于定性的人类评估来衡量正确的房间标签。


As presented in Tab. S.1, the view embedding method outperforms even the privileged baseline that relies on ground-truth object categories by $\sim  7\%$ wrt. the strict accuracy evaluation ( ${\mathrm{{Acc}}}_{\mathrm{{GT}}}$ ). We also observe a significant performance gap in terms of human evaluation, which is at 2.6%. There is only a single scene in which the privileged baseline outperforms our view embedding method (00877). The naïve baseline operating on predicted object categories is significantly outperformed, which is mostly due to over-segmentation and wrongly predicted top-1 object categories. Thus, we conclude that our method is robust and even outperforms privileged methods by a significant margin.
如表 S.1 所示，视图嵌入方法在严格准确率评估（ ${\mathrm{{Acc}}}_{\mathrm{{GT}}}$ ）方面，即使与依赖真实物体类别的特权基线相比，也表现更优，其提升来自 $\sim  7\%$。我们还观察到在人类评估方面存在显著的性能差距，为 2.6%。只有一个场景中，特权基线优于我们的视图嵌入方法（00877）。在使用预测物体类别的朴素基线上，该方法的表现明显更差，主要原因是过度分割以及错误预测的 top-1 物体类别。因此，我们得出结论：我们的方法具有稳健性，并且还能以显著幅度超过特权方法。


## D. Language-Grounded Navigation with Long Queries
## D. 面向长查询的语言引导导航


In order to support our proposed hierarchical segregation of the environment, we present another comparison with ConceptGraphs [22]. To do so, we compare the object retrieval from language queries performance to demonstrate the efficacy of hierarchically decomposing scenes. We draw this comparison by augmenting ConceptGraphs to also work with room and floor queries. For both HOV-SG as well as ConceptGraphs, we decompose the original query via GPT3.5 parsing as before. Using this, we obtain text variables stating the requested floor name, room name, and object name. Since the floor segmentation of HOV-SG consistently showed 100% accuracy, we directly provide ConceptGraphs with that information. Our augmentation of ConceptGraphs allows us to implicitly identify potential target rooms and objects: We compute the cosine similarity between the set of all object embeddings and the queried room text. Similarly, we compute the cosine similarity between the set of all objects and the queried object name. We combine these two similarities by taking the product of those scores per object to identify the most probable objects. This allows ConceptGraphs to draw answers at the floor level and room level. The remaining details of this evaluation are detailed in the main manuscript.
为支持我们提出的对环境进行层级化隔离的方法，我们再与 ConceptGraphs [22] 进行一次比较。为此，我们将语言查询中的目标检索性能作为依据，用以证明对场景进行层级化分解的有效性。我们通过扩展 ConceptGraphs，使其同时支持房间与楼层查询，从而进行这项对比。对于 HOV-SG 与 ConceptGraphs，我们仍如前所述，使用 GPT3.5 解析来分解原始查询。由此，我们得到若干文本变量，分别表示所请求的楼层名称、房间名称以及目标物体名称。由于 HOV-SG 的楼层分割始终显示为 100% 准确率，我们直接向 ConceptGraphs 提供该信息。对 ConceptGraphs 的这种扩展使其能够隐式识别潜在目标房间与物体：我们计算所有物体嵌入集合与所查询房间文本之间的余弦相似度。同样，我们计算所有物体与所查询物体名称之间的余弦相似度。我们通过对每个物体将这两项相似度得分相乘来进行组合，以找出最可能的物体。这样，ConceptGraphs 就可以在楼层层级与房间层级给出答案。本次评估的其余细节详见主论文手稿。


TABLE S.2
表 S.2


OBJECT RETRIEVAL FROM LANGUAGE QUERIES (HM3DSEM).
来自语言查询的目标检索（HM3DSEM）。


<table><tr><td>Query Type</td><td>Method</td><td>Scene</td><td>#Floors</td><td>#Regions</td><td>#Trials</td><td>${\mathrm{{SR}}}_{10}\left\lbrack  \% \right\rbrack$</td></tr><tr><td rowspan="18">(o, r, f)</td><td rowspan="9">ConceptGraphs</td><td>00824</td><td>1</td><td>10</td><td>33</td><td>33.33</td></tr><tr><td>00829</td><td>1</td><td>7</td><td>20</td><td>65.00</td></tr><tr><td>00843</td><td>2</td><td>13</td><td>26</td><td>03.85</td></tr><tr><td>00861</td><td>2</td><td>24</td><td>55</td><td>01.82</td></tr><tr><td>00862</td><td>3</td><td>36</td><td>90</td><td>21.11</td></tr><tr><td>00873</td><td>2</td><td>11</td><td>28</td><td>10.71</td></tr><tr><td>00877</td><td>2</td><td>13</td><td>32</td><td>09.38</td></tr><tr><td>00890</td><td>2</td><td>11</td><td>41</td><td>04.88</td></tr><tr><td>Overall</td><td>-</td><td>-</td><td>40.63</td><td>16.31</td></tr><tr><td rowspan="9">HOV-SG (ours) w/ OVSeg</td><td>00824</td><td>1</td><td>10</td><td>33</td><td>57.57</td></tr><tr><td>00829</td><td>1</td><td>7</td><td>20</td><td>45.00</td></tr><tr><td>00843</td><td>2</td><td>13</td><td>26</td><td>34.62</td></tr><tr><td>00861</td><td>2</td><td>24</td><td>55</td><td>25.45</td></tr><tr><td>00862</td><td>3</td><td>36</td><td>90</td><td>21.11</td></tr><tr><td>00873</td><td>2</td><td>11</td><td>28</td><td>14.29</td></tr><tr><td>00877</td><td>2</td><td>13</td><td>32</td><td>25.00</td></tr><tr><td>00890</td><td>2</td><td>11</td><td>41</td><td>21.95</td></tr><tr><td colspan="3">Overall</td><td>40.63</td><td>28.00</td></tr><tr><td rowspan="18">(o, r)</td><td rowspan="9">ConceptGraphs</td><td>00824</td><td>1</td><td>10</td><td>33</td><td>33.33</td></tr><tr><td>00829</td><td>1</td><td>7</td><td>20</td><td>65.00</td></tr><tr><td>00843</td><td>2</td><td>13</td><td>23</td><td>34.78</td></tr><tr><td>00861</td><td>2</td><td>24</td><td>46</td><td>19.57</td></tr><tr><td>00862</td><td>3</td><td>36</td><td>67</td><td>26.98</td></tr><tr><td>00873</td><td>2</td><td>11</td><td>25</td><td>30.00</td></tr><tr><td>00877</td><td>2</td><td>13</td><td>24</td><td>25.00</td></tr><tr><td>00890</td><td>2</td><td>11</td><td>41</td><td>21.95</td></tr><tr><td>Overall</td><td>-</td><td>-</td><td>34.88</td><td>29.26</td></tr><tr><td rowspan="9">HOV-SG (ours)</td><td>00824</td><td>1</td><td>10</td><td>33</td><td>57.58</td></tr><tr><td>00829</td><td>1</td><td>7</td><td>20</td><td>45.00</td></tr><tr><td>00843</td><td>2</td><td>13</td><td>23</td><td>39.13</td></tr><tr><td>00861</td><td>2</td><td>24</td><td>46</td><td>30.43</td></tr><tr><td>00862</td><td>3</td><td>36</td><td>67</td><td>20.63</td></tr><tr><td>00873</td><td>2</td><td>11</td><td>25</td><td>20.00</td></tr><tr><td>00877</td><td>2</td><td>13</td><td>24</td><td>33.33</td></tr><tr><td>00890</td><td>2</td><td>11</td><td>41</td><td>21.95</td></tr><tr><td>Overall</td><td>-</td><td>-</td><td>34.88</td><td>31.48</td></tr></table>
<table><tbody><tr><td>查询类型</td><td>方法</td><td>场景</td><td>#楼层</td><td>#区域</td><td>#试验次数</td><td>${\mathrm{{SR}}}_{10}\left\lbrack  \% \right\rbrack$</td></tr><tr><td rowspan="18">(o, r, f)</td><td rowspan="9">ConceptGraphs</td><td>00824</td><td>1</td><td>10</td><td>33</td><td>33.33</td></tr><tr><td>00829</td><td>1</td><td>7</td><td>20</td><td>65.00</td></tr><tr><td>00843</td><td>2</td><td>13</td><td>26</td><td>03.85</td></tr><tr><td>00861</td><td>2</td><td>24</td><td>55</td><td>01.82</td></tr><tr><td>00862</td><td>3</td><td>36</td><td>90</td><td>21.11</td></tr><tr><td>00873</td><td>2</td><td>11</td><td>28</td><td>10.71</td></tr><tr><td>00877</td><td>2</td><td>13</td><td>32</td><td>09.38</td></tr><tr><td>00890</td><td>2</td><td>11</td><td>41</td><td>04.88</td></tr><tr><td>总体</td><td>-</td><td>-</td><td>40.63</td><td>16.31</td></tr><tr><td rowspan="9">HOV-SG（我们的方法）配合 OVSeg</td><td>00824</td><td>1</td><td>10</td><td>33</td><td>57.57</td></tr><tr><td>00829</td><td>1</td><td>7</td><td>20</td><td>45.00</td></tr><tr><td>00843</td><td>2</td><td>13</td><td>26</td><td>34.62</td></tr><tr><td>00861</td><td>2</td><td>24</td><td>55</td><td>25.45</td></tr><tr><td>00862</td><td>3</td><td>36</td><td>90</td><td>21.11</td></tr><tr><td>00873</td><td>2</td><td>11</td><td>28</td><td>14.29</td></tr><tr><td>00877</td><td>2</td><td>13</td><td>32</td><td>25.00</td></tr><tr><td>00890</td><td>2</td><td>11</td><td>41</td><td>21.95</td></tr><tr><td colspan="3">总体</td><td>40.63</td><td>28.00</td></tr><tr><td rowspan="18">(o, r)</td><td rowspan="9">ConceptGraphs</td><td>00824</td><td>1</td><td>10</td><td>33</td><td>33.33</td></tr><tr><td>00829</td><td>1</td><td>7</td><td>20</td><td>65.00</td></tr><tr><td>00843</td><td>2</td><td>13</td><td>23</td><td>34.78</td></tr><tr><td>00861</td><td>2</td><td>24</td><td>46</td><td>19.57</td></tr><tr><td>00862</td><td>3</td><td>36</td><td>67</td><td>26.98</td></tr><tr><td>00873</td><td>2</td><td>11</td><td>25</td><td>30.00</td></tr><tr><td>00877</td><td>2</td><td>13</td><td>24</td><td>25.00</td></tr><tr><td>00890</td><td>2</td><td>11</td><td>41</td><td>21.95</td></tr><tr><td>总体</td><td>-</td><td>-</td><td>34.88</td><td>29.26</td></tr><tr><td rowspan="9">HOV-SG（我们的方法）</td><td>00824</td><td>1</td><td>10</td><td>33</td><td>57.58</td></tr><tr><td>00829</td><td>1</td><td>7</td><td>20</td><td>45.00</td></tr><tr><td>00843</td><td>2</td><td>13</td><td>23</td><td>39.13</td></tr><tr><td>00861</td><td>2</td><td>24</td><td>46</td><td>30.43</td></tr><tr><td>00862</td><td>3</td><td>36</td><td>67</td><td>20.63</td></tr><tr><td>00873</td><td>2</td><td>11</td><td>25</td><td>20.00</td></tr><tr><td>00877</td><td>2</td><td>13</td><td>24</td><td>33.33</td></tr><tr><td>00890</td><td>2</td><td>11</td><td>41</td><td>21.95</td></tr><tr><td>总体</td><td>-</td><td>-</td><td>34.88</td><td>31.48</td></tr></tbody></table>


Evaluation over 20 frequent distinct object categories in terms of the top-5 accuracy. A match is counted as a success when the IoU $> {0.1}$ between predicted object and ground truth. The floor and room counts refer to the ground-truth labels. The number of trials is lower for $\left( {\circ ,r}\right)$ compared to $\left( {\circ ,r,f}\right)$ because we observe a higher number of query duplicates whenever we drop the floor specification. The 20 categories evaluated are: picture, pillow, door, lamp, cabinet, book, chair; table, towel, plant, sink, stairs, bed, toilet, tv, desk, couch, flowerpot, nightstand, faucet.
在20个常见、彼此不同的物体类别上进行评估，采用top-5准确率。当预测物体与真实物体的IoU $> {0.1}$ 达到匹配标准时，计为成功。地面与房间的统计是指真实标签。与$\left( {\circ ,r,f}\right)$相比，当我们移除地面指定时，由于会观察到更多的查询重复，$\left( {\circ ,r}\right)$对应的试验次数更少。被评估的20个类别为：picture, pillow, door, lamp, cabinet, book, chair；table, towel, plant, sink, stairs, bed, toilet, tv, desk, couch, flowerpot, nightstand, faucet。


<img src="https://cdn.noedgeai.com/bo_d8j3o4491nqc738ucjg0_10.jpg?x=150&y=138&w=735&h=426&r=0"/>



Fig. S.6. Boston Dynamics Spot robot traversing a two-story office building with multiple types of rooms. The quadruped is equipped with an Azure Kinect RGB-D camera and a 3D LiDAR. We obtain accurate pose estimates from LiDAR-based odometry estimation.
图 S.6. 波士顿动力 Spot 机器人在一座有多种房间类型的双层办公楼中穿行。该四足机器人配备了 Azure Kinect RGB-D 相机和 3D LiDAR。我们通过基于 LiDAR 的里程计估计获得了准确的位姿估计。


The results in Tab. S. 2 regarding object-room-floor queries demonstrate a significant performance improvement of 11.69% when using HOV-SG compared to ConceptGraphs. We observe that ConceptGraphs struggles with larger scenes and under-segmentation of the produced maps, which often makes finding the object in question hard. Regarding the object-room queries, the drawbacks of ConceptGraphs are not as apparent because the search domain is significantly larger. Still, HOV-SG shows a 2.2% advantage over ConceptGraphs. In general, erroneous room segmentations produced by HOV-SG make finding the object in question hard, which remains subject to future work.
表 S.2 中关于物体-房间-地面查询的结果表明，与 ConceptGraphs 相比，使用 HOV-SG 的性能提升了11.69%。我们观察到，ConceptGraphs 在更大的场景中以及所生成地图的欠分割问题上表现困难，这通常会使得定位所关心的物体变得很难。就物体-房间查询而言，由于搜索范围显著更大，ConceptGraphs 的不足并不那么明显。尽管如此，HOV-SG 仍较 ConceptGraphs 具有2.2%的优势。总体而言，HOV-SG 产生的错误房间分割会使得定位所关心的物体变得困难，这仍有待未来工作解决。


## E. Graph Representation on HM3DSem
## E. HM3DSem 上的图表示


In the following, we also show the produced hierarchical 3D scene graphs on the set of 8 scenes we evaluated in Fig. S.7. Each distinct object is colored with a different color and the ground truth floor surface is underlayed for easier visibility. The blue nodes denote rooms and its links to the objects denote the object-room associations. The edges among the yellow nodes and the blue nodes show the association between rooms and floors. For clear visualization, we do not visualize the root node that connects multiple floors. We reject certain objects for visualization based on their top-1 predicted object category (out of 1624 categories). Any categories containing sub-strings of the following have not been visualized: wall, floor, ceiling, paneling, banner, overhang. All other predicted object categories are shown. Remarkably, this procedure removed the fair majority of ceilings, walls, etc., which confirms the accuracy of the top-1 predicted open-vocabulary object labels. Nonetheless, future work should address the problem of over- and under-segmentation in these maps. Coping with multiple overlapping masks produced during iterative mask merging is still an open question. While having multiple overlapping masks per point drives the recall in semantic retrieval, this does not produce visually appealing maps. In general, one could argue that depending on the language query at hand different concepts are requested. In case of a query such as "Find the sofa", one would like to obtain the mask that encloses the whole sofa. On the other hand, if the query comes in the form of "Find the cushion" (on the sofa), we would want to singulate the cushion in question. This however is difficult when the sofa is masked as one, which would then be considered under-segmentation. Thus, we envision maps that can hold multiple overlapping object masks that could represent various subconcepts. Essentially, this translates to an additional object hierarchy layer that decomposes objects into their parts.
下文中，我们还展示了在图 S.7 中评估的 8 个场景上的生成层级 3D 场景图。每个不同物体都用不同颜色标示，并叠加了真实地面表面以便更清晰地观察。蓝色节点表示房间，其与物体的连线表示物体-房间关联。黄色节点与蓝色节点之间的边表示房间与楼层之间的关联。为便于可视化，我们不显示连接多个楼层的根节点。我们依据部分对象的 top-1 预测类别（共 1624 类）将其排除在可视化之外。包含以下子字符串的类别不会被可视化：wall、floor、ceiling、paneling、banner、overhang。其余所有预测的对象类别均会显示。值得注意的是，这一过程去除了相当大多数的天花板、墙壁等，这验证了 top-1 预测的开放词汇对象标签的准确性。尽管如此，未来工作仍应解决这些地图中的过分割与欠分割问题。如何处理迭代式掩码合并过程中产生的多个重叠掩码仍是一个未解问题。虽然每个点对应多个重叠掩码有助于语义检索的召回率，但这不会生成美观的地图。一般而言，可以认为不同的语言查询会请求不同的概念。对于诸如“Find the sofa”这样的查询，人们希望得到包围整个沙发的掩码。另一方面，如果查询形式是“Find the cushion”（在沙发上），我们则希望将所指靠垫单独分离出来。然而，当沙发被作为一个整体掩码时，这就很难实现，而这会被视为欠分割。因此，我们设想的地图能够包含多个重叠的对象掩码，以表示各种子概念。本质上，这对应于增加一个对象层级，将对象分解为其部件。


TABLE S.3
表 S.3


REPRESENTATION SIZE (HM3DSEM)
表示大小（HM3DSEM）


<table><tr><td rowspan="2">Scene</td><td rowspan="2">Floor Number</td><td colspan="3">Size (MB)</td></tr><tr><td>VLMaps [7]</td><td>ConceptGraphs [22]</td><td>HOV-SG (ours)</td></tr><tr><td>00824</td><td>1</td><td>568</td><td>143</td><td>143</td></tr><tr><td>00829</td><td>1</td><td>407</td><td>110</td><td>99</td></tr><tr><td>00843</td><td>2</td><td>534</td><td>143</td><td>125</td></tr><tr><td>00861</td><td>2</td><td>943</td><td>255</td><td>225</td></tr><tr><td>00862</td><td>3</td><td>1808</td><td>474</td><td>479</td></tr><tr><td>00873</td><td>2</td><td>570</td><td>167</td><td>129</td></tr><tr><td>00877</td><td>2</td><td>556</td><td>154</td><td>131</td></tr><tr><td>00890</td><td>2</td><td>682</td><td>192</td><td>162</td></tr><tr><td>Sum</td><td>-</td><td>6068</td><td>1638</td><td>1493</td></tr></table>
<table><tbody><tr><td rowspan="2">场景</td><td rowspan="2">楼层数</td><td colspan="3">大小（MB）</td></tr><tr><td>VLMaps [7]</td><td>ConceptGraphs [22]</td><td>HOV-SG（我们的）</td></tr><tr><td>00824</td><td>1</td><td>568</td><td>143</td><td>143</td></tr><tr><td>00829</td><td>1</td><td>407</td><td>110</td><td>99</td></tr><tr><td>00843</td><td>2</td><td>534</td><td>143</td><td>125</td></tr><tr><td>00861</td><td>2</td><td>943</td><td>255</td><td>225</td></tr><tr><td>00862</td><td>3</td><td>1808</td><td>474</td><td>479</td></tr><tr><td>00873</td><td>2</td><td>570</td><td>167</td><td>129</td></tr><tr><td>00877</td><td>2</td><td>556</td><td>154</td><td>131</td></tr><tr><td>00890</td><td>2</td><td>682</td><td>192</td><td>162</td></tr><tr><td>总计</td><td>-</td><td>6068</td><td>1638</td><td>1493</td></tr></tbody></table>


Evaluation of representation size of HOV-SG compared to VLMaps and ConceptGraphs.
与 VLMaps 和 ConceptGraphs 相比，评估 HOV-SG 的表示规模。


## F. Real-World Environment
## F. 真实世界环境


In Fig. S.8, we present three real-world trials that were executed with a Boston Dynamics Spot quadrupedal robot, which allowed us to traverse multi-floor environments safely, see Fig.S.6. The trials are performed based on complex hierarchical language queries that specify the floor, the room, and the object to find. All hierarchical concepts relied on in these experiments are identified using our open-vocabulary HOV-SG pipeline. The top row in Fig. S. 8 shows the taken path (blue) from the start position (red) to the goal location (green). The following rows show the time-wise progression of the trial from top to bottom. The unique difficulty in these experiments is the typical office/lab environment with many similar rooms, which often produced similar room names. Having semantically varied rooms instead drastically simplifies these tasks. Nonetheless, as reported in the main manuscript, we reach real-world success rates of around 55%.
在图S.8中，我们展示了三个真实世界试验，这些试验由一台Boston Dynamics Spot四足机器人执行，使我们能够安全地穿越多层环境，见图S.6。这些试验基于复杂的层级语言查询，明确指定要寻找的楼层、房间和物体。这些实验所依赖的所有层级概念均通过我们的开放词汇HOV-SG流程识别。图S.8的第一行显示了从起始位置（红色）到目标位置（绿色）的行进路径（蓝色）。接下来的各行从上到下显示试验随时间的进展。这些实验的独特难点在于典型的办公/实验室环境中存在许多相似的房间，往往会产生相似的房间名称。相反，语义差异更大的房间会大大简化这些任务。尽管如此，正如主文中所报告的，我们在真实世界中的成功率约为55%。


## G. Representation Storage Overhead Evaluation
## G. 分层开放词汇 3D 场景图存储开销评估


A key advantage of HOV-SG is the compactness of the representation. We compare the sizes of VLMaps [7], ConceptGraphs [22], and HOV-SG created for the eight scenes in the Habitat Matterport 3D Semantic dataset and show the results in Table S.3. We adapt VLMaps to store LSeg features at 3D voxel locations. The backbone of the LSeg is ViT-B- 32, which has 512 dimensional features. ConceptGraphs and HOV-SG are using the ViT-H-14 CLIP backbones, which requires saving a 1024-dimension feature in the representation. VLMaps is optimized to only save features at voxels near object surfaces instead of saving redundant features at non-occupied voxels. Nonetheless, thanks to the compact graph structure, ConceptGraphs and HOV-SG are much smaller than their dense counterparts. HOV-SG even reduces as much as 75% of storage on average compared to VLMaps.
HOV-SG 的一项关键优势是表示紧凑。我们比较了 Habitat Matterport 3D Semantic 数据集中的八个场景所构建的 VLMaps [7]、ConceptGraphs [22] 和 HOV-SG 的大小，并将结果展示在表 S.3 中。我们将 VLMaps 调整为在 3D 体素位置存储 LSeg 特征。LSeg 的主干网络是 ViT-B-32，其特征维度为 512。ConceptGraphs 和 HOV-SG 使用 ViT-H-14 CLIP 主干网络，这需要在表示中保存 1024 维特征。VLMaps 经过优化，只在靠近物体表面的体素处保存特征，而不在未占据体素上保存冗余特征。尽管如此，得益于紧凑的图结构，ConceptGraphs 和 HOV-SG 仍比其稠密对应方法小得多。与 VLMaps 相比，HOV-SG 的存储开销平均甚至减少了多达 75%。


<img src="https://cdn.noedgeai.com/bo_d8j3o4491nqc738ucjg0_11.jpg?x=278&y=128&w=1229&h=1904&r=0"/>



Fig. S.7. We show a visualization of the hierarchical open-vocabulary scene graphs produced on HM3Dsem. To make the visualization more clear we do not show the root node connecting (multiple) floors. In addition, we underlay the ground-truth floor surface for easier visibility. We reject certain objects for visualization based on their top-1 predicted object category (out of 1624 categories). Any categories containing sub-strings of the following have not been visualized: wall, floor, ceiling, paneling, banner, overhang. All other predicted object categories are shown. Remarkably, this procedure removed the fair majority of ceilings, walls, etc., which confirms the accuracy of the top-1 predicted open-vocabulary object labels. Best viewed zoomed in.
图 S.7。我们展示了在 HM3Dsem 上生成的分层开放词汇场景图的可视化。为使可视化更清晰，我们不显示连接（多个）楼层的根节点。此外，我们叠加了真实的楼层表面以便更易观察。我们根据某些物体的 top-1 预测类别（共 1624 类）将其排除在可视化之外。凡类别名称包含以下子串者均未可视化：wall、floor、ceiling、paneling、banner、overhang。其余所有预测的物体类别均显示出来。值得注意的是，这一过程移除了绝大多数天花板、墙壁等，这验证了 top-1 预测开放词汇物体标签的准确性。放大后查看效果最佳。


<img src="https://cdn.noedgeai.com/bo_d8j3o4491nqc738ucjg0_12.jpg?x=191&y=109&w=1410&h=2021&r=0"/>



Fig. S.8. Real-World Object Navigation from Language Queries: We show a set of qualitative results of the real-world demonstration trials, which uses a Boston Dynamics Spot to allow for multi-floor traversals. The first row displays the observed scene and the taken path from the start (red) to the goal location (green). The following rows detail the time-wise progression (top-to-bottom).
图 S.8。基于语言查询的真实世界物体导航：我们展示了一组真实世界演示试验的定性结果，使用波士顿动力 Spot 实现多楼层穿行。第一行显示观察到的场景以及从起点（红色）到目标位置（绿色）所走的路径。后续各行按时间展示进展（自上而下）。