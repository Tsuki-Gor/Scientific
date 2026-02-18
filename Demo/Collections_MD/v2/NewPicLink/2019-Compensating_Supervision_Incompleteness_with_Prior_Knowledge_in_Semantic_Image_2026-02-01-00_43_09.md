# Compensating Supervision Incompleteness with Prior Knowledge in Semantic Image Interpretation
# 在语义图像解释中利用先验知识弥补监督不完备性


Ivan Donadello
Ivan Donadello


Fondazione Bruno Kessler
Fondazione Bruno Kessler


Trento, Italy
意大利，特伦托


donadello@fbk.eu



Luciano Serafini
Luciano Serafini


Fondazione Bruno Kessler
Fondazione Bruno Kessler


Trento, Italy
意大利，特伦托


serafini@fbk.eu



Abstract-Semantic Image Interpretation is the task of extracting a structured semantic description from images. This requires the detection of visual relationships: triples (subject, relation, object⟩ describing a semantic relation between a subject and an object. A pure supervised approach to visual relationship detection requires a complete and balanced training set for all the possible combinations of ⟨subject, relation, object⟩. However, such training sets are not available and would require a prohibitive human effort. This implies the ability of predicting triples which do not appear in the training set. This problem is called zero-shot learning. State-of-the-art approaches to zero-shot learning exploit similarities among relationships in the training set or external linguistic knowledge. In this paper, we perform zero-shot learning by using Logic Tensor Networks, a novel Statistical Relational Learning framework that exploits both the similarities with other seen relationships and background knowledge, expressed with logical constraints between subjects, relations and objects. The experiments on the Visual Relationship Dataset show that the use of logical constraints outperforms the current methods. This implies that background knowledge can be used to alleviate the incompleteness of training sets.
摘要—语义图像解释是从图像中提取结构化语义描述的任务。这需要检测视觉关系：即描述主体与客体之间语义关系的三元组 ⟨主体, 关系, 客体⟩。纯监督的视觉关系检测方法需要针对 ⟨主体, 关系, 客体⟩ 的所有可能组合提供完整且平衡的训练集。然而，此类训练集难以获取，且需要极大的人力投入。这意味着系统必须具备预测训练集中未出现三元组的能力。该问题被称为零样本学习。目前的零样本学习方法主要利用训练集中各关系间的相似性或外部语言知识。在本文中，我们通过逻辑张量网络（一种新型统计关系学习框架）进行零样本学习，该框架同时利用了与其他已知关系的相似性以及通过主体、关系和客体间的逻辑约束所表达的背景知识。在视觉关系数据集（Visual Relationship Dataset）上的实验表明，使用逻辑约束的效果优于当前方法。这说明背景知识可以用于缓解训练集的不完备性。


## I. INTRODUCTION
## I. 引言


Semantic Image Interpretation (SII) [20] concerns the automatic extraction of high-level information about the content of a visual scene. This information regards the objects in the scene, their attributes and the relations among them. Formally, SII extracts the so-called scene graph [14] from a picture: the labelled nodes refer to objects in the scene and their attributes, the labelled edges regard the relations between the corresponding nodes. SII enables important applications on an image content that a coarser image analysis (e.g., the object detection) does not allow. For example, the visual question answering (answering to natural language questions about the image content), the image captioning (generating a natural language sentence describing the image content), the complex image querying (retrieving images using structured queries about the image content) or the robot interaction (different configurations of objects allow different actions for a robot). The visual relationship detection (VRD) is a special instance of scene graph construction. Indeed, a visual relationship is a triple ⟨subject, predicate, object⟩ where the subject and the object are the labels (or semantic classes) of two bounding boxes in the image. The predicate is the label regarding the relationship between the two bounding boxes. The construction of a scene graph from visual relationships is pretty forward. The subject and the object are nodes in the graph with the corresponding labels, the predicate is a labelled edge from the subject node to the object node.
语义图像解释 (SII) [20] 涉及从视觉场景内容中自动提取高层信息。这些信息包括场景中的物体、它们的属性以及彼此间的关系。正式地讲，SII 从图像中提取所谓的场景图 [14]：标注的节点指代场景中的物体及其属性，标注的边则表示对应节点间的关系。SII 能够实现更粗粒度的图像分析（如目标检测）无法完成的图像内容重要应用。例如：视觉问答（回答关于图像内容的自然语言问题）、图像描述生成（生成描述图像内容的自然语言句子）、复杂图像查询（使用关于图像内容的结构化查询来检索图像）或机器人交互（物体的不同配置决定了机器人的不同动作）。视觉关系检测 (VRD) 是场景图构建的一个特例。事实上，视觉关系是一个三元组 ⟨主体, 谓词, 客体⟩，其中主体和客体是图像中两个边界框的标签（或语义类别）。谓词是关于这两个边界框之间关系的标签。从视觉关系构建场景图非常直接：主体和客体是图中带有对应标签的节点，谓词是从主体节点指向客体节点的标注边。


Visual relationships are mainly detected with supervised learning techniques [31]. These require large training sets of images annotated with bounding boxes and relationships [14], [19]. However, a complete and detailed annotation is not possible due to the high human effort of the annotators. For example, a person riding a horse can be annotated with the relations on, ride (person subject) or below, carry (horse subject). Thus, many types of relationships are not in the training set but can appear in the test set. The task of predicting visual relationships with never seen instances in the training phase is called zero-shot learning [16]. This can be achieved by exploiting the similarity with the triples in the training set or using a high-level description of the relationship. For example, the fact that people ride elephants can be derived from a certain similarity between elephants and horses and the (known) fact that people can ride horses. This is closer to human learning with respect to supervised learning. Indeed, humans are able to both generalize from seen or similar examples and to use their background knowledge to identifying never seen relationships [16]. This background knowledge can be linguistic information about the subject/object and predicate occurrences [19], [31], [33] or logical constraints [10], [28]. Logical knowledge is very expressive and it allows us to explicitly state relations between subjects/objects and predicates. For example, the formula $\forall x,y\left( {\operatorname{ride}\left( {x,y}\right)  \rightarrow  \text{ elephant }\left( y\right)  \vee  \text{ horse }\left( y\right) }\right)$ states that the objects of the riding relations are elephants and horses. Other formulas can state that horses and elephants cannot ride and this avoids wrong predictions.
视觉关系主要通过监督学习技术 [31] 进行检测。这需要大量带有边界框和关系标注的图像训练集 [14], [19]。然而，由于标注者的人力成本极高，实现完整且详尽的标注是不可能的。例如，一个人骑马的场景可以被标注为 on、ride（以人为主体）或 below、carry（以马为主体）。因此，许多类型的关系并未包含在训练集中，但可能出现在测试集中。在训练阶段预测从未见过的实例的视觉关系任务被称为零样本学习 [16]。这可以通过利用与训练集中三元组的相似性或使用关系的高层描述来实现。例如，“人骑大象”这一事实可以根据大象与马之间的某种相似性，以及“人可以骑马”这一（已知）事实推导出来。相对于监督学习，这更接近人类的学习方式。事实上，人类既能从见过或相似的例子中进行泛化，也能利用背景知识来识别从未见过的关系 [16]。这种背景知识可以是关于主体/客体与谓词出现情况的语言信息 [19], [31], [33]，也可以是逻辑约束 [10], [28]。逻辑知识具有很强的表达力，允许我们显式地陈述主体/客体与谓词之间的关系。例如，公式 $\forall x,y\left( {\operatorname{ride}\left( {x,y}\right)  \rightarrow  \text{ elephant }\left( y\right)  \vee  \text{ horse }\left( y\right) }\right)$ 规定了骑行关系的客体是大象和马。其他公式可以规定马和大象不能执行骑行行为，从而避免错误的预测。


In this paper, we address the zero-shot learning problem by using Logic Tensor Networks (LTNs) [27] for the detection of unseen visual relationships. LTNs is a Statistical Relational Learning framework that learns from relational data (exploiting the similarities with already seen triples) in presence of logical constraints. The results on the Visual Relationship Dataset show that the joint use of logical knowledge and data outperforms the state-of-the-art approaches based on data and/or linguistic knowledge. These promising results show that logical knowledge is able to counterbalance the incompleteness of the datasets due to the high annotation effort: this is a significant contribution to the zero-shot learning. LTNs have already been exploited for SII [10], [28]. However, these works are preliminary as they focus only on the part-whole relation (PASCAL-PART dataset [7]). We extend these works with the following contributions:
在本文中，我们通过使用逻辑张量网络（LTNs）[27]来解决零样本学习问题，以检测未见的视觉关系。LTNs是一个统计关系学习框架，它在存在逻辑约束的情况下从关系数据中学习（利用与已见三元组的相似性）。视觉关系数据集上的结果表明，逻辑知识和数据的联合使用优于基于数据和/或语言知识的现有最先进方法。这些有希望的结果表明，逻辑知识能够弥补由于高标注工作量导致的数据集不完整性：这是对零样本学习的重大贡献。LTNs已经被用于SII [10]，[28]。然而，这些工作是初步的，因为它们仅关注部分 - 整体关系（PASCAL - PART数据集[7]）。我们通过以下贡献扩展了这些工作：


- We conduct the experiments on the the Visual Relationship Dataset (VRD) [19], a more challenging dataset that contains 70 binary predicates.
- 我们在视觉关系数据集（VRD）[19]上进行实验，这是一个更具挑战性的数据集，包含 70 个二元谓词。


- We introduce new additional features for pairs of bounding-boxes that capture the geometric relations between bounding boxes. These features are necessary as they drastically improve the performance.
- 我们为边界框对引入了新的附加特征，用于捕捉边界框之间的几何关系。这些特征至关重要，因为它们显著提升了性能。


- We perform a theoretical analysis of the drawbacks of using a loss function based on t-norms. Therefore, we introduce a new loss function based on the harmonic mean among the truth values of the axioms in the background knowledge.
- 我们对基于 t-范数的损失函数的局限性进行了理论分析。因此，我们引入了一种新的损失函数，该函数基于背景知识中公理真值的调和平均值。


The effectiveness of the new features and the new mean-based loss function is proved with some ablation studies.
新特征和新的基于均值的损失函数的有效性通过消融研究得到了证实。


## II. RELATED WORK
## II. 相关工作


The detection of visual relationships in images is tightly connected to Semantic Image Interpretation (SII) [20]. SII extracts a graph [14] that describes an image semantic content: nodes are objects and attributes, edges are relations between nodes. In [20] the SII graph (i.e., the visual relationships) is generated with deductive reasoning and the low-level image features are encoded in a knowledge base with some logical axioms. In [22] abductive reasoning is used. However, writing axioms that map features into concepts/relations or defining the rules of abduction requires a high engineering effort, and dealing with the noise of the object detectors could be problematic. Fuzzy logic [12] deals with this noise. In [1], [13] a fuzzy logic ontology of spatial relations and an algorithm (based on morphological and logical reasoning) for building SII graphs are proposed. These works are limited to spatial relations. In [29] the SII graph is built with an iterative message passing algorithm where the information about the objects maximizes the likelihood of the relationships and vice versa. In [32] a combination of Long Short-Term Memories (LSTMs) is exploited. The first LSTM encodes the context given by the detected bounding boxes. This context is used for classifying both objects and relations with two different LSTMs without external knowledge. Here, only one edge is allowed between two nodes (as in [29], [31]). This assumption of mutual exclusivity between predicates does not hold in real life, e.g., if a person rides a horse then is on the horse. LTNs instead allow multiple edges between nodes. In [9] a clustering algorithm integrates low-level and semantic features to group parts belonging to the same whole object or event. Logical reasoning is applied for consistency checking. However, the method is tailored only on the part-whole relation.
图像中视觉关系的检测与语义图像解释 (SII) [20] 紧密相关。SII 提取描述图像语义内容的图 [14]：节点是对象和属性，边是节点间的关系。在 [20] 中，SII 图（即视觉关系）通过演绎推理生成，低级图像特征通过逻辑公理编码在知识库中。在 [22] 中使用了溯因推理。然而，编写将特征映射到概念/关系或定义溯因规则的公理需要极高的工程成本，且处理对象检测器的噪声可能存在困难。模糊逻辑 [12] 可处理这种噪声。在 [1] 和 [13] 中，提出了一种空间关系的模糊逻辑本体和一种（基于形态学和逻辑推理的）构建 SII 图的算法。这些工作局限于空间关系。在 [29] 中，SII 图是通过迭代消息传递算法构建的，其中对象信息使关系的可能性最大化，反之亦然。在 [32] 中，利用了长短期记忆网络 (LSTMs) 的组合。第一个 LSTM 编码由检测到的边界框给出的上下文。该上下文被两个不同的 LSTM 用于在没有外部知识的情况下对对象和关系进行分类。这里，两个节点之间只允许有一条边（如 [29]、[31]）。这种谓词互斥的假设在现实生活中并不成立，例如，如果一个人骑马，那么他就在马上。相比之下，LTN 允许节点之间存在多条边。在 [9] 中，一种聚类算法整合了低级和语义特征，以对属于同一整体对象或事件的部分进行分组。逻辑推理被应用于一致性检查。然而，该方法仅针对部分-整体关系量身定制。


Other methods start from a fully connected graph whose nodes and edges need to be labelled or discarded according to an energy minimization function. In [15] the graph is encoded with a Conditional Random Field (CRF) and potentials are defined by combining the object detection score with geometric relations between objects and text priors on the types of objects. Also in [7] the scene graph is encoded with a CRF and the work leverages the part-whole relation to improve the object detection. These works do not consider logical knowledge. In [6] the energy function combines visual information of the objects with logical constraints. However, this integration is hand-crafted and thus difficult to extend to other types of constraints.
其他方法从全连接图开始，根据能量最小化函数对节点和边进行标记或舍弃。在 [15] 中，该图由条件随机场 (CRF) 编码，并通过结合目标检测评分、目标间的几何关系以及目标类型的文本先验来定义势函数。同样，在 [7] 中，场景图采用 CRF 编码，并利用部分-整体关系来改进目标检测。这些工作未考虑逻辑知识。在 [6] 中，能量函数结合了目标的视觉信息与逻辑约束。然而，这种集成是手工构建的，因此难以推广到其他类型的约束。


A visual phrase [25] is the prototype of a visual relationship. Here a single bounding box contains both subject and object. However, training an object detector for every possible triple affects the scalability. The visual semantic role labelling [11] is a generalization of detecting visual relationships. This task generates a set of tuples, such as: $\left\langle  {\text{ predicate },\left\{  {\left\langle  {{\text{ role }}_{1},{\text{ label }}_{1}}\right\rangle  ,\ldots ,\left\langle  {{\text{ role }}_{N},{\text{ label }}_{N}}\right\rangle  }\right\}  }\right\rangle$ , where the roles are entities involved in the predicate, such as, subject, object, tool or place. However, the work is preliminary and limits the role of subject only to people.
视觉短语 [25] 是视觉关系的原型。在这里，单个边界框同时包含主体和客体。然而，为每种可能的元组训练目标检测器会影响可扩展性。视觉语义角色标注 [11] 是视觉关系检测的泛化。该任务生成一组元组，例如：$\left\langle  {\text{ predicate },\left\{  {\left\langle  {{\text{ role }}_{1},{\text{ label }}_{1}}\right\rangle  ,\ldots ,\left\langle  {{\text{ role }}_{N},{\text{ label }}_{N}}\right\rangle  }\right\}  }\right\rangle$，其中角色是谓词涉及的实体，如主体、客体、工具或地点。然而，该工作尚处初步阶段，且将主体角色仅限于人。


Other works exploit deep learning. In [8] the visual relationships are detected with a Deep Relational Network that exploits the statistical dependencies between relationships and the subjects/objects. In [18] a deep reinforcement learning framework to detect relationships and attributes is used. In [17] a message passing algorithm is developed to share information about the subject, object and predicate among neural networks. In [33] the visual relationship is the translating vector of the subject towards the object in an embedding space. In [30] an end-to-end system exploits the interaction of visual and geometric features of the subject, object and predicate. The end-to-end system in [34] exploits weakly supervised learning (i.e., the supervision is at image level). LTNs exploit the combination of the visual/geometric features of the subject/object with additional background knowledge.
其他工作利用了深度学习。在 [8] 中，视觉关系通过深度关系网络检测，该网络利用了关系与主/客体之间的统计依赖性。在 [18] 中，使用了一个深度强化学习框架来检测关系和属性。在 [17] 中，开发了一种消息传递算法，用于在神经网络间共享关于主体、客体和谓词的信息。在 [33] 中，视觉关系是嵌入空间中主体向客体偏移的平移向量。在 [30] 中，一个端到端系统利用了主体、客体和谓词的视觉与几何特征交互。 [34] 中的端到端系统利用了弱监督学习（即监督发生在图像层面）。LTNs 利用主/客体的视觉/几何特征与额外背景知识的结合。


Background knowledge is also exploited in a joint embedding with visual knowledge. In [24] the exploited logical constraints are implication, mutual exclusivity and type-of. In [19] the background knowledge is a word embedding of the subject/object labels. The visual knowledge consists in the features of the union of the subject and object bounding boxes. In [2] the background knowledge is statistical information (learnt with statistical link prediction methods [21]) about the training set triples. Contextual information between objects is used also in [23], [35] with different learning methods. In [31] the background knowledge (from the training set and Wikipedia) is a probability distribution of a relationship given the subject/object. This knowledge drives the learning of visual relationships. These works do not exploit any type of logical constraints as LTNs do.
背景知识也被利用于与视觉知识的联合嵌入中。在 [24] 中，利用的逻辑约束包括蕴含、互斥和类型。在 [19] 中，背景知识是主/客体标签的词嵌入。视觉知识由主体和客体边界框并集的特征组成。在 [2] 中，背景知识是关于训练集元组的统计信息（通过统计链路预测方法 [21] 学习）。[23] 和 [35] 也使用不同的学习方法利用了物体间的上下文信息。在 [31] 中，背景知识（来自训练集和维基百科）是给定主/客体时关系的概率分布。这种知识引导了视觉关系的学习。这些工作没有像 LTNs 那样利用任何类型的逻辑约束。


## III. LOGIC TENSOR NETWORKS
## III. 逻辑张量网络


In the following we describe the basic notions of Logic Tensor Networks (LTNs) [27], whereas in Section IV we present the novel contributions of our work evaluated in Section V. LTNs are a statistical relational learning framework that combine Neural Networks with logical constraints. LTNs adopt the syntax of a First-Order predicate language $\mathcal{P}\mathcal{L}$ whose signature is composed of two disjoint sets $\mathcal{C}$ and $\mathcal{P}$ denoting constants and predicate symbols, respectively. We do not present LTNs function symbols as they are not strictly necessary for the detection of visual relationships. $\mathcal{P}\mathcal{L}$ allows LTNs to express visual relationships and a priori knowledge about a domain. E.g., the visual relationship ⟨person, ride, horse⟩ is expressed with the atomic formulas $\operatorname{Person}\left( {p}_{1}\right)$ ,Horse $\left( {h}_{1}\right)$ and ride $\left( {{p}_{1},{h}_{1}}\right)$ . Common knowledge is expressed through logical constraints,e.g., $\forall x,y\left( {\operatorname{ride}\left( {x,y}\right)  \rightarrow  \neg \operatorname{Dog}\left( x\right) }\right)$ states that dogs do not ride.
下文描述逻辑张量网络（LTNs）[27] 的基本概念，第四节介绍我们在第五节中评估的新贡献。LTNs 是一种结合了神经网络与逻辑约束的统计关系学习框架。LTNs 采用一阶谓词语言 $\mathcal{P}\mathcal{L}$ 的语法，其签名由两个不相交的集合 $\mathcal{C}$ 和 $\mathcal{P}$ 组成，分别表示常量和谓词符号。由于检测视觉关系并非严格需要，我们不介绍 LTNs 函数符号。$\mathcal{P}\mathcal{L}$ 允许 LTNs 表达视觉关系和关于领域的先验知识。例如，视觉关系 ⟨person, ride, horse⟩ 由原子公式 $\operatorname{Person}\left( {p}_{1}\right)$、Horse $\left( {h}_{1}\right)$ 和 ride $\left( {{p}_{1},{h}_{1}}\right)$ 表达。常识通过逻辑约束表达，例如 $\forall x,y\left( {\operatorname{ride}\left( {x,y}\right)  \rightarrow  \neg \operatorname{Dog}\left( x\right) }\right)$ 表明狗不会骑行。


LTNs semantics deviates from the abstract semantics of Predicate Fuzzy Logic [12] towards a concrete semantics. Indeed,the interpretation domain is a subset of ${\mathbb{R}}^{n}$ ,i.e., constant symbols and closed terms are associated with a $n$ - dimensional real vector. This vector encodes $n$ numerical features of an object, such as, the confidence score of an object detector, the bounding box coordinates, local features, etc. Predicate symbols are interpreted as functions on real vectors to $\left\lbrack  {0,1}\right\rbrack$ . The interpretation of a formula is the degree of truth for that formula: higher values mean higher degrees of truth. LTNs use the term grounding as synonym of logical interpretation in a "real" world. A grounding has to capture the latent correlation between the features of objects and their categorical/relational properties. Let $\alpha \left( s\right)$ denote the arity of a predicate symbol $s$ .
LTNs 的语义从谓词模糊逻辑 [12] 的抽象语义偏向于具体语义。事实上，解释域是 ${\mathbb{R}}^{n}$ 的子集，即常量符号和闭合项与一个 $n$ 维实数向量相关联。该向量编码了物体的 $n$ 个数值特征，例如目标检测器的置信度得分、边界框坐标、局部特征等。谓词符号被解释为实数向量到 $\left\lbrack  {0,1}\right\rbrack$ 的函数。公式的解释是该公式的真值度：数值越高表示真值度越高。LTNs 使用“grounding”（落地）作为逻辑在“真实”世界中解释的同义词。一个 grounding 必须捕捉物体特征与其类别/关系属性之间的潜在关联。令 $\alpha \left( s\right)$ 表示谓词符号 $s$ 的元数。


Definition 1: An $n$ -grounding $\mathcal{G}$ (with $n \in  \mathbb{N},n > 0$ ),or simply grounding,for a First-Order Language $\mathcal{P}\mathcal{L}$ is a function from the signature of $\mathcal{P}\mathcal{L}$ that satisfies the conditions:
定义 1：一阶语言 $\mathcal{P}\mathcal{L}$ 的 $n$ -grounding $\mathcal{G}$ （其中 $n \in  \mathbb{N},n > 0$ ），或简称 grounding，是从 $\mathcal{P}\mathcal{L}$ 的签名出发并满足以下条件的函数：


1) $\mathcal{G}\left( c\right)  \in  {\mathbb{R}}^{n}$ ,for every $c \in  \mathcal{C}$ ;
1) $\mathcal{G}\left( c\right)  \in  {\mathbb{R}}^{n}$ ，对于每个 $c \in  \mathcal{C}$ ；


2) $\mathcal{G}\left( P\right)  \in  {\mathbb{R}}^{n \cdot  \alpha \left( P\right) } \rightarrow  \left\lbrack  {0,1}\right\rbrack$ ,for every $P \in  \mathcal{P}$ .
2) $\mathcal{G}\left( P\right)  \in  {\mathbb{R}}^{n \cdot  \alpha \left( P\right) } \rightarrow  \left\lbrack  {0,1}\right\rbrack$ ，对于每个 $P \in  \mathcal{P}$ 。


Given a grounding $\mathcal{G}$ and let $\operatorname{term}\left( \mathcal{{PL}}\right)  = \left\{  {{t}_{1},{t}_{2},{t}_{3},\ldots }\right\}$ be the set of closed terms of $\mathcal{P}\mathcal{L}$ ,the semantics of atomic formulas is inductively defined as follows:
给定 grounding $\mathcal{G}$ 且令 $\operatorname{term}\left( \mathcal{{PL}}\right)  = \left\{  {{t}_{1},{t}_{2},{t}_{3},\ldots }\right\}$ 为 $\mathcal{P}\mathcal{L}$ 的闭合项集，原子公式的语义递归定义如下：


$$
\mathcal{G}\left( {P\left( {{t}_{1},\ldots ,{t}_{m}}\right) }\right)  = \mathcal{G}\left( P\right) \left( {\mathcal{G}\left( {t}_{1}\right) ,\ldots ,\mathcal{G}\left( {t}_{m}\right) }\right) . \tag{1}
$$



The semantics for non-atomic formulas is defined according to t-norms functions used in Fuzzy Logic. If we take, for example, the Łukasiewicz t-norm, we have:
非原子公式的语义根据模糊逻辑中使用的 t-norm 函数定义。例如，若采用 Łukasiewicz t-norm，则有：


$$
\mathcal{G}\left( {\phi  \rightarrow  \psi }\right)  = \min \left( {1,1 - \mathcal{G}\left( \phi \right)  + \mathcal{G}\left( \psi \right) }\right)
$$



$$
\mathcal{G}\left( {\phi  \land  \psi }\right)  = \max \left( {0,\mathcal{G}\left( \phi \right)  + \mathcal{G}\left( \psi \right)  - 1}\right)
$$



$$
\mathcal{G}\left( {\phi  \vee  \psi }\right)  = \min \left( {1,\mathcal{G}\left( \phi \right)  + \mathcal{G}\left( \psi \right) }\right)
$$



$$
\mathcal{G}\left( {\neg \phi }\right)  = 1 - \mathcal{G}\left( \phi \right) \tag{2}
$$



The semantics for quantifiers differs from the semantics of standard Fuzzy Logic. Indeed,the interpretation of $\forall$ leads the definition:
量词的语义不同于标准模糊逻辑的语义。事实上，对 $\forall$ 的解释引出了如下定义：


$$
\mathcal{G}\left( {\forall {x\phi }\left( x\right) }\right)  = \inf \{ \mathcal{G}\left( {\phi \left( t\right) }\right)  \mid  t \in  \operatorname{term}\left( {\mathcal{P}\mathcal{L}}\right) \} . \tag{3}
$$



This definition does not tolerate exceptions. E.g., the presence of a dog riding a horse in a circus would falsify the formula that dogs do not ride. LTNs handle these outliers giving a higher truth-value to the formula $\forall {x\phi }\left( x\right)$ if many examples satisfy $\phi \left( x\right)$ . This is in the spirit of SII as in a picture (due to occlusions or unexpected situations) some common logical constraints are not always respected.
该定义不容许异常。例如，马戏团中出现骑马的狗会使“狗不骑马”这一公式证伪。LTNs 通过处理这些离群值，在有许多样本满足 $\phi \left( x\right)$ 时，赋予公式 $\forall {x\phi }\left( x\right)$ 更高的真值。这符合 SII 的精神，因为在图像中（由于遮挡或意外情况），一些常见的逻辑约束并不总是被遵守。


Definition 2: Let ${\operatorname{mean}}_{p}\left( {{x}_{1},\ldots ,{x}_{d}}\right)  = {\left( \frac{1}{d}\mathop{\sum }\limits_{{i = 1}}^{d}{x}_{i}^{p}\right) }^{\frac{1}{p}}$ , with ${p}^{1} \in  \mathbb{Z},d \in  \mathbb{N}$ ,the grounding for $\forall {x\phi }\left( x\right)$ is
定义 2：令 ${\operatorname{mean}}_{p}\left( {{x}_{1},\ldots ,{x}_{d}}\right)  = {\left( \frac{1}{d}\mathop{\sum }\limits_{{i = 1}}^{d}{x}_{i}^{p}\right) }^{\frac{1}{p}}$ ，其中 ${p}^{1} \in  \mathbb{Z},d \in  \mathbb{N}$ ，$\forall {x\phi }\left( x\right)$ 的 grounding 为


$$
\mathcal{G}\left( {\forall {x\phi }\left( x\right) }\right)  =
$$



$$
\mathop{\lim }\limits_{{d \rightarrow  \left| {\operatorname{term}\left( \mathcal{{PL}}\right) }\right| }}{\operatorname{mean}}_{p}\left( {\mathcal{G}\left( {\phi \left( {t}_{1}\right) }\right) ,\ldots ,\mathcal{G}\left( {\phi \left( {t}_{d}\right) }\right) }\right) . \tag{4}
$$



The grounding of a quantified formula $\forall {x\phi }\left( x\right)$ is the mean of the $d$ groundings of the quantifier-free formula $\phi \left( x\right)$ .
量化公式 $\forall {x\phi }\left( x\right)$ 的 grounding 是无量词公式 $\phi \left( x\right)$ 的 $d$ 个 grounding 的均值。


A suitable function for a grounding should preserve some form of regularity. Let $b \in  \mathcal{C}$ refer to a bounding box constant containing a horse. Let $\mathbf{v} = \mathcal{G}\left( b\right)$ be its feature vector,then it holds that $\mathcal{G}\left( \text{ Horse }\right) \left( \mathbf{v}\right)  \approx  1$ . Moreover,for every bounding box with feature vector ${\mathbf{v}}^{\prime }$ similar to $\mathbf{v},\mathcal{G}\left( \text{ Horse }\right) \left( {\mathbf{v}}^{\prime }\right)  \approx  1$ holds. These functions are learnt from data ${}^{2}$ by tweaking their inner parameters in a training process. The grounding for predicate symbols is a generalization of a neural tensor network: an effective architecture for relational learning [21]. Let ${b}_{1},\ldots ,{b}_{m} \in  \mathcal{C}$ with feature vectors ${\mathbf{v}}_{i} = \mathcal{G}\left( {b}_{i}\right)  \in  {\mathbb{R}}^{n}$ , with $i = 1\ldots m$ ,and $\mathbf{v} = \left\langle  {{\mathbf{v}}_{1};\ldots ;{\mathbf{v}}_{m}}\right\rangle$ is a ${mn}$ -ary vector given by the vertical stacking of each vector ${\mathbf{v}}_{i}$ . The grounding $\mathcal{G}\left( P\right)$ of an $m$ -ary predicate $P\left( {{b}_{1},\ldots ,{b}_{m}}\right)$ is:
合适的实例化函数应保持某种形式的正则性。设 $b \in  \mathcal{C}$ 为包含马的边界框常量。设 $\mathbf{v} = \mathcal{G}\left( b\right)$ 为其特征向量，则 $\mathcal{G}\left( \text{ Horse }\right) \left( \mathbf{v}\right)  \approx  1$ 成立。此外，对于每个特征向量 ${\mathbf{v}}^{\prime }$ 与 $\mathbf{v},\mathcal{G}\left( \text{ Horse }\right) \left( {\mathbf{v}}^{\prime }\right)  \approx  1$ 相似的边界框也成立。这些函数通过训练过程中调整内部参数从数据 ${}^{2}$ 中学习。谓词符号的实例化是神经张量网络的推广：一种有效的关系学习架构 [21]。设 ${b}_{1},\ldots ,{b}_{m} \in  \mathcal{C}$ 的特征向量为 ${\mathbf{v}}_{i} = \mathcal{G}\left( {b}_{i}\right)  \in  {\mathbb{R}}^{n}$ ，其中 $i = 1\ldots m$ ，且 $\mathbf{v} = \left\langle  {{\mathbf{v}}_{1};\ldots ;{\mathbf{v}}_{m}}\right\rangle$ 是由每个向量 ${\mathbf{v}}_{i}$ 垂直堆叠构成的 ${mn}$ 元向量。$m$ 元谓词 $P\left( {{b}_{1},\ldots ,{b}_{m}}\right)$ 的实例化 $\mathcal{G}\left( P\right)$ 为：


$$
\mathcal{G}\left( P\right) \left( \mathbf{v}\right)  = \sigma \left( {{u}_{P}^{\top }\tanh \left( {{\mathbf{v}}^{\top }{W}_{P}^{\left\lbrack  1 : k\right\rbrack  }\mathbf{v} + {V}_{P}\mathbf{v} + {b}_{P}}\right) }\right) \tag{5}
$$



with $\sigma$ the sigmoid function. The parameters for $P$ are: ${u}_{P} \in \; {\mathbb{R}}^{k}$ ,a 3-D tensor ${W}_{P}^{\left\lbrack  1 : k\right\rbrack  } \in  {\mathbb{R}}^{k \times  {mn} \times  {mn}},{V}_{P} \in  {\mathbb{R}}^{k \times  {mn}}$ and ${b}_{P} \in  {\mathbb{R}}^{k}$ . The parameter ${u}_{P}$ computes a linear combination of the quadratic features returned by the tensor product. With Equations (1) and (5) the grounding of a complex LTNs formula can be computed by first computing the groundings of the closed terms and the atomic formulas contained in the complex formula. Then, these groundings are combined using a specific t-norm, see Equation (2).
其中 $\sigma$ 为 sigmoid 函数。$P$ 的参数为：${u}_{P} \in \; {\mathbb{R}}^{k}$ 、3D 张量 ${W}_{P}^{\left\lbrack  1 : k\right\rbrack  } \in  {\mathbb{R}}^{k \times  {mn} \times  {mn}},{V}_{P} \in  {\mathbb{R}}^{k \times  {mn}}$ 和 ${b}_{P} \in  {\mathbb{R}}^{k}$ 。参数 ${u}_{P}$ 计算张量积返回的二次特征的线性组合。利用公式 (1) 和 (5)，可以通过先计算复杂公式中包含的闭式项和原子公式的实例化，来计算复杂 LTNs 公式 Northern 的实例化。然后，使用特定的 t-范数组合这些实例化结果，参见公式 (2)。


Learning the groundings involves the optimization of the truth values of the formulas in a LTNs knowledge base, a.k.a. grounded theory. A partial grounding $\widehat{\mathcal{G}}$ is a grounding defined on a subset of the signature of $\mathcal{P}\mathcal{L}$ . A grounding $\mathcal{G}$ for $\mathcal{P}\mathcal{L}$ is a completion of $\widehat{\mathcal{G}}$ (in symbols $\widehat{\mathcal{G}} \subseteq  \mathcal{G}$ ) if $\mathcal{G}$ coincides with $\widehat{\mathcal{G}}$ on the symbols where $\widehat{\mathcal{G}}$ is defined.
学习实例化涉及优化 LTNs 知识库（又称实例化理论）中公式的真值。部分实例化 $\widehat{\mathcal{G}}$ 是定义在 $\mathcal{P}\mathcal{L}$ 签名子集上的实例化。如果 $\mathcal{G}$ 在 $\widehat{\mathcal{G}}$ 已定义的符号上与 $\widehat{\mathcal{G}}$ 一致，则 $\mathcal{P}\mathcal{L}$ 的实例化 $\mathcal{G}$ 是 $\widehat{\mathcal{G}}$ 的补全（符号表示为 $\widehat{\mathcal{G}} \subseteq  \mathcal{G}$ ）。


Definition 3: A grounded theory GT is a pair $\langle \mathcal{K},\widehat{\mathcal{G}}\rangle$ with $\mathcal{K}$ a set of closed formulas and $\widehat{\mathcal{G}}$ a partial grounding.
定义 3：实例化理论 GT 是一个二元组 $\langle \mathcal{K},\widehat{\mathcal{G}}\rangle$ ，其中 $\mathcal{K}$ 是一组闭式公式，$\widehat{\mathcal{G}}$ 是部分实例化。


Definition 4: A grounding $\mathcal{G}$ satisfies a grounded theory $\langle \mathcal{K},\widehat{\mathcal{G}}\rangle$ if $\widehat{\mathcal{G}} \subseteq  \mathcal{G}$ and $\mathcal{G}\left( \phi \right)  = 1$ ,for all $\phi  \in  \mathcal{K}$ . A grounded theory $\langle \mathcal{K},\widehat{\mathcal{G}}\rangle$ is satisfiable if there exists a grounding $\mathcal{G}$ that satisfies $\langle \mathcal{K},\widehat{\mathcal{G}}\rangle$ .
定义 4：如果对于所有 $\phi  \in  \mathcal{K}$，$\widehat{\mathcal{G}} \subseteq  \mathcal{G}$ 且 $\mathcal{G}\left( \phi \right)  = 1$，则基态化 $\mathcal{G}$ 满足基态理论 $\langle \mathcal{K},\widehat{\mathcal{G}}\rangle$。如果存在满足 $\langle \mathcal{K},\widehat{\mathcal{G}}\rangle$ 的基态化 $\mathcal{G}$，则基态理论 $\langle \mathcal{K},\widehat{\mathcal{G}}\rangle$ 是可满足的。


According to the above definition,the satisfiability of $\langle \mathcal{K},\widehat{\mathcal{G}}\rangle$ can be obtained by searching for a grounding $\mathcal{G}$ that extends $\widehat{\mathcal{G}}$ such that every formula in $\mathcal{K}$ has value 1 . When a grounded theory is not satisfiable a user can be interested in a degree of satisfaction of the GT.
根据上述定义，$\langle \mathcal{K},\widehat{\mathcal{G}}\rangle$ 的可满足性可以通过寻找一个扩展了 $\widehat{\mathcal{G}}$ 的基态化 $\mathcal{G}$ 来获得，使得 $\mathcal{K}$ 中的每个公式的值均为 1。当一个基态理论不可满足时，用户可能会对该 GT 的满足程度感兴趣。


Definition 5: Let $\langle \mathcal{K},\widehat{\mathcal{G}}\rangle$ be a grounded theory,the best satisfiability problem amounts at searching an extension ${\mathcal{G}}^{ * }$ of $\widehat{\mathcal{G}}$ in $\mathbb{G}$ (the set of all possible groundings) that maximizes the truth value of the conjunction of the formulas in $\mathcal{K}$ :
定义 5：设 $\langle \mathcal{K},\widehat{\mathcal{G}}\rangle$ 为一个基态理论，最佳可满足性问题相当于在 $\mathbb{G}$（所有可能基态化的集合）中寻找 $\widehat{\mathcal{G}}$ 的一个扩展 ${\mathcal{G}}^{ * }$，使 $\mathcal{K}$ 中公式合取的真值最大化：


$$
{\mathcal{G}}^{ * } = \mathop{\operatorname{argmax}}\limits_{{\widehat{\mathcal{G}} \subseteq  \mathcal{G} \in  \mathbb{G}}}\mathcal{G}\left( {\mathop{\bigwedge }\limits_{{\phi  \in  \mathcal{K}}}\phi }\right) . \tag{6}
$$



---



${}^{1}$ The popular mean operators (arithmetic,geometric and harmonic mean) are obtained by setting $p = 1,2$ ,and -1,respectively.
${}^{1}$ 常用的均值算子（算术、几何和调和平均值）分别通过设置 $p = 1,2$ 为 1、0 和 -1 获得。


${}^{2}$ Some groundings can be manually defined with some rules [5]. However, for some predicates this could be time consuming or inaccurate [10].
${}^{2}$ 一些基态化可以通过某些规则手动定义 [5]。然而，对于某些谓词，这可能耗时或不准确 [10]。


---



The best satisfiability problem is an optimization problem on the set of parameters to be learned. Let $\Theta  = \; \left\{  {{W}_{P},{V}_{P},{b}_{P},{u}_{P} \mid  P \in  \mathcal{P}}\right\}$ be the set of parameters. Let $\mathcal{G}\left( {\cdot  \mid  \Theta }\right)$ be the grounding obtained by setting the parameters of the grounding functions to $\Theta$ . The best satisfiability problem tries to find the best set of parameters $\Theta$ :
最佳可满足性问题是一个关于待学习参数集的优化问题。设 $\Theta  = \; \left\{  {{W}_{P},{V}_{P},{b}_{P},{u}_{P} \mid  P \in  \mathcal{P}}\right\}$ 为参数集。设 $\mathcal{G}\left( {\cdot  \mid  \Theta }\right)$ 为通过将基态化函数的参数设置为 $\Theta$ 而获得的基态化。最佳可满足性问题旨在寻找最佳参数集 $\Theta$：


$$
{\Theta }^{ * } = {\operatorname{argmax}}_{\Theta }\mathcal{G}\left( {\mathop{\bigwedge }\limits_{{\phi  \in  \mathcal{K}}}\phi  \mid  \Theta }\right)  - \lambda \parallel \Theta {\parallel }_{2}^{2} \tag{7}
$$



with $\lambda \parallel \Theta {\parallel }_{2}^{2}$ a regularization term.
其中 $\lambda \parallel \Theta {\parallel }_{2}^{2}$ 为正则化项。


## IV. LTNS FOR VISUAL RELATIONSHIP DETECTION
## IV. 用于视觉关系检测的 LTNS


Similarly to [10], we encode the problem of detecting visual relationship with LTNs. However, the problem here is more challenging as the VRD contains many binary predicates and not only the partOf as in the PASCAL-PART dataset. In the following we describe the novel contributions of our work.
与 [10] 类似，我们使用 LTN 对视觉关系检测问题进行建模。然而，这里的问题更具挑战性，因为 VRD 包含许多二元谓词，而不像 PASCAL-PART 数据集那样仅包含 partOf。下面我们描述本工作的创新贡献。


Let Pics be a dataset of images. Given a picture $p \in$ Pics, let $B\left( p\right)$ the corresponding set of bounding boxes. Each bounding box in $B\left( p\right)$ is annotated with a set of labels that describe the contained physical object. Pairs of bounding boxes are annotated with the semantic relations between the contained physical objects. Let ${\sum }_{\mathrm{{SII}}} = \langle \mathcal{P},\mathcal{C}\rangle$ be a $\mathcal{P}\mathcal{L}$ signature where $\mathcal{P} = {\mathcal{P}}_{1} \cup  {\mathcal{P}}_{2}$ is the set of predicates. ${\mathcal{P}}_{1}$ is the set of unary predicates that are the object types (or semantic classes) used to label the bounding boxes, e.g., ${\mathcal{P}}_{1} = \{$ Horse,Person,Shirt,Pizza,... $\}$ . The set ${\mathcal{P}}_{2}$ contains binary predicates used to label pairs of bounding boxes, e.g., ${\mathcal{P}}_{2} = \{$ ride,on,wear,eat, $\ldots \}$ . Let $\mathcal{C} = \mathop{\bigcup }\limits_{{p \in  \text{ Pics }}}B\left( p\right)$ be the set of constants for all the bounding boxes in the dataset Pics. However, the information in Pics is incomplete: many bounding boxes (or pairs of) have no annotations or even some pictures have no annotations at all. Therefore, LTNs is used to exploit the information in Pics to complete the missing information, i.e., to predict the visual relationships. As in [10], we encode Pics with a grounded theory ${\mathcal{T}}_{\text{ SII }} = \left\langle  {{\mathcal{K}}_{\text{ SII }},{\mathcal{G}}_{\text{ SII }}}\right\rangle$ described in the following. The LTNs knowledge base ${\mathcal{K}}_{\text{ SII }}$ encodes the bounding box annotations in the dataset and some background knowledge about the domain. The task is to complete the partial knowledge in Pics by finding a grounding ${\mathcal{G}}_{\mathrm{{SII}}}^{ * }$ ,that extends ${\widehat{\mathcal{G}}}_{\mathrm{{SII}}}$ ,such that:
令 Pics 为一个图像数据集。给定一张图片 $p \in$ Pics，令 $B\left( p\right)$ 为对应的边界框集合。$B\left( p\right)$ 中的每个边界框都标注有一组描述所含物理对象的标签。边界框对标注有其所含物理对象之间的语义关系。令 ${\sum }_{\mathrm{{SII}}} = \langle \mathcal{P},\mathcal{C}\rangle$ 为一个 $\mathcal{P}\mathcal{L}$ 签名，其中 $\mathcal{P} = {\mathcal{P}}_{1} \cup  {\mathcal{P}}_{2}$ 是谓词集合。${\mathcal{P}}_{1}$ 是表示对象类型（或语义类别）的一元谓词集合，用于标注边界框，例如 ${\mathcal{P}}_{1} = \{$ Horse,Person,Shirt,Pizza,... $\}$。集合 ${\mathcal{P}}_{2}$ 包含用于标注边界框对的二元谓词，例如 ${\mathcal{P}}_{2} = \{$ ride,on,wear,eat, $\ldots \}$。令 $\mathcal{C} = \mathop{\bigcup }\limits_{{p \in  \text{ Pics }}}B\left( p\right)$ 为数据集 Pics 中所有边界框的常量集合。然而，Pics 中的信息是不完整的：许多边界框（或其组合）没有标注，甚至有些图片完全没有标注。因此，LTNs 被用来利用 Pics 中的信息来补全缺失信息，即预测视觉关系。如 [10] 中所述，我们用下文描述的实例化理论 ${\mathcal{T}}_{\text{ SII }} = \left\langle  {{\mathcal{K}}_{\text{ SII }},{\mathcal{G}}_{\text{ SII }}}\right\rangle$ 对 Pics 进行编码。LTNs 知识库 ${\mathcal{K}}_{\text{ SII }}$ 编码了数据集中的边界框标注以及关于该领域的一些背景知识。任务是通过寻找一个扩展 ${\widehat{\mathcal{G}}}_{\mathrm{{SII}}}$ 的解释 ${\mathcal{G}}_{\mathrm{{SII}}}^{ * }$ 来补全 Pics 中的部分知识，使得：


$$
{\mathcal{G}}_{\mathrm{{SII}}}^{ * }\left( {C\left( b\right) }\right) \; \mapsto  \;\left\lbrack  {0,1}\right\rbrack
$$



$$
{\mathcal{G}}_{\mathrm{{SII}}}^{ * }\left( {R\left( {{b}_{1},{b}_{2}}\right) }\right) \; \mapsto  \;\left\lbrack  {0,1}\right\rbrack
$$



for every unary $\left( C\right)$ and binary $\left( R\right)$ predicate symbol and for every (pair of) bounding box in the dataset.
对于每一个一元 $\left( C\right)$ 和二元 $\left( R\right)$ 谓词符号，以及数据集中的每一对（或每一个）边界框。


### A.The Knowledge Base ${\mathcal{K}}_{\text{ SII }}$
### A.知识库 ${\mathcal{K}}_{\text{ SII }}$


The knowledge base ${\mathcal{K}}_{\text{ SII }}$ contains positive and negative examples (used for learning the grounding of the predicates in $\mathcal{P}$ ) and the background knowledge. The positive examples (taken from the annotations in Pics) for a semantic class $C$ are the atomic formulas $C\left( b\right)$ ,for every bounding box $b$ labelled with class $C \in  {\mathcal{P}}_{1}$ in Pics. The positive examples for a relation $R$ are the atomic formulas $R\left( {{b}_{1},{b}_{2}}\right)$ ,for every pair of bounding boxes $\left\langle  {{b}_{1},{b}_{2}}\right\rangle$ labelled with the binary relation $R \in  {\mathcal{P}}_{2}$ in Pics. Regarding the negative examples,for a semantic class $C$ we consider the atomic formulas $\neg C\left( b\right)$ , for every bounding box $b$ not labelled with $C$ . The negative examples for a relation $R$ are the atomic formulas $\neg R\left( {{b}_{1},{b}_{2}}\right)$ , for every pair of bounding boxes $\left\langle  {{b}_{1},{b}_{2}}\right\rangle$ not labelled with $R$ .
知识库 ${\mathcal{K}}_{\text{ SII }}$ 包含正例和负例（用于学习 $\mathcal{P}$ 中谓词的解释）以及背景知识。对于语义类别 $C$，其正例（取自 Pics 中的标注）是对 Pics 中每个标注为类别 $C \in  {\mathcal{P}}_{1}$ 的边界框 $b$ 的原子公式 $C\left( b\right)$。对于关系 $R$，其正例是对 Pics 中每个标注为二元关系 $R \in  {\mathcal{P}}_{2}$ 的边界框对 $\left\langle  {{b}_{1},{b}_{2}}\right\rangle$ 的原子公式 $R\left( {{b}_{1},{b}_{2}}\right)$。关于负例，对于语义类别 $C$，我们考虑对每个未标注为 $C$ 的边界框 $b$ 的原子公式 $\neg C\left( b\right)$。对于关系 $R$，负例是对每个未标注为 $R$ 的边界框对 $\left\langle  {{b}_{1},{b}_{2}}\right\rangle$ 的原子公式 $\neg R\left( {{b}_{1},{b}_{2}}\right)$。


Regarding the background knowledge, we manually build the logical constraints. We focus on the negative domain and range constraints that list which are the semantic classes that cannot be the subject/object for a predicate. E.g., clothes cannot drive. For every unary predicate Dress in ${\mathcal{P}}_{1}$ that refers to a dress,the constraint $\forall {xy}\left( {\operatorname{drive}\left( {x,y}\right)  \rightarrow  \neg \operatorname{Dress}\left( x\right) }\right)$ is added to $\mathcal{{BK}}$ . In a large scale setting,these constraints can be retrieved by on-line linguistic resources such as FrameNet [3] and VerbNet [26] that provide the range and domain of binary relations through the so-called frames data structure. Then, by applying mutual exclusivity between classes we obtain the negative domain and range constraints. This class of constraints brings to good performance. Experiments with other classes of constraints (such as IsA, mutual exclusivity, symmetry or reflexivity properties) are left as future work.
关于背景知识，我们手动构建逻辑约束。我们侧重于负向定义域和值域约束，列出哪些语义类别不能作为谓词的主语/宾语。例如，衣服不能驾驶。对于 ${\mathcal{P}}_{1}$ 中指向一件衣服的每个一元谓词 Dress，将约束 $\forall {xy}\left( {\operatorname{drive}\left( {x,y}\right)  \rightarrow  \neg \operatorname{Dress}\left( x\right) }\right)$ 添加到 $\mathcal{{BK}}$ 中。在大规模设置下，这些约束可以通过在线语言资源（如 FrameNet [3] 和 VerbNet [26]）获取，这些资源通过所谓的框架数据结构提供二元关系的定义域和值域。然后，通过应用类之间的互斥性，我们获得负向定义域和值域约束。此类约束带来了良好的性能。与其他类别约束（如 IsA、互斥性、对称性或自反性性质）的实验将作为未来的工作。


### B.The Grounding ${\widehat{\mathcal{G}}}_{\text{ SII }}$
### B. 基元化 ${\widehat{\mathcal{G}}}_{\text{ SII }}$


The grounding of each bounding box constant $b \in  \mathcal{C}$ is a feature vector ${\widehat{\mathcal{G}}}_{\text{ SII }}\left( b\right)  = {\mathbf{v}}_{b} \in  {\mathbb{R}}^{\left| {\mathcal{P}}_{1}\right|  + 4}$ of semantic and geometric features:
每个边界框常量 $b \in  \mathcal{C}$ 的基元化是一个由语义和几何特征组成的特征向量 ${\widehat{\mathcal{G}}}_{\text{ SII }}\left( b\right)  = {\mathbf{v}}_{b} \in  {\mathbb{R}}^{\left| {\mathcal{P}}_{1}\right|  + 4}$：


$$
{\mathbf{v}}_{b} = \left\langle  {\operatorname{score}\left( {{C}_{1},b}\right) ,\ldots ,\operatorname{score}\left( {{C}_{\left| {\mathcal{P}}_{1}\right| },b}\right) }\right.
$$



$$
\left. {{x}_{0}\left( b\right) ,{y}_{0}\left( b\right) ,{x}_{1}\left( b\right) ,{y}_{1}\left( b\right) }\right\rangle  , \tag{8}
$$



with ${x}_{0}\left( b\right) ,{y}_{0}\left( b\right) ,{x}_{1}\left( b\right) ,{y}_{1}\left( b\right)$ the coordinates of the top-left and bottom-right corners of $b$ and $\operatorname{score}\left( {{C}_{i},b}\right)$ is the classification score of an object detector for $b$ according to the class ${C}_{i} \in  {\mathcal{P}}_{1}$ . However,here we adopt the one-hot encoding: the semantic features take value 1 in the position of the class with the highest detection score, 0 otherwise. The geometric features remain unchanged. The grounding for a pair of bounding boxes $\left\langle  {{b}_{1},{b}_{2}}\right\rangle$ is the concatenation of the groundings of the single bounding boxes $\left\langle  {{\mathbf{v}}_{{b}_{1}} : {\mathbf{v}}_{{b}_{2}}}\right\rangle$ . However,when dealing with $n$ -tuples of objects,adding new extra feature regarding geometrical joint properties of these $n$ bounding boxes improves the performance of the LTNs model. Differently from [10],we add more joint features ${}^{3}$ that better capture the geometric interactions between bounding boxes:
其中 ${x}_{0}\left( b\right) ,{y}_{0}\left( b\right) ,{x}_{1}\left( b\right) ,{y}_{1}\left( b\right)$ 是 $b$ 左上角和右下角的坐标，$\operatorname{score}\left( {{C}_{i},b}\right)$ 是目标检测器根据类别 ${C}_{i} \in  {\mathcal{P}}_{1}$ 对 $b$ 的分类评分。然而，这里我们采用独热编码：语义特征在检测评分最高的类别位置取值为 1，否则为 0。几何特征保持不变。边界框对 $\left\langle  {{b}_{1},{b}_{2}}\right\rangle$ 的基元化是单个边界框基元化 $\left\langle  {{\mathbf{v}}_{{b}_{1}} : {\mathbf{v}}_{{b}_{2}}}\right\rangle$ 的拼接。然而，在处理 $n$ 元组对象时，添加关于这些 $n$ 个边界框几何关节属性的新额外特征可提高 LTNs 模型的性能。与 [10] 不同，我们添加了更多关节特征 ${}^{3}$，以更好地捕捉边界框之间的几何交互：


$$
{\mathbf{v}}_{{b}_{1},{b}_{2}} = \left\langle  {{\mathbf{v}}_{{b}_{1}} : {\mathbf{v}}_{{b}_{2}},\operatorname{ir}\left( {{b}_{1},{b}_{2}}\right) ,\operatorname{ir}\left( {{b}_{2},{b}_{1}}\right) ,\frac{\operatorname{area}\left( {b}_{1}\right) }{\operatorname{area}\left( {b}_{2}\right) },}\right.
$$



$$
\frac{\operatorname{area}\left( {b}_{2}\right) }{\operatorname{area}\left( {b}_{1}\right) },\operatorname{euclid\_ dist}\left( {{b}_{1},{b}_{2}}\right) ,\sin \left( {{b}_{1},{b}_{2}}\right) ,\cos \left( {{b}_{1},{b}_{2}}\right) \rangle \tag{9}
$$



where:
其中：


- $\operatorname{ir}\left( {{b}_{1},{b}_{2}}\right)  = \operatorname{intersec}\left( {{b}_{1},{b}_{2}}\right) /\operatorname{area}\left( {b}_{1}\right)$ is the inclusion ratio, see [10];
- $\operatorname{ir}\left( {{b}_{1},{b}_{2}}\right)  = \operatorname{intersec}\left( {{b}_{1},{b}_{2}}\right) /\operatorname{area}\left( {b}_{1}\right)$ 是包含率，见 [10]；


---



${}^{3}$ All the features are normalized in the interval $\left\lbrack  {-1,1}\right\rbrack$ .
${}^{3}$ 所有特征都归一化到区间 $\left\lbrack  {-1,1}\right\rbrack$ 内。


---



- $\operatorname{area}\left( b\right)$ is the area of $b$ ;
- $\operatorname{area}\left( b\right)$ 是 $b$ 的面积；


- intersec $\left( {{b}_{1},{b}_{2}}\right)$ is the area of the intersection of ${b}_{1},{b}_{2}$ ;
- intersec $\left( {{b}_{1},{b}_{2}}\right)$ 是 ${b}_{1},{b}_{2}$ 交集的面积；


- euclid_dist $\left( {{b}_{1},{b}_{2}}\right)$ is the Euclidean distance between the centroids of bounding boxes ${b}_{1},{b}_{2}$ ;
- euclid_dist $\left( {{b}_{1},{b}_{2}}\right)$ 是边界框 ${b}_{1},{b}_{2}$ 质心之间的欧几里得距离；


- $\sin \left( {{b}_{1},{b}_{2}}\right)$ and $\cos \left( {{b}_{1},{b}_{2}}\right)$ are the sine and cosine of the angle between the centroids of ${b}_{1}$ and ${b}_{2}$ computed in a counter-clockwise manner.
- $\sin \left( {{b}_{1},{b}_{2}}\right)$ 和 $\cos \left( {{b}_{1},{b}_{2}}\right)$ 是 ${b}_{1}$ 和 ${b}_{2}$ 质心之间以逆时针方式计算的夹角的正弦和余弦值。


Regarding the unary predicates in ${\mathcal{P}}_{1}$ ,we adopt a rule-based grounding. Given a bounding box constant $b$ ,its feature vector ${\mathbf{v}}_{b} = \left\langle  {{v}_{1},\ldots ,{v}_{\left| {\mathcal{P}}_{1}\right|  + 4}}\right\rangle$ ,and a predicate symbol ${C}_{i} \in  {\mathcal{P}}_{1}$ ,the grounding for ${C}_{i}$ is:
对于 ${\mathcal{P}}_{1}$ 中的一元谓词，我们采用基于规则的实例化。给定边界框常量 $b$ 、其特征向量 ${\mathbf{v}}_{b} = \left\langle  {{v}_{1},\ldots ,{v}_{\left| {\mathcal{P}}_{1}\right|  + 4}}\right\rangle$ 以及谓词符号 ${C}_{i} \in  {\mathcal{P}}_{1}$ ， ${C}_{i}$ 的实例化为：


$$
{\widehat{\mathcal{G}}}_{\mathrm{{SII}}}\left( {C}_{i}\right) \left( {\mathbf{v}}_{b}\right)  = \left\{  \begin{array}{ll} 1 & \text{ if }i = {\operatorname{argmax}}_{1 \leq  l \leq  \left| {\mathcal{P}}_{1}\right| }{\mathbf{v}}_{b}^{l} \\  0 & \text{ otherwise. } \end{array}\right. \tag{10}
$$



Regarding the binary predicates in ${\mathcal{P}}_{2}$ ,a rule-based grounding would require a different analysis for each predicate and could be inaccurate. Therefore, this grounding is learned from data by maximizing the truth values of the formulas in ${\mathcal{K}}_{\text{ SII }}$ . The grounding of the logical constraints in ${\mathcal{K}}_{\text{ SII }}$ is computed by (i) instantiating a tractable sample of the constraints with bounding box constants belonging to the same picture. (ii) Computing the groundings of the atomic formulas of every instantiated constraint; (iii) combining the groundings of the atomic formulas according to the LTNs semantics; (iv) aggregating the groundings of every instantiated constraint according to the LTNs semantics of $\forall$ .
对于 ${\mathcal{P}}_{2}$ 中的二元谓词，基于规则的实例化需要对每个谓词进行不同的分析且可能不准确。因此，该实例化通过最大化 ${\mathcal{K}}_{\text{ SII }}$ 中公式的真值从数据中学习。 ${\mathcal{K}}_{\text{ SII }}$ 中逻辑约束的实例化通过以下方式计算：(i) 用属于同一图片的边界框常量实例化约束的可处理样本；(ii) 计算每个实例化约束的原子公式的实例化；(iii) 根据 LTNs 语义组合原子公式的实例化；(iv) 根据 $\forall$ 的 LTNs 语义聚合每个实例化约束的实例化。


### C.The Optimization of ${\mathcal{T}}_{\text{ SII }}$
### C. ${\mathcal{T}}_{\text{ SII }}$ 的优化


Equation (7) defines how to learn the LTNs parameters by maximizing the grounding of the conjunctions of the formulas in ${\mathcal{K}}_{\text{ SII }}$ . Here we analyze some problems that can arise when the optimization is performed with the main t-norms:
等式 (7) 定义了如何通过最大化 ${\mathcal{K}}_{\text{ SII }}$ 中公式合取的实例化来学习 LTNs 参数。这里我们分析了使用主要 t-范数进行优化时可能出现的一些问题：


Lukasiewicz t-norm The satisfiability of ${\mathcal{K}}_{\text{ SII }}$ is given by: ${\widehat{\mathcal{G}}}_{\mathrm{{SII}}}\left( {\mathop{\bigwedge }\limits_{{\phi  \in  {\mathcal{K}}_{\mathrm{{SII}}}}}\phi }\right)  = \max \left\{  {0,\mathop{\sum }\limits_{{\phi  \in  {\mathcal{K}}_{\mathrm{{SII}}}}}{\widehat{\mathcal{G}}}_{\mathrm{{SII}}}\left( \phi \right)  - \left| {\mathcal{K}}_{\mathrm{{SII}}}\right|  + }\right.$ 1\}. Thus, the higher the number of formulas the higher their grounding should be to have a satisfiability value bigger than zero. However, even a small number of formulas in ${\mathcal{K}}_{\text{ SII }}$ with a low grounding value can lead the knowledge base satisfiability to zero.
Lukasiewicz t-范数 ${\mathcal{K}}_{\text{ SII }}$ 的可满足性由 ${\widehat{\mathcal{G}}}_{\mathrm{{SII}}}\left( {\mathop{\bigwedge }\limits_{{\phi  \in  {\mathcal{K}}_{\mathrm{{SII}}}}}\phi }\right)  = \max \left\{  {0,\mathop{\sum }\limits_{{\phi  \in  {\mathcal{K}}_{\mathrm{{SII}}}}}{\widehat{\mathcal{G}}}_{\mathrm{{SII}}}\left( \phi \right)  - \left| {\mathcal{K}}_{\mathrm{{SII}}}\right|  + }\right.$ 1\} 给出。因此，公式数量越多，其实例化值就必须越高，才能使可满足性值大于零。然而，即使 ${\mathcal{K}}_{\text{ SII }}$ 中只有少量具有低实例化值的公式，也可能导致知识库的可满足性降至零。


Gödel t-norm The satisfiability of ${\mathcal{K}}_{\text{ SII }}$ is the minimum of the groundings of all its formulas: ${\widehat{\mathcal{G}}}_{\mathrm{{SII}}}\left( {\mathop{\bigwedge }\limits_{{\phi  \in  {\mathcal{K}}_{\mathrm{{SII}}}}}\phi }\right)  = \; \min \left\{  {{\widehat{\mathcal{G}}}_{\mathrm{{SII}}}\left( \phi \right)  \mid  \phi  \in  {\mathcal{K}}_{\mathrm{{SII}}}}\right\}$ . Here the optimization process could get stuck in a local optimum. Indeed, a single predicate could be too difficult to learn, the optimizer tries to increase this value without any improvement and thus leaving out the other predicates from the optimization.
Gödel t-范数 ${\mathcal{K}}_{\text{ SII }}$ 的可满足性是其所有公式实例化的最小值： ${\widehat{\mathcal{G}}}_{\mathrm{{SII}}}\left( {\mathop{\bigwedge }\limits_{{\phi  \in  {\mathcal{K}}_{\mathrm{{SII}}}}}\phi }\right)  = \; \min \left\{  {{\widehat{\mathcal{G}}}_{\mathrm{{SII}}}\left( \phi \right)  \mid  \phi  \in  {\mathcal{K}}_{\mathrm{{SII}}}}\right\}$ 。在这里，优化过程可能会陷入局部最优。事实上，单个谓词可能太难学习，优化器试图增加该值却没有任何改进，从而使其他谓词脱离了优化。


Product t-norm The satisfiability of ${\mathcal{K}}_{\text{ SII }}$ is the product of the groundings of all its formulas: ${\widehat{\mathcal{G}}}_{\mathrm{{SII}}}\left( {\mathop{\bigwedge }\limits_{{\phi  \in  {\mathcal{K}}_{\mathrm{{SII}}}}}\phi }\right)  = \; \mathop{\prod }\limits_{{\phi  \in  {\mathcal{K}}_{\mathrm{{SII}}}}}{\widehat{\mathcal{G}}}_{\mathrm{{SII}}}\left( \phi \right)$ . As ${\mathcal{K}}_{\mathrm{{SII}}}$ can have many formulas,the product of hundreds of groundings can result in a very small number and thus incurring in underflow problems.
乘积 T-范数 ${\mathcal{K}}_{\text{ SII }}$ 的可满足度是其所有公式示例化的乘积：${\widehat{\mathcal{G}}}_{\mathrm{{SII}}}\left( {\mathop{\bigwedge }\limits_{{\phi  \in  {\mathcal{K}}_{\mathrm{{SII}}}}}\phi }\right)  = \; \mathop{\prod }\limits_{{\phi  \in  {\mathcal{K}}_{\mathrm{{SII}}}}}{\widehat{\mathcal{G}}}_{\mathrm{{SII}}}\left( \phi \right)$。由于 ${\mathcal{K}}_{\mathrm{{SII}}}$ 可能包含大量公式，数百个示例化值的乘积可能导致极小的数值，从而引发下溢问题。


Differently from [10], we provide another definition of satisfiability. We use a mean operator in Equation (7) that returns a global satisfiability of ${\mathcal{K}}_{\text{ SII }}$ avoiding the mentioned issues:
与 [10] 不同，我们提供了另一种可满足度定义。我们在等式 (7) 中使用均值算子来返回 ${\mathcal{K}}_{\text{ SII }}$ 的全局可满足度，从而避免上述问题：


$$
{\Theta }^{ * } = {\operatorname{argmax}}_{\Theta }{\operatorname{mean}}_{p}\left( {{\widehat{\mathcal{G}}}_{\mathrm{{SII}}}\left( {\phi  \mid  \Theta }\right)  \mid  \phi  \in  {\mathcal{K}}_{\mathrm{{SII}}}}\right)  - \lambda \parallel \Theta {\parallel }_{2}^{2},
$$



(11)
with $p \in  \mathbb{Z}$ . Here,we avoid $p \geq  1$ as the obtained means are more influenced by the higher grounding values, that is, by the predicates easy to learn. These means return a too optimistic value of the satisfiability and this wrongly avoids the need of optimization. The computation of Equation (11) is linear with respect to the number of formulas in ${\mathcal{K}}_{\text{ SII }}$ .
其中 $p \in  \mathbb{Z}$。这里我们避免使用 $p \geq  1$，因为所得均值受较高示例化值（即易于学习的谓词）影响更大。这些均值返回的可满足度过于乐观，会错误地规避优化需求。等式 (11) 的计算量与 ${\mathcal{K}}_{\text{ SII }}$ 中的公式数量成线性关系。


## D. Post Processing
## D. 后处理


Given a trained grounded theory ${\mathcal{T}}_{\text{ SII }}$ ,we compute the set of groundings ${\left\{  {\widehat{\mathcal{G}}}_{\mathrm{{SII}}}\left( \mathrm{r}\left( b,{b}^{\prime }\right) \right) \right\}  }_{\mathrm{r} \in  {\mathcal{P}}_{2}}$ ,with $\left\langle  {b,{b}^{\prime }}\right\rangle$ a new pair of bounding boxes. Then,every grounding ${\widehat{\mathcal{G}}}_{\mathrm{{SII}}}\left( {\mathrm{r}\left( {b,{b}^{\prime }}\right) }\right)$ is multiplied with a prior: the frequency of the predicate $r$ in the training set. In addition, we exploit equivalences between the binary predicates (e.g., beside is equivalent to next to) to normalize the groundings. In the specific, ${\widehat{\mathcal{G}}}_{\mathrm{{SII}}}\left( {{\mathrm{r}}_{1}\left( {b,{b}^{\prime }}\right) }\right)  = \; {\widehat{\mathcal{G}}}_{\mathrm{{SII}}}\left( {{\mathrm{r}}_{2}\left( {b,{b}^{\prime }}\right) }\right)  = \max \left\{  {{\widehat{\mathcal{G}}}_{\mathrm{{SII}}}\left( {{\mathrm{r}}_{1}\left( {b,{b}^{\prime }}\right) }\right) ,{\widehat{\mathcal{G}}}_{\mathrm{{SII}}}\left( {{\mathrm{r}}_{2}\left( {b,{b}^{\prime }}\right) }\right) }\right\}$ ,if ${\mathrm{r}}_{1}$ and ${\mathrm{r}}_{2}$ are equivalent.
给定训练好的实例化理论 ${\mathcal{T}}_{\text{ SII }}$，我们计算示例化集合 ${\left\{  {\widehat{\mathcal{G}}}_{\mathrm{{SII}}}\left( \mathrm{r}\left( b,{b}^{\prime }\right) \right) \right\}  }_{\mathrm{r} \in  {\mathcal{P}}_{2}}$，其中 $\left\langle  {b,{b}^{\prime }}\right\rangle$ 为一组新的边界框。接着，每个示例化 ${\widehat{\mathcal{G}}}_{\mathrm{{SII}}}\left( {\mathrm{r}\left( {b,{b}^{\prime }}\right) }\right)$ 都会乘以一个先验值：谓词 $r$ 在训练集中的频率。此外，我们利用二元谓词之间的等价性（例如 beside 等同于 next to）来归一化示例化值。具体而言，若 ${\mathrm{r}}_{1}$ 与 ${\mathrm{r}}_{2}$ 等价，则 ${\widehat{\mathcal{G}}}_{\mathrm{{SII}}}\left( {{\mathrm{r}}_{1}\left( {b,{b}^{\prime }}\right) }\right)  = \; {\widehat{\mathcal{G}}}_{\mathrm{{SII}}}\left( {{\mathrm{r}}_{2}\left( {b,{b}^{\prime }}\right) }\right)  = \max \left\{  {{\widehat{\mathcal{G}}}_{\mathrm{{SII}}}\left( {{\mathrm{r}}_{1}\left( {b,{b}^{\prime }}\right) }\right) ,{\widehat{\mathcal{G}}}_{\mathrm{{SII}}}\left( {{\mathrm{r}}_{2}\left( {b,{b}^{\prime }}\right) }\right) }\right\}$。


## V. EXPERIMENTS
## V. 实验


We conduct the experiments ${}^{4}$ on the Visual Relationship Dataset (VRD) [19] that contains 4000 images for training and 1000 for testing annotated with visual relationships. Bounding boxes are annotated with a label in ${\mathcal{P}}_{1}$ containing 100 unary predicates. These labels refer to animals, vehicles, clothes and generic objects. Pairs of bounding boxes are annotated with a label in ${\mathcal{P}}_{2}$ containing 70 binary predicates. These labels refer to actions, prepositions, spatial relations, comparatives or preposition phrases. The dataset has 37993 instances of visual relationships and 6672 types of relationships. 1877 instances of relationships occur only in the test set and they are used to evaluate the zero-shot learning scenario.
我们在视觉关系数据集 (VRD) [19] 上进行实验 ${}^{4}$，该数据集包含 4000 张训练图像和 1000 张测试图像，并带有视觉关系标注。边界框由包含 100 个一元谓词的 ${\mathcal{P}}_{1}$ 标签标注，涉及动物、车辆、衣物及通用物体。边界框对由包含 70 个二元谓词的 ${\mathcal{P}}_{2}$ 标签标注，涉及动作、介词、空间关系、比较级或介词短语。该数据集拥有 37993 个视觉关系实例和 6672 种关系类型。其中 1877 个关系实例仅出现在测试集中，用于评估零样本学习场景。


a) VRD Tasks: The performance of LTNs are tested on the following VRD standard tasks. The phrase detection is the prediction of a correct triple ⟨subject, predicate, object⟩ and its localization in a single bounding box containing both the subject and the object. The triple is a true positive if the labels are the same of the ground truth triple and if the predicted bounding box has at least 50% of overlap with a corresponding bounding box in the ground truth. The ground truth bounding box is the union of the ground truth bounding boxes of the subject and of the object. The relationship detection task predicts a correct triple/relationship and the bounding boxes containing the subject and the object of the relationship. The triple is a true positive if both bounding boxes have at least ${50}\%$ of overlap with the corresponding ones in the ground truth. The labels for the predicted triple have to match with the corresponding ones in the ground truth. The predicate detection task predicts a set of correct binary predicates between a given set of bounding boxes. Here, the prediction does not depend on the performance of an object detector. The focus is only on the ability of LTNs to predict binary predicates.
a) VRD任务：LTN的性能在以下VRD标准任务上进行了测试。短语检测是指预测正确的三元组⟨主语, 谓词, 宾语⟩，并将其定位在包含主语和宾语的单个边界框中。如果三元组标签与真值（ground truth）一致，且预测边界框与真值边界框的重合度至少达到50%，则该三元组为真阳性。真值边界框是主语和宾语真值边界框的并集。关系检测任务预测正确的三元组/关系，以及包含关系主语和宾语的边界框。如果两个边界框与真值中对应边界框的重合度均至少达到${50}\%$，则该三元组为真阳性。预测三元组的标签必须与真值中的对应标签匹配。谓词检测任务预测给定边界框集合之间的一组正确二元谓词。在此任务中，预测不依赖于目标检测器的性能，重点仅在于LTN预测二元谓词的能力。


---



${}^{4}$ The source code and the models are available at https://github.com/ivanDonadello/Visual-Relationship-Detection-LTN.
${}^{4}$ 源代码和模型可在 https://github.com/ivanDonadello/Visual-Relationship-Detection-LTN 获取。


A video showing a demo of the SII system can be seen at https://www.youtube.com/watch?v=y2-altg3FFw.
展示 SII 系统演示的视频可见于 https://www.youtube.com/watch?v=y2-altg3FFw。


---



b) Comparison: The performance of LTNs on these tasks have been evaluated with two LTNs grounded theories (or models): ${\mathcal{T}}_{\text{ expl }}$ and ${\mathcal{T}}_{\text{ prior }}$ . In the first one, ${\mathcal{T}}_{\text{ expl }} = \; \left\langle  {{\mathcal{K}}_{\text{ expl }},{\widehat{\mathcal{G}}}_{\text{ SII }}}\right\rangle  ,{\mathcal{K}}_{\text{ expl }}$ contains only positive and negative examples for the predicates in $\mathcal{P}$ . This theory gives us the first results on the effectiveness of LTNs on visual relationship detection with respect to the state-of-the-art. In the second grounded theory ${\mathcal{T}}_{\text{ prior }} = \left\langle  {{\mathcal{K}}_{\text{ prior }},{\widehat{\mathcal{G}}}_{\text{ SII }}}\right\rangle  ,{\mathcal{K}}_{\text{ prior }}$ contains examples and the logical constraints. With ${\mathcal{T}}_{\text{ prior }}$ we check the contribution of the logical constraints w.r.t. a standard machine learning approach. We first train the LTNs models on the VRD training set and then we evaluate them on the VRD test set. The evaluation tests the ability of LTNs to generalize to the 1877 relationships never seen in the training phase.
b) 比较：LTN在这些任务上的性能已通过两种LTN落地理论（或模型）进行了评估：${\mathcal{T}}_{\text{ expl }}$ 和 ${\mathcal{T}}_{\text{ prior }}$。在第一种理论中，${\mathcal{T}}_{\text{ expl }} = \; \left\langle  {{\mathcal{K}}_{\text{ expl }},{\widehat{\mathcal{G}}}_{\text{ SII }}}\right\rangle  ,{\mathcal{K}}_{\text{ expl }}$ 仅包含 $\mathcal{P}$ 中谓词的正例和负例。该理论为我们提供了关于LTN在视觉关系检测上相对于现有技术有效性的初步结果。在第二种落地理论中，${\mathcal{T}}_{\text{ prior }} = \left\langle  {{\mathcal{K}}_{\text{ prior }},{\widehat{\mathcal{G}}}_{\text{ SII }}}\right\rangle  ,{\mathcal{K}}_{\text{ prior }}$ 包含示例和逻辑约束。通过 ${\mathcal{T}}_{\text{ prior }}$，我们检查了逻辑约束相对于标准机器学习方法的贡献。我们首先在VRD训练集上训练LTN模型，然后在VRD测试集上进行评估。该评估测试了LTN对训练阶段从未见过的1877种关系的泛化能力。


Before comparing the two models with the state-of-the-art, we perform some ablation studies to see what are the key components of our SII system based on LTNs (see Table I). A first key feature are the logical constraints in ${\mathcal{T}}_{\text{ prior }}$ w.r.t. ${\mathcal{T}}_{\text{ expl }}$ . The second component is the contribution of the new joint features between bounding boxes, Equation (9). These features represent a novelty of our work w.r.t. the classical features of the LTNs framework. The third important aspect is the adoption of a loss function based on the harmonic mean of the clauses in ${\mathcal{K}}_{\text{ SII }}$ ,see Equation (11). This differs from the classical LTNs loss function based on the t-norm (e.g., the minimum) of the clauses in ${\mathcal{K}}_{\text{ SII }}$ . We test ${\mathcal{T}}_{\text{ expl }}$ and ${\mathcal{T}}_{\text{ prior }}$ by adding and removing the new features and by adopting the new loss function instead of the classical one of LTNs.
在将这两个模型与现有技术进行比较之前，我们进行了一些消融研究，以确定基于LTN的SII系统的关键组件（见表I）。第一个关键特性是 ${\mathcal{T}}_{\text{ prior }}$ 相对于 ${\mathcal{T}}_{\text{ expl }}$ 的逻辑约束。第二个组件是边界框之间新联合特征的贡献，见公式(9)。这些特征代表了我们工作相对于传统LTN框架特征的新颖性。第三个重要方面是采用了基于 ${\mathcal{K}}_{\text{ SII }}$ 中子句调和平均值的损失函数，见公式(11)。这与基于 ${\mathcal{K}}_{\text{ SII }}$ 中子句t-范数（例如最小值）的传统LTN损失函数不同。我们通过添加和删除新特征，并采用新损失函数代替传统LTN损失函数，对 ${\mathcal{T}}_{\text{ expl }}$ 和 ${\mathcal{T}}_{\text{ prior }}$ 进行了测试。


The LTNs models ${\mathcal{T}}_{\text{ expl }}$ and ${\mathcal{T}}_{\text{ prior }}$ are then compared with the following methods of the state-of-the-art, see Table II. VRD [19] is the seminal work on visual relationship detection and provides the VRD. The method detects visual relationships by combining visual and semantic information. The visual information is the classification score given by two convolutional neural networks. The first network classifies single bounding boxes according to the labels in ${\mathcal{P}}_{1}$ . The second one classifies the union of two bounding boxes (subject and object) according to the labels in ${\mathcal{P}}_{2}$ . These scores are combined with a language prior score (based on word embeddings) that models the semantics of the visual relationships. The methods in [2] also combines visual and semantic information. However, link prediction methods (RESCAL, MultiwayNN, CompleEx, DistMult) are used for modelling the visual relationship semantics in place of word embeddings. In LKD [31] every visual relationship is predicted with a neural network trained on visual features and on the word embeddings of the subject/object labels. This network is regularized with a term that encodes statistical dependencies (taken from the training set and Wikipedia) between the predicates and subjects/objects. In VRL [18] the semantic information of the triples is modelled by building a graph of the visual relationships in the training set. A visual relationship is discovered (starting from the proposals coming from an object detector) with a graph traversal algorithm in a reinforcement learning setting. Context-AwareVRD [35] and WeaklySup [23] encode the features of pairs of bounding boxes similarly to Equation (9). However, in Context-AwareVRD the learning is performed with a neural network, whereas in WeaklySup the learning is based on a weakly-supervised discriminative clustering I.e., the supervision on a given relationship is not at triples level but on an image level.
随后将 LTNs 模型 ${\mathcal{T}}_{\text{ expl }}$ 和 ${\mathcal{T}}_{\text{ prior }}$ 与以下最先进的方法进行对比，见表 II。VRD [19] 是视觉关系检测的开创性工作并提供了 VRD。该方法通过结合视觉和语义信息来检测视觉关系。视觉信息是两个卷积神经网络给出的分类得分。第一个网络根据 ${\mathcal{P}}_{1}$ 中的标签对单个边界框进行分类。第二个网络根据 ${\mathcal{P}}_{2}$ 中的标签对两个边界框（主语和宾语）的并集进行分类。这些得分与模拟视觉关系语义的语言先验得分（基于词嵌入）相结合。[2] 中的方法也结合了视觉和语义信息。然而，其使用链接预测方法（RESCAL、MultiwayNN、CompleEx、DistMult）代替词嵌入来建模视觉关系语义。在 LKD [31] 中，每个视觉关系都通过一个在视觉特征和主/宾语标签词嵌入上训练的神经网络进行预测。该网络使用一个编码谓词与主/宾语之间统计依赖关系（取自训练集和维基百科）的项进行正则化。在 VRL [18] 中，三元组的语义信息通过构建训练集中视觉关系的图来建模。在强化学习设置中，通过图遍历算法发现视觉关系（从目标检测器提供的建议框开始）。Context-AwareVRD [35] 和 WeaklySup [23] 对边界框对的特征编码类似于等式 (9)。然而，Context-AwareVRD 使用神经网络进行学习，而 WeaklySup 基于弱监督判别聚类进行学习，即对给定关系的监督不是在三元组层面，而是在图像层面。


c) Evaluation Metric: For each image in the test set, we use ${\mathcal{T}}_{\text{ expl }}$ and ${\mathcal{T}}_{\text{ prior }}$ to compute the ranked set of groundings ${\left\{  {\widehat{\mathcal{G}}}_{\mathrm{{SII}}}\left( \mathrm{r}\left( b,{b}^{\prime }\right) \right) \right\}  }_{\mathrm{r} \in  {\mathcal{P}}_{2}}$ ,with $\left\langle  {b,{b}^{\prime }}\right\rangle$ bounding boxes computed with an object detector (the R-CNN model in ${\left\lbrack  {19}\right\rbrack  }^{5}$ ) or taken from the ground truth (for the predicate detection). Then we perform the post processing. As metrics we use the recall@100/50 [19] as the annotation is not complete and precision would wrongly penalize true positives. We classify every pair $\left\langle  {b,{b}^{\prime }}\right\rangle$ with all the predicates in ${\mathcal{P}}_{2}$ as many predicates can occur between two objects (e.g., a person rides and is on a horse at the same time) and it is not always possible to define a preference between predicates. This choice is counterbalanced by predicting the correct relationships within the top 100 and 50 positions.
c) 评估指标：对于测试集中的每幅图像，我们使用 ${\mathcal{T}}_{\text{ expl }}$ 和 ${\mathcal{T}}_{\text{ prior }}$ 来计算排序后的落地集 ${\left\{  {\widehat{\mathcal{G}}}_{\mathrm{{SII}}}\left( \mathrm{r}\left( b,{b}^{\prime }\right) \right) \right\}  }_{\mathrm{r} \in  {\mathcal{P}}_{2}}$，其中 $\left\langle  {b,{b}^{\prime }}\right\rangle$ 边界框由目标检测器（${\left\lbrack  {19}\right\rbrack  }^{5}$ 中的 R-CNN 模型）计算或取自真值（用于谓词检测）。随后我们进行后处理。由于标注不完整且精确率会错误地惩罚真阳性，我们使用 recall@100/50 [19] 作为指标。由于两个物体之间可能存在多种谓词（例如，一个人骑在马上且同时在马上），且并不总是能定义谓词间的优先级，因此我们使用 ${\mathcal{P}}_{2}$ 中的所有谓词对每个对 $\left\langle  {b,{b}^{\prime }}\right\rangle$ 进行分类。这种选择通过在前 100 和 50 位中预测出正确关系来制衡。


d) Implementation Details: In Equations (4) and (11) we set $p =  - 1$ (harmonic mean). The chosen t-norm is the Łukasiewicz one. The number of tensor layers in Equation (5) is $k = 5$ and $\lambda  = {10}^{-{10}}$ in Equation (11). The optimization is performed separately on ${\mathcal{K}}_{\text{ prior }}$ and ${\mathcal{K}}_{\text{ expl }}$ with 2500 training epochs of the RMSProp optimizer in TENSORFLOW ${}^{TM}$ .
d) 实现细节：在等式 (4) 和 (11) 中，我们设置 $p =  - 1$（调和平均值）。选定的 t-范数为 Łukasiewicz 范数。等式 (5) 中的张量层数为 $k = 5$，等式 (11) 中为 $\lambda  = {10}^{-{10}}$。优化在 TENSORFLOW ${}^{TM}$ 中使用 RMSProp 优化器对 ${\mathcal{K}}_{\text{ prior }}$ 和 ${\mathcal{K}}_{\text{ expl }}$ 分别进行 2500 个训练轮次。


## A. Ablation Studies
## A. 消融研究


Table I shows the results of the ablation studies. We perform the training 10 times obtaining 10 models for ${\mathcal{T}}_{\text{ expl }}$ and ${\mathcal{T}}_{\text{ prior }}$ , respectively. For each task and for each grounded theory we report the mean and the standard deviation of the results given by these models. In addition, we perform the t-test with p-value 0.05 to statistically compare the LTNs models in each task. The first LTNs models use the classical LTNs loss function without the new joint features. Their performance are statistically improved of approximately 2 points in all the tasks if we add the new joint features. However, the minimum t-norm (i.e., the Gödel t-norm) leads the optimization process to a local optimum, as stated in Section IV-C, thus vanishing the effect of the logical constraints. Indeed, ${\mathcal{T}}_{\text{ expl }}$ and ${\mathcal{T}}_{\text{ prior }}$ have similar performance by adopting the Gödel t-norm ${}^{6}$ . If we adopt the harmonic-mean-based loss function we can see a statistical improvement of the performance of both ${\mathcal{T}}_{\text{ expl }}$ (hmean) and ${\mathcal{T}}_{\text{ prior }}$ (hmean). These results are further statistically improved by adding the new features, see results for ${\mathcal{T}}_{\text{ expl }}$ (hmean,newfeats) and ${\mathcal{T}}_{\text{ prior }}$ (hmean,newfeats). This novelty is fundamental as it allows us to prove the contribution of the logical constraints. Indeed, ${\mathcal{T}}_{\text{ prior }}$ has statistically better performance (in bold) of ${\mathcal{T}}_{\text{ expl }}$ in the predicate detection. In the other tasks the improvement can be statistically observed for the recall@50, indeed the logical constraints improve the precision of the system. The introduction of the new joint features and the new loss function gives an improvement of the recall of approximately 9 points for the predicate detection task and of approximately 5 points for the other tasks.
表 I 显示了消融研究的结果。我们分别对 ${\mathcal{T}}_{\text{ expl }}$ 和 ${\mathcal{T}}_{\text{ prior }}$ 进行了 10 次训练，各获得 10 个模型。对于每个任务和每个扎根理论，我们报告了这些模型所得结果的平均值和标准差。此外，我们进行了 p 值为 0.05 的 t 检验，以对每个任务中的 LTNs 模型进行统计比较。第一批 LTNs 模型使用不含新联合特征的经典 LTNs 损失函数。如果添加新联合特征，它们在所有任务中的性能在统计上提升了约 2 个百分点。然而，如第 IV-C 节所述，最小 t-范数（即 Gödel t-范数）会导致优化过程陷入局部最优，从而使逻辑约束的效果消失。事实上，通过采用 Gödel t-范数 ${}^{6}$ ，${\mathcal{T}}_{\text{ expl }}$ 和 ${\mathcal{T}}_{\text{ prior }}$ 具有相似的性能。如果我们采用基于调和平均数的损失函数，可以看到 ${\mathcal{T}}_{\text{ expl }}$ (hmean) 和 ${\mathcal{T}}_{\text{ prior }}$ (hmean) 的性能都有统计学意义上的提升。通过添加新特征，这些结果得到了进一步的统计提升，参见 ${\mathcal{T}}_{\text{ expl }}$ (hmean,newfeats) 和 ${\mathcal{T}}_{\text{ prior }}$ (hmean,newfeats) 的结果。这一创新至关重要，因为它允许我们证明逻辑约束的贡献。事实上，${\mathcal{T}}_{\text{ prior }}$ 在谓词检测方面的性能（加粗部分）在统计上优于 ${\mathcal{T}}_{\text{ expl }}$ 。在其他任务中，可以在 recall@50 中观察到统计学上的提升，事实上逻辑约束提高了系统的精度。引入新的联合特征和新的损失函数使谓词检测任务的召回率提高了约 9 个百分点，其他任务提高了约 5 个百分点。


---



${}^{5}$ https://github.com/Prof-Lu-Cewu/Visual-Relationship-Detection
${}^{5}$ https://github.com/Prof-Lu-Cewu/Visual-Relationship-Detection


${}^{6}$ The other t-norms lead to a non-converging optimization process due to the numerical issues mentioned in Section IV-C.
${}^{6}$ 由于第 IV-C 节中提到的数值问题，其他 t-范数会导致优化过程不收敛。


---



TABLE I



ABLATION STUDIES FOR THE LTNS MODELS. THE COMBINATION OF THE HARMONIC-MEAN-BASED LOSS FUNCTION (HMEAN) AND THE NEW FEATURES (NEWFEATS) LEADS TO THE BEST RESULTS FOR BOTH ${\mathcal{T}}_{\text{ expl }}$ AND ${\mathcal{T}}_{\text{ prior }}$ .
LTNS 模型的消融研究。基于调和平均数的损失函数 (HMEAN) 与新特征 (NEWFEATS) 的结合使 ${\mathcal{T}}_{\text{ expl }}$ 和 ${\mathcal{T}}_{\text{ prior }}$ 均获得了最佳结果。


<table><tr><td>Task</td><td>Phrase Det.</td><td>Phrase Det.</td><td>Relationship Det.</td><td>Relationship Det.</td><td>Predicate Det.</td><td>Predicate Det.</td></tr><tr><td>Evaluation</td><td>R@100</td><td>R@50</td><td>R@100</td><td>R@50</td><td>R@100</td><td>R@50</td></tr><tr><td>${\mathcal{T}}_{\text{ expl }}$ (min)</td><td>10.04 ± 0.46</td><td>6.15 ± 0.28</td><td>9.11 ± 0.41</td><td>5.56 ± 0.24</td><td>68.06 ± 0.67</td><td>48.80 ± 1.04</td></tr><tr><td>${\mathcal{T}}_{\text{ prior }}$ (min)</td><td>10.21 ± 0.63</td><td>6.29 ± 0.45</td><td>9.31 ± 0.58</td><td>5.70 ± 0.39</td><td>67.83 ± 0.80</td><td>48.41 ± 0.84</td></tr><tr><td>${\mathcal{T}}_{\text{ expl }}$ (min,newfeats)</td><td>12.24 ± 0.36</td><td>${8.35} \pm  {0.51}$</td><td>11.20 ± 0.38</td><td>7.60 ± 0.46</td><td>69.91 ± 0.78</td><td>51.40 ± 0.76</td></tr><tr><td>${\mathcal{T}}_{\text{ prior }}$ (min,newfeats)</td><td>11.92 ± 0.34</td><td>${8.15} \pm  {0.45}$</td><td>10.90 ± 0.34</td><td>7.47 ± 0.44</td><td>69.83 ± 0.72</td><td>51.22 ± 1.01</td></tr><tr><td>${\mathcal{T}}_{\text{ expl }}$ (hmean)</td><td>13.35 ± 0.56</td><td>${8.97} \pm  {0.50}$</td><td>12.22 ± 0.53</td><td>8.12 ± 0.46</td><td>71.11 ± 0.75</td><td>51.36 ± 0.91</td></tr><tr><td>${\mathcal{T}}_{\text{ prior }}$ (hmean)</td><td>13.63 ± 0.41</td><td>${9.25} \pm  {0.45}$</td><td>12.52 ± 0.46</td><td>8.49 ± 0.44</td><td>72.68 ± 0.52</td><td>52.78 ± 0.36</td></tr><tr><td>${\mathcal{T}}_{\text{ expl }}$ (hmean,newfeats)</td><td>15.91 ± 0.54</td><td>11.40 ± 0.34</td><td>14.65 ± 0.54</td><td>10.01 ± 0.38</td><td>74.71 ± 0.73</td><td>56.25 ± 0.75</td></tr><tr><td>${\mathcal{T}}_{\text{ prior }}$ (hmean,newfeats)</td><td>15.74 ± 0.44</td><td>11.40 ± 0.31</td><td>14.43 ± 0.41</td><td>10.47 ± 0.32</td><td>77.16 ± 0.60</td><td>57.34 ± 0.78</td></tr></table>
<table><tbody><tr><td>任务</td><td>短语检测</td><td>短语检测</td><td>关系检测</td><td>关系检测</td><td>谓语检测</td><td>谓语检测</td></tr><tr><td>评估</td><td>R@100</td><td>R@50</td><td>R@100</td><td>R@50</td><td>R@100</td><td>R@50</td></tr><tr><td>${\mathcal{T}}_{\text{ expl }}$ (最小)</td><td>10.04 ± 0.46</td><td>6.15 ± 0.28</td><td>9.11 ± 0.41</td><td>5.56 ± 0.24</td><td>68.06 ± 0.67</td><td>48.80 ± 1.04</td></tr><tr><td>${\mathcal{T}}_{\text{ prior }}$ (最小)</td><td>10.21 ± 0.63</td><td>6.29 ± 0.45</td><td>9.31 ± 0.58</td><td>5.70 ± 0.39</td><td>67.83 ± 0.80</td><td>48.41 ± 0.84</td></tr><tr><td>${\mathcal{T}}_{\text{ expl }}$ (最小,新特征)</td><td>12.24 ± 0.36</td><td>${8.35} \pm  {0.51}$</td><td>11.20 ± 0.38</td><td>7.60 ± 0.46</td><td>69.91 ± 0.78</td><td>51.40 ± 0.76</td></tr><tr><td>${\mathcal{T}}_{\text{ prior }}$ (最小,新特征)</td><td>11.92 ± 0.34</td><td>${8.15} \pm  {0.45}$</td><td>10.90 ± 0.34</td><td>7.47 ± 0.44</td><td>69.83 ± 0.72</td><td>51.22 ± 1.01</td></tr><tr><td>${\mathcal{T}}_{\text{ expl }}$ (调和平均)</td><td>13.35 ± 0.56</td><td>${8.97} \pm  {0.50}$</td><td>12.22 ± 0.53</td><td>8.12 ± 0.46</td><td>71.11 ± 0.75</td><td>51.36 ± 0.91</td></tr><tr><td>${\mathcal{T}}_{\text{ prior }}$ (调和平均)</td><td>13.63 ± 0.41</td><td>${9.25} \pm  {0.45}$</td><td>12.52 ± 0.46</td><td>8.49 ± 0.44</td><td>72.68 ± 0.52</td><td>52.78 ± 0.36</td></tr><tr><td>${\mathcal{T}}_{\text{ expl }}$ (调和平均,新特征)</td><td>15.91 ± 0.54</td><td>11.40 ± 0.34</td><td>14.65 ± 0.54</td><td>10.01 ± 0.38</td><td>74.71 ± 0.73</td><td>56.25 ± 0.75</td></tr><tr><td>${\mathcal{T}}_{\text{ prior }}$ (调和平均,新特征)</td><td>15.74 ± 0.44</td><td>11.40 ± 0.31</td><td>14.43 ± 0.41</td><td>10.47 ± 0.32</td><td>77.16 ± 0.60</td><td>57.34 ± 0.78</td></tr></tbody></table>


## B. Comparison with the State-of-the-Art
## B. 与先进技术的比较


Table II shows the LTNs results compared with the state-of-the-art. The phrase and the relationship detection tasks are the hardest tasks, as they include also the detection of the bounding boxes of the subject/object. Therefore, the errors coming from the object detector propagate also to the visual relationship detection models. Adopting the same object detector used by VRD, [2] and WeaklySup allows us to compare LTNs results starting from the same level of error coming from the bounding boxes detection. The ${\mathcal{T}}_{\text{ expl }}$ and ${\mathcal{T}}_{\text{ prior }}$ models outperform these competitors showing that LTNs deal with the object detection errors in a better way. The predicate detection task, instead, is easier as it is independent from object detection. In this task, it is possible to see all the effectiveness of LTNs due to the good improvement of performance. Regarding the other competitors, a fully comparison is possible only in the predicate detection task as in the other tasks they use a different object detector for bounding boxes detection. However, both LTNs models effectively exploit the structure of the data to infer similarity between relationships and outperform VRL and Context-AwareVRD. Moreover,the LTNs model ${\mathcal{T}}_{\text{ prior }}$ trained with data and constraints outperforms the model ${\mathcal{T}}_{\text{ expl }}$ trained with only data. This means that the negative domain and range constraints are effective at excluding some binary predicates for a bounding box with a given subject or object. For example, if the subject is a physical object, then the predicate cannot be sleep on. This peculiarity of LTNs is fundamental in the comparison with LKD,as ${\mathcal{T}}_{\text{ prior }}$ achieves the improvement that outperforms LKD on predicate detection. The performance on the other tasks are comparable even if the data coming from the object detection are different. Indeed, LKD uses a more recent object detector with better performance than the one provided by VRD. LKD exploits statistical dependencies (co-occurrences) between subjects/objects and relationships, whereas LTNs exploit logical knowledge. This can express more information (e.g., positive or negative dependencies between subjects/objects and relationships, properties of the relationships or relationships) and allows a more accurate reasoning on the visual relationships. Moreover, LKD has to predict $\mathcal{O}\left( {{\left| {\mathcal{P}}_{1}\right| }^{2}\left| {\mathcal{P}}_{2}\right| }\right)$ possible relationships,whereas the searching space of LTNs is $\mathcal{O}\left( {\left| {\mathcal{P}}_{1}\right|  + \left| {\mathcal{P}}_{2}\right| }\right)$ . This implies an important reduction of the parameters of the method and a substantial advantage on scalability.
表 II 显示了 LTNs 与先进技术的对比结果。短语和关系检测任务最为困难，因为它们还包括对主语/宾语边界框的检测。因此，来自目标检测器的错误也会传播到视觉关系检测模型。通过采用与 VRD、[2] 和 WeaklySup 相同的目标检测器，我们可以对比 LTNs 在相同边界框检测误差水平下的结果。${\mathcal{T}}_{\text{ expl }}$ 和 ${\mathcal{T}}_{\text{ prior }}$ 模型优于这些竞争对手，表明 LTNs 能更好地处理目标检测误差。相比之下，谓词检测任务更简单，因为它独立于目标检测。在该任务中，由于性能的大幅提升，可以看到 LTNs 的全部效力。对于其他竞争对手，仅在谓词检测任务中可以进行全面比较，因为在其他任务中，他们使用了不同的目标检测器进行边界框检测。然而，两种 LTNs 模型都有效地利用了数据结构来推断关系之间的相似性，并优于 VRL 和 Context-AwareVRD。此外，结合数据和约束训练的 LTNs 模型 ${\mathcal{T}}_{\text{ prior }}$ 优于仅用数据训练的模型 ${\mathcal{T}}_{\text{ expl }}$。这意味着负向定义域和值域约束能有效地为给定主语或宾语的边界框排除某些二元谓词。例如，如果主语是物理对象，则谓词不能是“sleep on”。LTNs 的这一特性在与 LKD 的比较中至关重要，因为 ${\mathcal{T}}_{\text{ prior }}$ 在谓词检测上的提升使其超越了 LKD。尽管来自目标检测的数据不同，其他任务的性能仍具有可比性。事实上，LKD 使用了比 VRD 提供的更新、性能更好的目标检测器。LKD 利用主语/宾语与关系之间的统计依赖（共现），而 LTNs 利用逻辑知识。这可以表达更多信息（例如，主语/宾语与关系之间的正向或负向依赖、关系或关系集的属性），并允许对视觉关系进行更准确的推理。此外，LKD 必须预测 $\mathcal{O}\left( {{\left| {\mathcal{P}}_{1}\right| }^{2}\left| {\mathcal{P}}_{2}\right| }\right)$ 种可能的关系，而 LTNs 的搜索空间为 $\mathcal{O}\left( {\left| {\mathcal{P}}_{1}\right|  + \left| {\mathcal{P}}_{2}\right| }\right)$。这意味着该方法参数显著减少，且在可扩展性方面具有实质性优势。


These results prove two statements: (i) the general LTNs ability in the visual relationship detection problem to generalize to never seen relationships; (ii) the LTNs ability to leverage logical constraints in the background knowledge to improve the results in a zero-shot learning scenario. This important achievement states that it possible to use the logical constraints to compensate the lack of information in the datasets due to the effort of the annotation. These exploited logical constraints are general and can be retrieved from on-line linguistic resources.
这些结果证明了两个论点：(i) LTNs 在视觉关系检测问题中对未见关系进行泛化的通用能力；(ii) LTNs 在零样本学习场景下利用背景知识中的逻辑约束来改进结果的能力。这一重要成就表明，可以利用逻辑约束来补偿由于标注工作量导致的数据集信息缺失。这些被利用的逻辑约束具有通用性，可以从在线语言资源中获取。


## VI. CONCLUSIONS AND FUTURE WORK
## VI. 结论与未来工作


The zero-shot learning in SII is the problem of detecting visual relationships whose instances do not appear in a training set. This is an emerging problem in AI datasets due to the high annotation effort and their consequent incompleteness. Our proposal is based on Logic Tensor Networks that are able to learn the similarity with other seen triples in presence of logical background knowledge. The results on the Visual Relationship Dataset show that the jointly use of data and logical constraints outperforms the state-of-the-art methods. The rationale is that the logical constraints explicitly state the relations between a given relationship and its subject/object. Therefore, logical knowledge can compensate at the incompleteness of annotation in datasets. In addition, some ablation studies prove the effectiveness of the introduced novelties with respect to the standard LTNs framework. As future work, we plan to apply LTNs to the Visual Genome dataset [14]. We also plan to study the performance by using different categories of constraints. This allows us to check if some constraints more effective than others. In addition, we want to check the robustness of a SII system given by the logical constraints. This can be achieved by training with an increasing amount of flipped labels in the training data as performed during a poisoning attack in adversarial learning [4].
SII 中的零样本学习是指检测其实例未在训练集中出现的视觉关系的问题。由于高昂的标注成本及其导致的不完整性，这是 AI 数据集中一个新兴的问题。我们的提议基于逻辑张量网络（LTNs），它能够在存在逻辑背景知识的情况下，学习与其他已见三元组的相似性。在 Visual Relationship Dataset 上的结果表明，结合使用数据和逻辑约束的效果优于先进技术方法。其原理是，逻辑约束明确阐述了特定关系与其主语/宾语之间的联系。因此，逻辑知识可以弥补数据集中标注的不完整性。此外，一些消融实验证明了所引入的创新点相对于标准 LTNs 框架的有效性。作为未来工作，我们计划将 LTNs 应用于 Visual Genome 数据集 [14]。我们还计划通过使用不同类别的约束来研究性能。这将使我们能够检查某些约束是否比其他约束更有效。此外，我们希望检查逻辑约束赋予 SII 系统的鲁棒性。这可以通过在训练数据中增加翻转标签的数量进行训练来实现，类似于对抗学习中投毒攻击的过程 [4]。


## REFERENCES
## 参考文献


[1] Jamal Atif, Céline Hudelot, and Isabelle Bloch. Explanatory reasoning for image understanding using formal concept analysis and description logics. IEEE Trans. Systems, Man, and Cybernetics: Systems, 44(5):552- 570, 2014.
[1] Jamal Atif, Céline Hudelot, and Isabelle Bloch. 利用形式概念分析与描述逻辑进行图像理解的解释性推理。IEEE Trans. Systems, Man, and Cybernetics: Systems, 44(5):552- 570, 2014.


TABLE II



RESULTS ON THE VISUAL RELATIONSHIP DATASET (R@N STANDS FOR RECALL AT N). THE USE OF THE LOGICAL CONSTRAINTS IN ${\mathcal{T}}_{\text{ prior }}$ LEADS TO THE BEST RESULTS AND OUTPERFORMS THE STATE-OF-THE-ART IN THE PREDICATE DETECTION TASK.
视觉关系数据集上的结果（R@N 代表 N 处的召回率）。在 ${\mathcal{T}}_{\text{ prior }}$ 中使用逻辑约束取得了最佳结果，并在谓词检测任务中超越了现有最先进技术。


<table><tr><td>Task</td><td>Phrase Det.</td><td>Phrase Det.</td><td>Relationship Det.</td><td>Relationship Det.</td><td>Predicate Det.</td><td>Predicate Det.</td></tr><tr><td>Evaluation</td><td>R@100</td><td>R@50</td><td>R@100</td><td>R@50</td><td>R@100</td><td>R@50</td></tr><tr><td>VRD [19]</td><td>3.75</td><td>3.36</td><td>3.52</td><td>3.13</td><td>8.45</td><td>8.45</td></tr><tr><td>RESCAL [2]</td><td>6.59</td><td>5.82</td><td>6.07</td><td>5.30</td><td>16.34</td><td>16.34</td></tr><tr><td>MultiwayNN [2]</td><td>6.93</td><td>5.73</td><td>6.24</td><td>5.22</td><td>16.60</td><td>16.60</td></tr><tr><td>ComplEx [2]</td><td>6.50</td><td>5.73</td><td>5.82</td><td>5.05</td><td>15.74</td><td>15.74</td></tr><tr><td>DistMult [2]</td><td>4.19</td><td>3.34</td><td>3.85</td><td>3.08</td><td>12.40</td><td>12.40</td></tr><tr><td>${\mathcal{T}}_{\text{ expl }}$ (hmean,newfeats)</td><td>15.91 ± 0.54</td><td>11.00 ± 0.34</td><td>14.65 ± 0.54</td><td>10.01 ± 0.38</td><td>74.71 ± 0.73</td><td>56.25 ± 0.75</td></tr><tr><td>${\mathcal{T}}_{\text{ prior }}$ (hmean,newfeats)</td><td>15.74 ± 0.44</td><td>11.40 ± 0.31</td><td>14.43 ± 0.41</td><td>10.47 ± 0.32</td><td>77.16±0.60</td><td>57.34 ± 0.78</td></tr><tr><td>LKD [31]</td><td>17.24</td><td>12.96</td><td>15.89</td><td>12.02</td><td>74.65</td><td>54.20</td></tr><tr><td>VRL [18]</td><td>10.31</td><td>9.17</td><td>8.52</td><td>7.94</td><td>-</td><td>-</td></tr><tr><td>Context-AwareVRD [35]</td><td>11.30</td><td>10.78</td><td>10.26</td><td>9.54</td><td>16.37</td><td>16.37</td></tr><tr><td>WeaklySup [23]</td><td>7.80</td><td>6.80</td><td>7.40</td><td>6.40</td><td>21.60</td><td>21.60</td></tr></table>
<table><tbody><tr><td>任务</td><td>短语检测</td><td>短语检测</td><td>关系检测</td><td>关系检测</td><td>谓语检测</td><td>谓语检测</td></tr><tr><td>评估</td><td>R@100</td><td>R@50</td><td>R@100</td><td>R@50</td><td>R@100</td><td>R@50</td></tr><tr><td>VRD [19]</td><td>3.75</td><td>3.36</td><td>3.52</td><td>3.13</td><td>8.45</td><td>8.45</td></tr><tr><td>RESCAL [2]</td><td>6.59</td><td>5.82</td><td>6.07</td><td>5.30</td><td>16.34</td><td>16.34</td></tr><tr><td>MultiwayNN [2]</td><td>6.93</td><td>5.73</td><td>6.24</td><td>5.22</td><td>16.60</td><td>16.60</td></tr><tr><td>ComplEx [2]</td><td>6.50</td><td>5.73</td><td>5.82</td><td>5.05</td><td>15.74</td><td>15.74</td></tr><tr><td>DistMult [2]</td><td>4.19</td><td>3.34</td><td>3.85</td><td>3.08</td><td>12.40</td><td>12.40</td></tr><tr><td>${\mathcal{T}}_{\text{ expl }}$ (hmean,新特征)</td><td>15.91 ± 0.54</td><td>11.00 ± 0.34</td><td>14.65 ± 0.54</td><td>10.01 ± 0.38</td><td>74.71 ± 0.73</td><td>56.25 ± 0.75</td></tr><tr><td>${\mathcal{T}}_{\text{ prior }}$ (hmean,新特征)</td><td>15.74 ± 0.44</td><td>11.40 ± 0.31</td><td>14.43 ± 0.41</td><td>10.47 ± 0.32</td><td>77.16±0.60</td><td>57.34 ± 0.78</td></tr><tr><td>LKD [31]</td><td>17.24</td><td>12.96</td><td>15.89</td><td>12.02</td><td>74.65</td><td>54.20</td></tr><tr><td>VRL [18]</td><td>10.31</td><td>9.17</td><td>8.52</td><td>7.94</td><td>-</td><td>-</td></tr><tr><td>上下文感知VRD [35]</td><td>11.30</td><td>10.78</td><td>10.26</td><td>9.54</td><td>16.37</td><td>16.37</td></tr><tr><td>弱监督 [23]</td><td>7.80</td><td>6.80</td><td>7.40</td><td>6.40</td><td>21.60</td><td>21.60</td></tr></tbody></table>


[2] Stephan Baier, Yunpu Ma, and Volker Tresp. Improving visual relationship detection using semantic modeling of scene descriptions. In ISWC (1), pages 53-68. Springer, 2017.
[2] Stephan Baier, Yunpu Ma, and Volker Tresp. 利用场景描述的语义建模改进视觉关系检测. In ISWC (1), pages 53-68. Springer, 2017.


[3] Collin F. Baker, Charles J. Fillmore, and John B. Lowe. The berkeley framenet project. In COLING-ACL, pages 86-90. Morgan Kaufmann Publishers / ACL, 1998.
[3] Collin F. Baker, Charles J. Fillmore, and John B. Lowe. Berkeley FrameNet 项目. In COLING-ACL, pages 86-90. Morgan Kaufmann Publishers / ACL, 1998.


[4] Battista Biggio, Blaine Nelson, and Pavel Laskov. Poisoning attacks against support vector machines. In ICML. icml.cc / Omnipress, 2012.
[4] Battista Biggio, Blaine Nelson, and Pavel Laskov. 针对支持向量机的投毒攻击. In ICML. icml.cc / Omnipress, 2012.


[5] Isabelle Bloch. Fuzzy spatial relationships for image processing and interpretation: a review. Image Vision Comput., 23(2):89-110, 2005.
[5] Isabelle Bloch. 用于图像处理和解释的模糊空间关系：综述. Image Vision Comput., 23(2):89-110, 2005.


[6] Na Chen, Qian-Yi Zhou, and Viktor K. Prasanna. Understanding web images by object relation network. In WWW, pages 291-300. ACM, 2012.
[6] Na Chen, Qian-Yi Zhou, and Viktor K. Prasanna. 通过对象关系网络理解网络图像. In WWW, pages 291-300. ACM, 2012.


[7] Xianjie Chen, Roozbeh Mottaghi, Xiaobai Liu, Sanja Fidler, Raquel Urtasun, and Alan L. Yuille. Detect what you can: Detecting and representing objects using holistic models and body parts. In CVPR, pages 1979-1986. IEEE Computer Society, 2014.
[7] Xianjie Chen, Roozbeh Mottaghi, Xiaobai Liu, Sanja Fidler, Raquel Urtasun, and Alan L. Yuille. 检测能力所及：使用整体模型和身体部位检测并表示对象. In CVPR, pages 1979-1986. IEEE Computer Society, 2014.


[8] Bo Dai, Yuqi Zhang, and Dahua Lin. Detecting visual relationships with deep relational networks. In CVPR, pages 3298-3308. IEEE Computer Society, 2017.
[8] Bo Dai, Yuqi Zhang, and Dahua Lin. 使用深度关系网络检测视觉关系. In CVPR, pages 3298-3308. IEEE Computer Society, 2017.


[9] Ivan Donadello and Luciano Serafini. Mixing low-level and semantic features for image interpretation - A framework and a simple case study. In ECCV Workshops (2), volume 8926 of Lecture Notes in Computer Science, pages 283-298. Springer, 2014.
[9] Ivan Donadello and Luciano Serafini. 混合低层与语义特征进行图像解释 —— 框架与简单案例研究. In ECCV Workshops (2), volume 8926 of Lecture Notes in Computer Science, pages 283-298. Springer, 2014.


[10] Ivan Donadello, Luciano Serafini, and Artur S. d'Avila Garcez. Logic tensor networks for semantic image interpretation. In IJCAI, pages 1596-1602. ijcai.org, 2017.
[10] Ivan Donadello, Luciano Serafini, and Artur S. d'Avila Garcez. 用于语义图像解释的逻辑张量网络. In IJCAI, pages 1596-1602. ijcai.org, 2017.


[11] Saurabh Gupta and Jitendra Malik. Visual semantic role labeling. arXiv preprint arXiv:1505.04474, 2015.
[11] Saurabh Gupta and Jitendra Malik. 视觉语义角色标注. arXiv preprint arXiv:1505.04474, 2015.


[12] P. Hájek. Metamathematics of Fuzzy Logic. Trends in Logic. Springer, 2001.
[12] P. Hájek. 模糊逻辑的元数学. Trends in Logic. Springer, 2001.


[13] Céline Hudelot, Jamal Atif, and Isabelle Bloch. Fuzzy spatial relation ontology for image interpretation. Fuzzy Sets and Systems, 159(15):1929-1951, 2008.
[13] Céline Hudelot, Jamal Atif, and Isabelle Bloch. 用于图像解释的模糊空间关系本体. Fuzzy Sets and Systems, 159(15):1929-1951, 2008.


[14] Ranjay Krishna, Yuke Zhu, Oliver Groth, Justin Johnson, Kenji Hata, Joshua Kravitz, Stephanie Chen, Yannis Kalantidis, Li-Jia Li, David A. Shamma, Michael S. Bernstein, and Li Fei-Fei. Visual genome: Connecting language and vision using crowdsourced dense image annotations. International Journal of Computer Vision, 123(1):32-73, 2017.
[14] Ranjay Krishna, Yuke Zhu, Oliver Groth, Justin Johnson, Kenji Hata, Joshua Kravitz, Stephanie Chen, Yannis Kalantidis, Li-Jia Li, David A. Shamma, Michael S. Bernstein, and Li Fei-Fei. Visual Genome：使用众包密集图像注释连接语言与视觉. International Journal of Computer Vision, 123(1):32-73, 2017.


[15] Girish Kulkarni, Visruth Premraj, Sagnik Dhar, Siming Li, Yejin Choi, Alexander C. Berg, and Tamara L. Berg. Baby talk: Understanding and generating simple image descriptions. In CVPR, pages 1601-1608. IEEE Computer Society, 2011.
[15] Girish Kulkarni, Visruth Premraj, Sagnik Dhar, Siming Li, Yejin Choi, Alexander C. Berg, and Tamara L. Berg. 婴儿学语：理解并生成简单的图像描述. In CVPR, pages 1601-1608. IEEE Computer Society, 2011.


[16] Christoph H. Lampert, Hannes Nickisch, and Stefan Harmeling. Attribute-based classification for zero-shot visual object categorization. IEEE Trans. Pattern Anal. Mach. Intell., 36(3):453-465, 2014.
[16] Christoph H. Lampert, Hannes Nickisch, and Stefan Harmeling. 用于零样本视觉对象分类的基于属性的分类. IEEE Trans. Pattern Anal. Mach. Intell., 36(3):453-465, 2014.


[17] Yikang Li, Wanli Ouyang, Xiaogang Wang, and Xiaoou Tang. Vip-cnn: Visual phrase guided convolutional neural network. In CVPR, pages 7244-7253. IEEE Computer Society, 2017.
[17] Yikang Li, Wanli Ouyang, Xiaogang Wang, and Xiaoou Tang. ViP-CNN：视觉短语引导的卷积神经网络. In CVPR, pages 7244-7253. IEEE Computer Society, 2017.


[18] Xiaodan Liang, Lisa Lee, and Eric P. Xing. Deep variation-structured reinforcement learning for visual relationship and attribute detection. In CVPR, pages 4408-4417. IEEE Computer Society, 2017.
[18] Xiaodan Liang, Lisa Lee, and Eric P. Xing. 基于深度变分结构强化学习的视觉关系与属性检测. In CVPR, pages 4408-4417. IEEE Computer Society, 2017.


[19] Cewu Lu, Ranjay Krishna, Michael S. Bernstein, and Fei-Fei Li. Visual relationship detection with language priors. In ECCV (1), volume 9905 of LNCS, pages 852-869. Springer, 2016.
[19] Cewu Lu, Ranjay Krishna, Michael S. Bernstein, and Fei-Fei Li. 基于语言先验的视觉关系检测. In ECCV (1), volume 9905 of LNCS, pages 852-869. Springer, 2016.


[20] Bernd Neumann and Ralf Möller. On scene interpretation with description logics. Image Vision Comput., 26(1):82-101, 2008.
[20] Bernd Neumann and Ralf Möller. 论基于描述逻辑的场景理解. Image Vision Comput., 26(1):82-101, 2008.


[21] Maximilian Nickel, Kevin Murphy, Volker Tresp, and Evgeniy Gabrilovich. A review of relational machine learning for knowledge graphs. Proceedings of the IEEE, 104(1):11-33, 2016.
[21] Maximilian Nickel, Kevin Murphy, Volker Tresp, and Evgeniy Gabrilovich. 知识图谱关系机器学习综述. Proceedings of the IEEE, 104(1):11-33, 2016.


[22] Irma Sofía Espinosa Peraldí, Atila Kaya, and Ralf Möller. Formalizing multimedia interpretation based on abduction over description logic aboxes. In Description Logics, volume 477 of CEUR Workshop Proceedings. CEUR-WS.org, 2009.
[22] Irma Sofía Espinosa Peraldí, Atila Kaya, and Ralf Möller. 基于描述逻辑 ABox 溯因推理的多媒体理解形式化. In Description Logics, volume 477 of CEUR Workshop Proceedings. CEUR-WS.org, 2009.


[23] Julia Peyre, Ivan Laptev, Cordelia Schmid, and Josef Sivic. Weakly-supervised learning of visual relations. In ICCV, pages 5189-5198. IEEE Computer Society, 2017.
[23] Julia Peyre, Ivan Laptev, Cordelia Schmid, and Josef Sivic. 视觉关系的弱监督学习. In ICCV, pages 5189-5198. IEEE Computer Society, 2017.


[24] Vignesh Ramanathan, Congcong Li, Jia Deng, Wei Han, Zhen Li, Kunlong Gu, Yang Song, Samy Bengio, Chuck Rosenberg, and Fei-Fei Li. Learning semantic relationships for better action retrieval in images. In CVPR, pages 1100-1109. IEEE Computer Society, 2015.
[24] Vignesh Ramanathan, Congcong Li, Jia Deng, Wei Han, Zhen Li, Kunlong Gu, Yang Song, Samy Bengio, Chuck Rosenberg, and Fei-Fei Li. 学习语义关系以优化图像中的动作检索. In CVPR, pages 1100-1109. IEEE Computer Society, 2015.


[25] Mohammad Amin Sadeghi and Ali Farhadi. Recognition using visual phrases. In CVPR, pages 1745-1752. IEEE, 2011.
[25] Mohammad Amin Sadeghi and Ali Farhadi. 基于视觉短语的识别. In CVPR, pages 1745-1752. IEEE, 2011.


[26] Karin Kipper Schuler. Verbnet: A Broad-coverage, Comprehensive Verb Lexicon. PhD thesis, University of Pennsylvania, Philadelphia, PA, USA, 2005. AAI3179808.
[26] Karin Kipper Schuler. Verbnet：广覆盖、综合性的动词词典. 博士论文, University of Pennsylvania, Philadelphia, PA, USA, 2005. AAI3179808.


[27] Luciano Serafini and Artur S. d'Avila Garcez. Learning and reasoning with logic tensor networks. In $A{I}^{ * }{IA}$ ,volume 10037 of ${LNCS}$ ,pages 334-348. Springer, 2016.
[27] Luciano Serafini and Artur S. d'Avila Garcez. 基于逻辑张量网络的学习与推理. In $A{I}^{ * }{IA}$ ,volume 10037 of ${LNCS}$ ,pages 334-348. Springer, 2016.


[28] Luciano Serafini, Ivan Donadello, and Artur S. d'Avila Garcez. Learning and reasoning in logic tensor networks: theory and application to semantic image interpretation. In SAC, pages 125-130. ACM, 2017.
[28] Luciano Serafini, Ivan Donadello, and Artur S. d'Avila Garcez. 逻辑张量网络中的学习与推理：理论及其在语义图像理解中的应用. In SAC, pages 125-130. ACM, 2017.


[29] Danfei Xu, Yuke Zhu, Christopher B. Choy, and Li Fei-Fei. Scene graph generation by iterative message passing. In CVPR. IEEE Computer Society, July 2017.
[29] Danfei Xu, Yuke Zhu, Christopher B. Choy, and Li Fei-Fei. 基于迭代消息传递的场景图生成. In CVPR. IEEE Computer Society, July 2017.


[30] Guojun Yin, Lu Sheng, Bin Liu, Nenghai Yu, Xiaogang Wang, Jing Shao, and Chen Change Loy. Zoom-net: Mining deep feature interactions for visual relationship recognition. arXiv preprint arXiv:1807.04979, 2018.
[30] Guojun Yin, Lu Sheng, Bin Liu, Nenghai Yu, Xiaogang Wang, Jing Shao, and Chen Change Loy. Zoom-net：挖掘用于视觉关系识别的深度特征交互. arXiv preprint arXiv:1807.04979, 2018.


[31] Ruichi Yu, Ang Li, Vlad I. Morariu, and Larry S. Davis. Visual relationship detection with internal and external linguistic knowledge distillation. In ICCV, pages 1068-1076. IEEE Computer Society, 2017.
[31] Ruichi Yu, Ang Li, Vlad I. Morariu, and Larry S. Davis. 基于内外语言知识蒸馏的视觉关系检测. In ICCV, pages 1068-1076. IEEE Computer Society, 2017.


[32] Rowan Zellers, Mark Yatskar, Sam Thomson, and Yejin Choi. Neural motifs: Scene graph parsing with global context. In CVPR, pages 5831- 5840. IEEE Computer Society, 2018.
[32] Rowan Zellers, Mark Yatskar, Sam Thomson, and Yejin Choi. 神经基元：基于全局上下文的场景图解析. In CVPR, pages 5831- 5840. IEEE Computer Society, 2018.


[33] Hanwang Zhang, Zawlin Kyaw, Shih-Fu Chang, and Tat-Seng Chua. Visual translation embedding network for visual relation detection. In CVPR, pages 3107-3115. IEEE Computer Society, 2017.
[33] Hanwang Zhang, Zawlin Kyaw, Shih-Fu Chang, and Tat-Seng Chua. 用于视觉关系检测的视觉翻译嵌入网络. In CVPR, pages 3107-3115. IEEE Computer Society, 2017.


[34] Hanwang Zhang, Zawlin Kyaw, Jinyang Yu, and Shih-Fu Chang. PPR-FCN: weakly supervised visual relation detection via parallel pairwise R-FCN. In ICCV, pages 4243-4251. IEEE Computer Society, 2017.
[34] Hanwang Zhang, Zawlin Kyaw, Jinyang Yu, and Shih-Fu Chang. PPR-FCN: 通过并行成对 R-FCN 进行弱监督视觉关系检测. In ICCV, pages 4243-4251. IEEE Computer Society, 2017.


[35] Bohan Zhuang, Lingqiao Liu, Chunhua Shen, and Ian D. Reid. Towards context-aware interaction recognition for visual relationship detection. In ICCV, pages 589-598. IEEE Computer Society, 2017.
[35] Bohan Zhuang, Lingqiao Liu, Chunhua Shen, and Ian D. Reid. 迈向用于视觉关系检测的上下文感知交互识别. In ICCV, pages 589-598. IEEE Computer Society, 2017.