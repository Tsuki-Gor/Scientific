# Logic Tensor Networks: Deep Learning and Logical Reasoning from Data and Knowledge*
# 逻辑张量网络：基于数据与知识的深度学习与逻辑推理*


Luciano Serafini ${}^{1}$ and Artur d’Avila Garcez ${}^{2}$
Luciano Serafini ${}^{1}$ 与 Artur d’Avila Garcez ${}^{2}$


${}^{1}$ Fondazione Bruno Kessler,Trento,Italy,serafini@fbk.eu ${}^{2}$ City University London,UK,a.garcez@city.ac.uk
${}^{1}$ Fondazione Bruno Kessler, Trento, Italy, serafini@fbk.eu ${}^{2}$ City University London, UK, a.garcez@city.ac.uk


Abstract. We propose Logic Tensor Networks: a uniform framework for integrating automatic learning and reasoning. A logic formalism called Real Logic is defined on a first-order language whereby formulas have truth-value in the interval $\left\lbrack  {0,1}\right\rbrack$ and semantics defined concretely on the domain of real numbers. Logical constants are interpreted as feature vectors of real numbers. Real Logic promotes a well-founded integration of deductive reasoning on a knowledge-base and efficient data-driven relational machine learning. We show how Real Logic can be implemented in deep Tensor Neural Networks with the use of Google's TEN-SORFLOW ${}^{\mathrm{{TM}}}$ primitives. The paper concludes with experiments applying Logic Tensor Networks on a simple but representative example of knowledge completion.
摘要。我们提出逻辑张量网络：一种将自动学习与推理整合的统一框架。一个名为 Real Logic 的逻辑形式在一阶语言上被定义，其公式在区间 $\left\lbrack  {0,1}\right\rbrack$ 取值，且语义在实数域上具体定义。逻辑常量被解释为实数特征向量。Real Logic 提倡在知识库的演绎推理与基于数据的高效关系型机器学习之间进行良性融合。我们展示了如何通过使用 Google 的 TEN-SORFLOW ${}^{\mathrm{{TM}}}$ 原语在深度张量神经网络中实现 Real Logic。本文以一个简单但具代表性的知识补全示例，给出 Logic Tensor Networks 的实验结论。


Keywords: Knowledge Representation, Relational Learning, Tensor Networks, Neural-Symbolic Computation, Data-driven Knowledge Completion.
关键词：知识表示、关系学习、张量网络、神经符号计算、基于数据的知识补全。


## 1 Introduction
## 1 介绍


The recent availability of large-scale data combining multiple data modalities, such as image, text, audio and sensor data, has opened up various research and commercial opportunities, underpinned by machine learning methods and techniques [5, 12, 17,18]. In particular, recent work in machine learning has sought to combine logical services, such as knowledge completion, approximate inference, and goal-directed reasoning with data-driven statistical and neural network-based approaches. We argue that there are great possibilities for improving the current state of the art in machine learning and artificial intelligence (AI) thought the principled combination of knowledge representation, reasoning and learning. Guha's recent position paper [15] is a case in point, as it advocates a new model theory for real-valued numbers. In this paper, we take inspiration from such recent work in AI, but also less recent work in the area of neural-symbolic integration $\left\lbrack  {8,{10},{11}}\right\rbrack$ and in semantic attachment and symbol grounding [4] to achieve a vector-based representation which can be shown adequate for integrating machine learning and reasoning in a principled way.
近来大规模数据的可获得性，涵盖图像、文本、音频与传感数据等多模态数据，促成了多种研究与商业机会，这一切都以机器学习方法与技术为支撑 [5, 12, 17,18]。具体而言，近期的机器学习工作旨在将逻辑服务如知识补全、近似推理与目标导向推理，与基于数据统计与神经网络的方法结合起来。我们认为通过知识表示、推理与学习的 principled 融合，可以显著提升当前机器学习与人工智能（AI）的水平。Guha 的近期立场论文 [15] 就是一个典型例子，倡导一种对实值数的新的模型理论。在本文中，我们受到 AI 领域此类近期工作的启发，同时也借鉴在神经符号整合 $\left\lbrack  {8,{10},{11}}\right\rbrack$ 与语义附着与符号着落 [4] 方面的较早工作，以实现向量化表示，从而在 principled 的方式下实现机器学习与推理的整合。


---



* The first author acknowledges the Mobility Program of FBK, for supporting a long term visit at City University London. He also acknowledges NVIDIA Corporation for supporting this research with the donation of a GPU.
* 第一作者感谢 FBK 的 Mobility Program，资助其在伦敦城市大学的长期访问。他还感谢 NVIDIA 公司通过捐赠 GPU 对本研究的支持。


---



This paper proposes a framework called Logic Tensor Networks (LTN) which integrates learning based on tensor networks [26] with reasoning using first-order many-valued logic [6], all implemented in TENSORFLOW™ [13]. This enables, for the first time, a range of knowledge-based tasks using rich knowledge representation in first-order logic (FOL) to be combined with efficient data-driven machine learning based on the manipulation of real-valued vectors ${}^{1}$ . Given data available in the form of real-valued vectors, logical soft and hard constraints and relations which apply to certain subsets of the vectors can be specified compactly in first-order logic. Reasoning about such constraints can help improve learning, and learning from new data can revise such constraints thus modifying reasoning. An adequate vector-based representation of the logic, first proposed in this paper, enables the above integration of learning and reasoning, as detailed in what follows.
本文提出一个名为 Logic Tensor Networks (LTN) 的框架，将基于张量网络的学习 [26] 与使用一阶多值逻辑 [6] 的推理相结合，均在 TENSORFLOW™ [13] 中实现。首次实现了一系列基于知识的任务，使用一阶逻辑中的丰富知识表示，与基于数据的高效机器学习相结合，操作的是实值向量 ${}^{1}$。给定以实值向量形式可用的数据，逻辑中的软约束与硬约束以及应用于向量子集的关系可以在一阶逻辑中紧凑地表示。对此类约束的推理有助于改进学习，而对新数据的学习又能修正这些约束，从而修改推理。本文首次提出的向量化逻辑表示 enables 上述学习与推理的整合，后文将详细阐述。


We are interested in providing a computationally adequate approach to implementing learning and reasoning [28] in an integrated way within an idealized agent. This agent has to manage knowledge about an unbounded, possibly infinite, set of objects $O = \left\{  {{o}_{1},{o}_{2},\ldots }\right\}$ . Some of the objects are associated with a set of quantitative attributes,represented by an $n$ -tuple of real values $\mathcal{G}\left( {o}_{i}\right)  \in  {\mathbb{R}}^{n}$ ,which we call grounding. For example, a person may have a grounding into a 4-tuple containing some numerical representation of the person's name, her height, weight, and number of friends in some social network. Object tuples can participate in a set of relations $\mathcal{R} = \left\{  {{R}_{1},\ldots ,{R}_{k}}\right\}$ , with ${R}_{i} \subseteq  {O}^{\alpha \left( {R}_{i}\right) }$ ,where $\alpha \left( {R}_{i}\right)$ denotes the arity of relation ${R}_{i}$ . We presuppose the existence of a latent (unknown) relation between the above numerical properties, i.e. groundings,and partial relational structure $\mathcal{R}$ on $O$ . Starting from this partial knowledge, an agent is required to: (i) infer new knowledge about the relational structure on the objects of $O$ ; (ii) predict the numerical properties or the class of the objects in $O$ .
我们有兴趣在一个理想化代理内以计算上充足的方式实现学习与推理 [28]，并以整合的方式进行。该代理必须管理关于一个无界、可能无限的对象集合 $O = \left\{  {{o}_{1},{o}_{2},\ldots }\right\}$ 的知识。其中一些对象与一组定量属性相关联，用一个 $n$-元组的实数值 $\mathcal{G}\left( {o}_{i}\right)  \in  {\mathbb{R}}^{n}$ 来表示，我们称之为“接地”。例如，一个人可能在包含该人姓名的数值表示、身高、体重和社交网络中朋友数量的四元组中具有接地。对象元组可以参与一组关系 $\mathcal{R} = \left\{  {{R}_{1},\ldots ,{R}_{k}}\right\}$，与 ${R}_{i} \subseteq  {O}^{\alpha \left( {R}_{i}\right) }$ 相关，其中 $\alpha \left( {R}_{i}\right)$ 表示关系 ${R}_{i}$ 的阶数。我们假设上述数值属性（即接地）与部分关系结构 $\mathcal{R}$ 在 $O$ 上之间存在潜在（未知）关系。基于这部分知识，代理需要：(i) 推断 $O$ 对象的关系结构的新知识；(ii) 预测 $O$ 中对象的数值属性或类别。


Classes and relations are not normally independent. For example, it may be the case that if an object $x$ is of class $C,C\left( x\right)$ ,and it is related to another object $y$ through relation $R\left( {x,y}\right)$ then this other object $y$ should be in the same class $C\left( y\right)$ . In logic: $\forall x\exists y\left( {\left( {C\left( x\right)  \land  R\left( {x,y}\right) }\right)  \rightarrow  C\left( y\right) }\right)$ . Whether or not $C\left( y\right)$ holds will depend on the application: through reasoning,one may derive $C\left( y\right)$ where otherwise there might not have been evidence of $C\left( y\right)$ from training examples only; through learning,one may need to revise such a conclusion once examples to the contrary become available. The vectorial representation proposed in this paper permits both reasoning and learning as exemplified above and detailed in the next section.
类别与关系通常并非独立。例如，如果一个对象 $x$ 属于类 $C,C\left( x\right)$，并且它通过关系 $R\left( {x,y}\right)$ 与另一个对象 $y$ 相连，那么这个对象 $y$ 应该也在同一类 $C\left( y\right)$ 中。在逻辑中：$\forall x\exists y\left( {\left( {C\left( x\right)  \land  R\left( {x,y}\right) }\right)  \rightarrow  C\left( y\right) }\right)$ 。是否成立 $C\left( y\right)$ 将取决于具体应用：通过推理，可能会推导出 $C\left( y\right)$，在训练样本只存在有限证据时本来可能没有证据；通过学习，当出现与之相反的样例时，可能需要修正这样的结论。本文提出的向量化表示同时允许上述推理和学习，且在下一节将给出详细实例。


The above forms of reasoning and learning are integrated in a unifying framework, implemented within tensor networks, and exemplified in relational domains combining data and relational knowledge about the objects. It is expected that, through an adequate integration of numerical properties and relational knowledge, differently from the immediate related literature $\left\lbrack  {9,2,1}\right\rbrack$ ,the framework introduced in this paper will be capable of combining in an effective way first-order logical inference on open domains with efficient relational multi-class learning using tensor networks.
上述形式的推理和学习被整合在一个统一框架中，该框架在张量网络内实现，并在结合对对象进行数据和关系知识的关系域中得到示例。期望通过对数值属性与关系知识的充分整合，与直接相关的文献 $\left\lbrack  {9,2,1}\right\rbrack$ 不同，本文提出的框架将能够以高效的方式将对开放域的一阶逻辑推理与使用张量网络进行的高效关系多类学习结合起来。


The main contribution of this paper is two-fold. It introduces a novel framework for the integration of learning and reasoning which can take advantage of the representational power of (multi-valued) first-order logic, and it instantiates the framework using tensor networks into an efficient implementation which shows that the proposed vector-based representation of the logic offers an adequate mapping between symbols and their real-world manifestations, which is appropriate for both rich inference and learning from examples.
本文的主要贡献有两方面。它提出了一个新颖的学习与推理整合框架，该框架利用（多值）一阶逻辑的表征能力，并通过张量网络将其实例化为高效实现，展示了所提出的基于向量的逻辑表示与符号及其现实世界表现之间的适当映射，这对于丰富的推理与从样例学习都很适用。


---



${}^{1}$ In practice,FOL reasoning including function symbols is approximated through the usual iterative deepening of clause depth.
${}^{1}$ 实践中，包含函数符号的 FOL 推理通过对子句深度的通常迭代性加深来近似。


---



The paper is organized as follows. In Section 2, we define Real Logic. In Section 3, we propose the Learning-as-Inference framework. In Section 4, we instantiate the framework by showing how Real Logic can be implemented in deep Tensor Neural Networks leading to Logic Tensor Networks (LTN). Section 5 contains an example of how LTN handles knowledge completion using (possibly inconsistent) data and knowledge from the well-known smokers and friends experiment. Section 6 concludes the paper and discusses directions for future work.
本文结构如下。第2节定义 Real Logic。第3节提出 Learning-as-Inference 框架。第4节通过展示 Real Logic 如何在深度张量神经网络中实现，从而得到 Logic Tensor Networks (LTN)，对框架进行实例化。第5节给出一个示例，说明 LTN 如何在使用（可能不一致的）来自著名的烟民与朋友实验的数据与知识的知识完成任务中工作。第6节对本文进行总结并讨论未来工作方向。


## 2 Real Logic
## 2 Real Logic


We start from a first order language $\mathcal{L}$ ,whose signature contains a set $\mathcal{C}$ of constant symbols,a set $\mathcal{F}$ of functional symbols,and a set $\mathcal{P}$ of predicate symbols. The sentences of $\mathcal{L}$ are used to express relational knowledge,e.g. the atomic formula $R\left( {{o}_{1},{o}_{2}}\right)$ states that objects ${o}_{1}$ and ${o}_{2}$ are related to each other through binary relation $R;\forall {xy}.(R\left( {x,y}\right)  \rightarrow \; R\left( {y,x}\right) )$ states that $R$ is a symmetric relation,where $x$ and $y$ are variables; $\exists y.R\left( {{o}_{1},y}\right)$ states that there is an (unknown) object which is related to object ${o}_{1}$ through $R$ . For simplicity,without loss of generality,we assume that all logical sentences of $\mathcal{L}$ are in prenex conjunctive,skolemised normal form [16],e.g. a sentence $\forall x\left( {A\left( x\right)  \rightarrow  \exists {yR}\left( {x,y}\right) }\right)$ is transformed into an equivalent clause $\neg A\left( x\right)  \vee  R\left( {x,f\left( x\right) }\right)$ ,where $f$ is a new function symbol.
我们从一阶语言 $\mathcal{L}$ 开始，其签名包含常量符号集合 $\mathcal{C}$、函数符号集合 $\mathcal{F}$、谓词符号集合 $\mathcal{P}$。$\mathcal{L}$ 的句子用于表达关系知识，例如原子公式 $R\left( {{o}_{1},{o}_{2}}\right)$ 表示对象 ${o}_{1}$ 与 ${o}_{2}$ 通过二元关系 $R;\forall {xy}.(R\left( {x,y}\right)  \rightarrow \; R\left( {y,x}\right) )$ 相关联，$R;\forall {xy}.(R\left( {x,y}\right)  \rightarrow \; R\left( {y,x}\right) )$ 表示 $R$ 是一个对称关系，其中 $x$ 与 $y$ 为变量；$\exists y.R\left( {{o}_{1},y}\right)$ 表示存在一个（未知的）对象通过 $R$ 与对象 ${o}_{1}$ 相关联。为简化起见、在不损失一般性的前提下，我们假设 $\mathcal{L}$ 的所有逻辑句子均处于前束合取、Skolem 化的标准形式 [16]，例如句子 $\forall x\left( {A\left( x\right)  \rightarrow  \exists {yR}\left( {x,y}\right) }\right)$ 被变换为等价子句 $\neg A\left( x\right)  \vee  R\left( {x,f\left( x\right) }\right)$，其中 $f$ 是新的函数符号。


As for the semantics of $\mathcal{L}$ ,we deviate from the standard abstract semantics of FOL, and we propose a concrete semantics with sentences interpreted as tuples of real numbers. To emphasise the fact that $\mathcal{L}$ is interpreted in a "real" world,we use the term (semantic) grounding,denoted by $\mathcal{G}$ ,instead of the more standard interpretation ${}^{2}$ .
至于 $\mathcal{L}$ 的语义，我们偏离标准的一阶逻辑抽象语义，提出一种具体语义，其中句子被解释为实数元组。为强调 $\mathcal{L}$ 在一个“真实”世界中的解释，我们使用术语语义性基底（semantic grounding），记为 $\mathcal{G}$，而非更常见的解释 ${}^{2}$。


- $\mathcal{G}$ associates an $n$ -tuple of real numbers $\mathcal{G}\left( t\right)$ to any closed term $t$ of $\mathcal{L}$ ; intuitively $\mathcal{G}\left( t\right)$ is the set of numeric features of the object denoted by $t$ .
- $\mathcal{G}$ 将一个 $n$ -tuple 的实数 $\mathcal{G}\left( t\right)$ 分配给任意闭合项 $t$ 的 $\mathcal{L}$ ; 直观地 $\mathcal{G}\left( t\right)$ 是由 $t$ 所指对象的数值特征集合。


- $\mathcal{G}$ associates a real number in the interval $\left\lbrack  {0,1}\right\rbrack$ to each clause $\phi$ of $\mathcal{L}$ . Intuitively, $\mathcal{G}\left( \phi \right)$ represents one’s confidence in the truth of $\phi$ ; the higher the value,the higher the confidence.
- $\mathcal{G}$ 将区间 $\left\lbrack  {0,1}\right\rbrack$ 内的一个实数与 $\mathcal{L}$ 的每个子句 $\phi$ 相关联。直觉上，$\mathcal{G}\left( \phi \right)$ 表示对 $\phi$ 真值的信心程度；数值越高，信心越高。


A grounding is specified only for the elements of the signature of $\mathcal{L}$ . The grounding of terms and clauses is defined inductively, as follows.
一个基底仅对 $\mathcal{L}$ 的签名元素进行指定。对项与子句的基底按归纳方式定义，如下所述。


Definition 1. A grounding $\mathcal{G}$ for a first order language $\mathcal{L}$ is a function from the signature of $\mathcal{L}$ to the real numbers that satisfies the following conditions:
定义 1。对一阶语言 $\mathcal{L}$ 的基底 $\mathcal{G}$ 是一个从 $\mathcal{L}$ 签名映射到实数的函数，需满足以下条件：


1. $\mathcal{G}\left( c\right)  \in  {\mathbb{R}}^{n}$ for every constant symbol $c \in  \mathcal{C}$ ;
1. 对每个常量符号 $c \in  \mathcal{C}$，有 $\mathcal{G}\left( c\right)  \in  {\mathbb{R}}^{n}$。


2. $\mathcal{G}\left( f\right)  \in  {\mathbb{R}}^{n \cdot  \alpha \left( f\right) } \rightarrow  {\mathbb{R}}^{n}$ for every $f \in  \mathcal{F}$ ;
2. 对每个 $f \in  \mathcal{F}$，有 $\mathcal{G}\left( f\right)  \in  {\mathbb{R}}^{n \cdot  \alpha \left( f\right) } \rightarrow  {\mathbb{R}}^{n}$。


---



${}^{2}$ In logic,the term "grounding" indicates the operation of replacing the variables of a term/formula with constants. To avoid confusion, we use the term "instantiation" for this.
${}^{2}$ 在逻辑中，术语“基底”指将项/公式中的变量替换为常量的运算。为避免混淆，我们将此称为“实例化”。


---



3. $\mathcal{G}\left( P\right)  \in  {\mathbb{R}}^{n \cdot  \alpha \left( R\right) } \rightarrow  \left\lbrack  {0,1}\right\rbrack$ for every $P \in  \mathcal{P}$ ;
3. 对每个 $P \in  \mathcal{P}$，有 $\mathcal{G}\left( P\right)  \in  {\mathbb{R}}^{n \cdot  \alpha \left( R\right) } \rightarrow  \left\lbrack  {0,1}\right\rbrack$。


A grounding $\mathcal{G}$ is inductively extended to all the closed terms and clauses,as follows:
一个 grounding $\mathcal{G}$ 被归结地扩展到所有闭式项和子句，如下所示：


$$
\mathcal{G}\left( {f\left( {{t}_{1},\ldots ,{t}_{m}}\right) }\right)  = \mathcal{G}\left( f\right) \left( {\mathcal{G}\left( {t}_{1}\right) ,\ldots ,\mathcal{G}\left( {t}_{m}\right) }\right)
$$



$$
\mathcal{G}\left( {P\left( {{t}_{1},\ldots ,{t}_{m}}\right) }\right)  = \mathcal{G}\left( P\right) \left( {\mathcal{G}\left( {t}_{1}\right) ,\ldots ,\mathcal{G}\left( {t}_{m}\right) }\right)
$$



$$
\mathcal{G}\left( {\neg P\left( {{t}_{1},\ldots ,{t}_{m}}\right) }\right)  = 1 - \mathcal{G}\left( {P\left( {{t}_{1},\ldots ,{t}_{m}}\right) }\right)
$$



$$
\mathcal{G}\left( {{\phi }_{1} \vee  \cdots  \vee  {\phi }_{k}}\right)  = \mu \left( {\mathcal{G}\left( {\phi }_{1}\right) ,\ldots ,\mathcal{G}\left( {\phi }_{k}\right) }\right)
$$



where $\mu$ is an s-norm operator,also known as a t-co-norm operator (i.e. the dual of some t-norm operator). ${}^{3}$
其中 $\mu$ 是一个 s-范数算子，也称为一个 t-共轭范数算子（即某个 t-范数算子的对偶）。 ${}^{3}$


Example 1. Suppose that $O = \left\{  {{o}_{1},{o}_{2},{o}_{3}}\right\}$ is a set of documents defined on a finite dictionary $D = \left\{  {{w}_{1},\ldots ,{w}_{n}}\right\}$ of $n$ words. Let $\mathcal{L}$ be the language that contains the binary function symbol concat $\left( {x,y}\right)$ denoting the document resulting from the concatenation of documents $x$ with $y$ . Let $\mathcal{L}$ contain also the binary predicate ${Sim}$ which is supposed to be true if document $x$ is deemed to be similar to document $y$ . An example of grounding is the one that associates to each document its bag-of-words vector [7]. As a consequence, a natural grounding of the concat function would be the sum of the vectors, and of the Sim predicate, the cosine similarity between the vectors. More formally:
示例 1。设 $O = \left\{  {{o}_{1},{o}_{2},{o}_{3}}\right\}$ 是在有限词典 $D = \left\{  {{w}_{1},\ldots ,{w}_{n}}\right\}$ 的 $n$ 个词上定义的一组文档。设 $\mathcal{L}$ 为包含二元函数符号 concat $\left( {x,y}\right)$ 的语言，表示将文档 $x$ 与 $y$ 连接得到的文档。又设 $\mathcal{L}$ 还包含二元谓词 ${Sim}$，若文档 $x$ 被视为与文档 $y$ 相似则应为真。 grounding 的一个例子是将每个文档关联到其词袋向量 [7]。因此，对 concat 函数的自然 grounding 将是向量之和，对 Sim 谓词则是向量之间的余弦相似度。更正式地：


- $\mathcal{G}\left( {o}_{i}\right)  = \left\langle  {{n}_{{w}_{1}}^{{o}_{i}},\ldots ,{n}_{{w}_{n}}^{{o}_{i}}}\right\rangle$ ,where ${n}_{w}^{d}$ is the number of occurrences of word $w$ in document $d$ ;
- $\mathcal{G}\left( {o}_{i}\right)  = \left\langle  {{n}_{{w}_{1}}^{{o}_{i}},\ldots ,{n}_{{w}_{n}}^{{o}_{i}}}\right\rangle$ ，其中 ${n}_{w}^{d}$ 是词 $w$ 在文档 $d$ 中的出现次数；


$$
\text{ - if }\mathbf{v},\mathbf{u} \in  {\mathbb{R}}^{n},\mathcal{G}\left( \text{ concat }\right) \left( {\mathbf{u},\mathbf{v}}\right)  = \mathbf{u} + \mathbf{v}\text{ ; }
$$



- if $\mathbf{v},\mathbf{u} \in  {\mathbb{R}}^{n},\mathcal{G}\left( \operatorname{Sim}\right) \left( {\mathbf{u},\mathbf{v}}\right)  = \frac{\mathbf{u} \cdot  \mathbf{v}}{\parallel \mathbf{u}\parallel \parallel \mathbf{v}\parallel }$ .
- 如果 $\mathbf{v},\mathbf{u} \in  {\mathbb{R}}^{n},\mathcal{G}\left( \operatorname{Sim}\right) \left( {\mathbf{u},\mathbf{v}}\right)  = \frac{\mathbf{u} \cdot  \mathbf{v}}{\parallel \mathbf{u}\parallel \parallel \mathbf{v}\parallel }$ 。


For instance,if the three documents are ${o}_{1} =$ "John studies logic and plays football", ${o}_{2}$ = "Mary plays football and logic games", ${o}_{3} =$ "John and Mary play football and study logic together",and $W = \{$ John,Mary,and,football,game,logic,play,study,together $\}$ then the following are examples of the grounding of terms, atomic formulas and clauses.
例如，如果三份文档是 ${o}_{1} =$ "John studies logic and plays football"、${o}_{2}$ "Mary plays football and logic games"、${o}_{3} =$ "John and Mary play football and study logic together"，且 $W = \{$ John, Mary, and, football, game, logic, play, study, together $\}$，那么下列是术语、原子公式和子句 grounding 的示例。


$$
\mathcal{G}\left( {o}_{1}\right)  = \langle 1,0,1,1,0,1,1,1,0\rangle
$$



$$
\mathcal{G}\left( {o}_{2}\right)  = \langle 0,1,1,1,1,1,1,0,0\rangle
$$



$$
\mathcal{G}\left( {o}_{3}\right)  = \langle 1,1,2,1,0,1,1,1,1\rangle
$$



$$
\mathcal{G}\left( {\operatorname{concat}\left( {{o}_{1},{o}_{2}}\right) }\right)  = \mathcal{G}\left( {o}_{1}\right)  + \mathcal{G}\left( {o}_{2}\right)  = \langle 1,1,2,2,1,2,2,1,0\rangle
$$



$$
\mathcal{G}\left( {\operatorname{Sim}\left( {\operatorname{concat}\left( {{o}_{1},{o}_{2}}\right) ,{o}_{3}}\right)  = \frac{\mathcal{G}\left( {\operatorname{concat}\left( {{o}_{1},{o}_{2}}\right) }\right)  \cdot  \mathcal{G}\left( {o}_{3}\right) }{\begin{Vmatrix}{\mathcal{G}\left( {\operatorname{concat}\left( {{o}_{1},{o}_{2}}\right) }\right) }\end{Vmatrix} \cdot  \begin{Vmatrix}{\mathcal{G}\left( {o}_{3}\right) }\end{Vmatrix}} \approx  \frac{13}{14.83} \approx  {0.88}}\right.
$$



$$
\mathcal{G}\left( {\operatorname{Sim}\left( {{o}_{1},{o}_{3}}\right)  \vee  \operatorname{Sim}\left( {{o}_{2},{o}_{3}}\right) }\right)  = {\mu }_{\max }\left( {\mathcal{G}\left( {\operatorname{Sim}\left( {{o}_{1},{o}_{3}}\right) ,\mathcal{G}\left( {\operatorname{Sim}\left( {{o}_{2},{o}_{3}}\right) }\right) }\right. }\right.
$$



$$
\approx  \max \left( {{0.86},{0.73}}\right)  = {0.86}
$$



---



${}^{3}$ Examples of t-norms which can be chosen here are Lukasiewicz,product,and Gödel. Lukasiewicz s-norm is defined as ${\mu }_{Luk}\left( {x,y}\right)  = \min \left( {x + y,1}\right)$ ; Product s-norm is defined as ${\mu }_{Pr}\left( {x,y}\right)  = x + y - x \cdot  y$ ; Gödel s-norm is defined as ${\mu }_{\max }\left( {x,y}\right)  = \max \left( {x,y}\right)$ .
${}^{3}$ 可在此处选择的 t-范数示例包括 Lukasiewicz、乘积和 Gödel。Lukasiewicz 的 s-范数定义为 ${\mu }_{Luk}\left( {x,y}\right)  = \min \left( {x + y,1}\right)$ ；乘积 s-范数定义为 ${\mu }_{Pr}\left( {x,y}\right)  = x + y - x \cdot  y$ ；Gödel s-范数定义为 ${\mu }_{\max }\left( {x,y}\right)  = \max \left( {x,y}\right)$ 。


---



## 3 Learning as approximate satisfiability
## 3 学习作为近似可满足性


We start by defining ground theory and their satisfiability.
我们首先定义 grounding 理论及其可满足性。


Definition 2 (Satisfiability). Let $\phi$ be a closed clause in $\mathcal{L},\mathcal{G}$ a grounding,and $v \leq \; w \in  \left\lbrack  {0,1}\right\rbrack$ . We say that $\mathcal{G}$ satisfies $\phi$ in the confidence interval $\left\lbrack  {v,w}\right\rbrack$ ,written $\mathcal{G} \vDash  {}_{v}^{w}\phi$ , if $v \leq  \mathcal{G}\left( \phi \right)  \leq  w$ .
定义 2（可满足性）。设 $\phi$ 为 $\mathcal{L},\mathcal{G}$ 中的一个闭合子句在 $\mathcal{L},\mathcal{G}$ 的 grounding 中，且 $v \leq \; w \in  \left\lbrack  {0,1}\right\rbrack$。如果 $\mathcal{G}$ 在置信区间 $\left\lbrack  {v,w}\right\rbrack$ 内满足 $\phi$，记作 $\mathcal{G} \vDash  {}_{v}^{w}\phi$，则当且仅当 $v \leq  \mathcal{G}\left( \phi \right)  \leq  w$ 。


A partial grounding,denoted by $\widehat{\mathcal{G}}$ ,is a grounding that is defined on a subset of the signature of $\mathcal{L}$ . A grounded theory is a set of clauses in the language of $\mathcal{L}$ and partial grounding $\widehat{\mathcal{G}}$ .
部分 grounding，用记号 $\widehat{\mathcal{G}}$ 表示，是在 $\mathcal{L}$ 的签名子集上定义的 grounding。grounded theory 是语言 $\mathcal{L}$ 中的一组子句与部分 grounding $\widehat{\mathcal{G}}$ 的集合。


Definition 3 (Grounded Theory). A grounded theory (GT) is a pair $\langle \mathcal{K},\widehat{\mathcal{G}}\rangle$ where $\mathcal{K}$ is a set of pairs $\langle \left\lbrack  {v,w}\right\rbrack  ,\phi \left( \mathbf{x}\right) \rangle$ ,where $\phi \left( \mathbf{x}\right)$ is a clause of $\mathcal{L}$ containing the set $\mathbf{x}$ of free variables,and $\left\lbrack  {v,w}\right\rbrack   \subseteq  \left\lbrack  {0,1}\right\rbrack$ is an interval contained in $\left\lbrack  {0,1}\right\rbrack$ ,and $\widehat{\mathcal{G}}$ is a partial grounding.
定义 3（扎根理论）。一个扎根理论（GT）是一对 $\langle \mathcal{K},\widehat{\mathcal{G}}\rangle$，其中 $\mathcal{K}$ 是一组对 $\langle \left\lbrack  {v,w}\right\rbrack  ,\phi \left( \mathbf{x}\right) \rangle$ 的集合，$\phi \left( \mathbf{x}\right)$ 是包含自由变量集合 $\mathbf{x}$ 的 $\mathcal{L}$ 的一个子句，$\left\lbrack  {v,w}\right\rbrack   \subseteq  \left\lbrack  {0,1}\right\rbrack$ 是包含在 $\left\lbrack  {0,1}\right\rbrack$ 内的区间，且 $\widehat{\mathcal{G}}$ 是一个部分扎根。


Definition 4 (Satisfiability of a Grounded Theory). A GT $\langle \mathcal{K},\widehat{\mathcal{G}}\rangle$ is satisfiabile if there exists a grounding $\mathcal{G}$ ,which extends $\widehat{\mathcal{G}}$ such that for all $\langle \left\lbrack  {v,w}\right\rbrack  ,\phi \left( \mathbf{x}\right) \rangle  \in  \mathcal{K}$ and any tuple $\mathbf{t}$ of closed terms, $\mathcal{G}{ \vDash  }_{v}^{w}\phi \left( \mathbf{t}\right)$ .
定义 4（扎根理论的可满足性）。若存在一个延伸自 $\widehat{\mathcal{G}}$ 的扎根 $\mathcal{G}$，并且对所有 $\langle \left\lbrack  {v,w}\right\rbrack  ,\phi \left( \mathbf{x}\right) \rangle  \in  \mathcal{K}$ 及任意闭合项元组 $\mathbf{t}$，$\mathcal{G}{ \vDash  }_{v}^{w}\phi \left( \mathbf{t}\right)$，则该 GT $\langle \mathcal{K},\widehat{\mathcal{G}}\rangle$ 可满足。


From the previous definiiton it follows that checking if a GT $\langle \mathcal{K},\widehat{\mathcal{G}}\rangle$ is satisfiable amounts to seaching for an extension of the partial grounding $\widehat{\mathcal{G}}$ in the space of all possible groundings,such that all the instantiations of the clauses in $\mathcal{K}$ are satisfied w.r.t. the specified interval. Clearly this is unfeasible from a practical point of view. As is usual, we must restrict both the space of grounding and clause instantiations. Let us consider each in turn: To check satisfiability on a subset of all the functions on real numbers, recall that a grounding should capture a latent correlation between the quantitative attributes of an object and its relational properties ${}^{4}$ . In particular,we are interested in searching within a specific class of functions, in this paper based on tensor networks, although other family of functions can be considered. To limit the number of clause instantiations,which in general might be infinite since $\mathcal{L}$ admits function symbols, the usual approach is to consider the instantiations of each clause up to a certain depth [3].
从前面的定义可以得出，检查一个 GT $\langle \mathcal{K},\widehat{\mathcal{G}}\rangle$ 是否可满足，等同于在所有可能的扎根空间中搜索对部分扎根 $\widehat{\mathcal{G}}$ 的扩展，使得 $\mathcal{K}$ 中的子句的所有实例相对于给定区间被满足。显然，从实际角度看这是不可行的。常规做法是同时限制扎根空间和子句实例化的数量。让我们依次考虑：要在实数上的所有函数子集上检查可满足性，记得扎根应该捕捉对象的定量属性与其关系属性 ${}^{4}$ 之间的潜在相关性。特别地，我们感兴趣的是在一个特定类的函数中进行搜索，本文基于张量网络，尽管也可考虑其他函数族。为限制子句实例化数量，通常认为对于每个子句的实例化到某个深度 [3] 即可，因为 $\mathcal{L}$ 含有函数符号。


When a grounded theory $\langle \mathcal{K},\widehat{\mathcal{G}}\rangle$ is inconsitent,that is,there is no grounding $\mathcal{G}$ that satisfies it, we are interested in finding a grounding which satisfies as much as possible of $\langle \mathcal{K},\widehat{\mathcal{G}}\rangle$ . For any $\langle \left\lbrack  {v,w}\right\rbrack  ,\phi \rangle  \in  \mathcal{K}$ we want to find a grounding $\mathcal{G}$ that minimizes the satisfiability error. An error occurs when a grounding $\mathcal{G}$ assigns a value $\mathcal{G}\left( \phi \right)$ to a clause $\phi$ which is outside the interval $\left\lbrack  {v,w}\right\rbrack$ prescribed by $\mathcal{K}$ . The measure of this error can be defined as the minimal distance between the points in the interval $\left\lbrack  {v,w}\right\rbrack$ and $\mathcal{G}\left( \phi \right)$ :
当一个扎根理论 $\langle \mathcal{K},\widehat{\mathcal{G}}\rangle$ 不一致时，即不存在满足它的扎根 $\mathcal{G}$，我们希望找到尽可能多地满足 $\langle \mathcal{K},\widehat{\mathcal{G}}\rangle$ 的扎根。对于任意 $\langle \left\lbrack  {v,w}\right\rbrack  ,\phi \rangle  \in  \mathcal{K}$，我们希望找到一个使可满足性误差最小的扎根 $\mathcal{G}$。当一个扎根 $\mathcal{G}$ 给一个子句 $\phi$ 指定的值 $\mathcal{G}\left( \phi \right)$ 落在由 $\mathcal{K}$ 规定的区间 $\left\lbrack  {v,w}\right\rbrack$ 之外时，即发生误差。这个误差的度量可以定义为区间 $\left\lbrack  {v,w}\right\rbrack$ 与 $\mathcal{G}\left( \phi \right)$ 之间的最近距离：


$$
\operatorname{Loss}\left( {\mathcal{G},\langle \left\lbrack  {v,w}\right\rbrack  ,\phi \rangle }\right)  = \left| {x - \mathcal{G}\left( \phi \right) }\right| ,v \leq  x \leq  w \tag{1}
$$



Notice that if $\mathcal{G}\left( \phi \right)  \in  \left\lbrack  {v,w}\right\rbrack  ,\operatorname{Loss}\left( {\mathcal{G},\phi }\right)  = 0$ .
注意如果 $\mathcal{G}\left( \phi \right)  \in  \left\lbrack  {v,w}\right\rbrack  ,\operatorname{Loss}\left( {\mathcal{G},\phi }\right)  = 0$ 。


---



${}^{4}$ For example,whether a document is classified as from the field of Artificial Intelligence (AI) depends on its bag-of-words grounding. If the language $\mathcal{L}$ contains the unary predicate ${AI}\left( x\right)$ standing for " $x$ is a paper about AI" then the grounding of ${AI}\left( x\right)$ ,which is a function from bag-of-words vectors to $\left\lbrack  {0,1}\right\rbrack$ ,should assign values close to 1 to the vectors which are close semantically to ${AI}$ . Furthermore,if two vectors are similar (e.g. according to the cosine similarity measure) then their grounding should be similar.
${}^{4}$ 例如，文档是否来自人工智能领域（AI）取决于其词袋基础。若语言 $\mathcal{L}$ 含有表示“ $x$ 是一篇关于 AI 的论文”的一元谓词 ${AI}\left( x\right)$，则 ${AI}\left( x\right)$ 的 grounding（从词袋向量到 $\left\lbrack  {0,1}\right\rbrack$ 的函数）应对在语义上与 ${AI}$ 接近的向量赋予接近1的值。此外，若两个向量相似（如根据余弦相似度衡量），则它们的 grounding 也应相似。


---



The above gives rise to the following definition of approximate satisfiability w.r.t. a family $\mathbb{G}$ of grounding functions on the language $\mathcal{L}$ .
上述给出关于一个 grounding 函数族 $\mathbb{G}$ 的语言 $\mathcal{L}$ 的近似可满足性的定义。


Definition 5 (Approximate satisfiability). Let $\langle \mathcal{K},\widehat{\mathcal{G}}\rangle$ be a grounded theory and ${\mathcal{K}}_{0}$ a finite subset of the instantiations of the clauses in $\mathcal{K}$ ,i.e.
定义5（近似可满足性）。令 $\langle \mathcal{K},\widehat{\mathcal{G}}\rangle$ 为一个 grounded theory，且 ${\mathcal{K}}_{0}$ 为在 $\mathcal{K}$ 条款实例中的有限子集，即


$$
{K}_{0} \subseteq  \{ \langle \left\lbrack  {v,w}\right\rbrack  ,\phi \left( \mathbf{t}\right) \rangle \}  \mid  \langle \left\lbrack  {v,w}\right\rbrack  ,\phi \left( \mathbf{x}\right) \rangle  \in  \mathcal{K}\text{ and }\mathbf{t}\text{ is any }n\text{ -tuple of closed terms. }\}
$$



Let $\mathbb{G}$ be a family of grounding functions. We define the best satisfiability problem as the problem of finding an extensions ${\mathcal{G}}^{ * }$ of $\widehat{\mathcal{G}}$ in $\mathbb{G}$ that minimizes the satisfiability error on the set ${\mathcal{K}}_{0}$ ,that is:
设 $\mathbb{G}$ 为 grounding 函数族。我们将最佳可满足性问题定义为在 $\mathbb{G}$ 中找到对 $\widehat{\mathcal{G}}$ 的扩展 ${\mathcal{G}}^{ * }$，以在集合 ${\mathcal{K}}_{0}$ 上最小化可满足性误差，即：


$$
{\mathcal{G}}^{ * } = \mathop{\operatorname{argmin}}\limits_{{\widehat{\mathcal{G}} \subseteq  \mathcal{G} \in  \mathbb{G}}}\mathop{\sum }\limits_{{\langle \left\lbrack  {v,w}\right\rbrack  ,\phi \left( \mathbf{t}\right) \rangle  \in  {\mathcal{K}}_{0}}}\operatorname{Loss}\left( {\mathcal{G},\langle \left\lbrack  {v,w}\right\rbrack  ,\phi \left( \mathbf{t}\right) \rangle }\right)
$$



## 4 Implementing Real Logic in Tensor Networks
## 4 在张量网络中实现真实逻辑


Specific instances of Real Logic can be obtained by selectiong the space $\mathbb{G}$ of ground-ings and the specific s-norm for the interpretation of disjunction. In this section, we describe a realization of real logic where $\mathbb{G}$ is the space of real tensor transformations of order $k$ (where $k$ is a parameter). In this space,function symbols are interpreted as linear transformations. More precisely,if $f$ is a function symbol of arity $m$ and ${\mathbf{v}}_{1},\ldots ,{\mathbf{v}}_{m} \in  {\mathbb{R}}^{n}$ are real vectors corresponding to the grounding of $m$ terms then $\mathcal{G}\left( f\right) \left( {{\mathbf{v}}_{1},\ldots ,{\mathbf{v}}_{m}}\right)$ can be written as:
通过选择 grounding 的空间 $\mathbb{G}$ 和对析取的解释使用的具体 s-范数，可以获得真实逻辑的具体实例。在本节中，我们描述一种真实逻辑的实现，其中 $\mathbb{G}$ 是阶数为 $k$ 的实值张量变换空间（其中 $k$ 是一个参数）。在该空间中，函数符号被解释为线性变换。更确切地说，若 $f$ 是一个 $m$ 阶的函数符号，且 ${\mathbf{v}}_{1},\ldots ,{\mathbf{v}}_{m} \in  {\mathbb{R}}^{n}$ 是对应 $m$ 项 grounding 的实向量，则 $\mathcal{G}\left( f\right) \left( {{\mathbf{v}}_{1},\ldots ,{\mathbf{v}}_{m}}\right)$ 可以写成：


$$
\mathcal{G}\left( f\right) \left( {{\mathbf{v}}_{1},\ldots ,{\mathbf{v}}_{m}}\right)  = {M}_{f}\mathbf{v} + {N}_{f}
$$



for some $n \times  {mn}$ matrix ${M}_{f}$ and $n$ -vector ${N}_{f}$ ,where $\mathbf{v} = \left\langle  {{\mathbf{v}}_{1},\ldots ,{\mathbf{v}}_{n}}\right\rangle$ .
对于某些 $n \times  {mn}$ 矩阵 ${M}_{f}$ 与 $n$ 向量 ${N}_{f}$，其中 $\mathbf{v} = \left\langle  {{\mathbf{v}}_{1},\ldots ,{\mathbf{v}}_{n}}\right\rangle$ 。


The grounding of $m$ -ary predicate $P,\mathcal{G}\left( P\right)$ ,is defined as a generalization of the neural tensor network [26] (which has been shown effective at knowledge compilation in the presence of simple logical constraints),as a function from ${\mathbb{R}}^{mn}$ to $\left\lbrack  {0,1}\right\rbrack$ ,as follows:
对于 $m$-元谓词 $P,\mathcal{G}\left( P\right)$ 的 grounding，被定义为对神经张量网络的推广 [26]（在存在简单逻辑约束时已被证明在知识压缩方面有效），作为从 ${\mathbb{R}}^{mn}$ 到 $\left\lbrack  {0,1}\right\rbrack$ 的函数，具体如下：


$$
\mathcal{G}\left( P\right)  = \sigma \left( {{u}_{P}^{T}\tanh \left( {{\mathbf{v}}^{T}{W}_{P}^{\left\lbrack  1 : k\right\rbrack  }\mathbf{v} + {V}_{P}\mathbf{v} + {B}_{P}}\right) }\right) \tag{2}
$$



where ${W}_{P}^{\left\lbrack  1 : k\right\rbrack  }$ is a 3-D tensor in ${\mathbb{R}}^{{mn} \times  {mn} \times  k},{V}_{P}$ is a matrix in ${\mathbb{R}}^{k \times  {mn}}$ ,and ${B}_{P}$ is a vector in ${\mathbb{R}}^{k}$ ,and $\sigma$ is the sigmoid function. With this encoding,the grounding (i.e. truth-value) of a clause can be determined by a neural network which first computes the grounding of the literals contained in the clause, and then combines them using the specific s-norm. An example of tensor network for $\neg P\left( {x,y}\right)  \rightarrow  A\left( y\right)$ is shown in Figure 1. This architecture is a generalization of the structure proposed in [26], that has been shown rather effective for the task of knowledge compilation, also in presence of simple logical constraints. In the above tensor network formulation, ${W}_{ * },{V}_{ * },{B}_{ * }$ and ${u}_{ * }$ with $*  \in  \{ P,A\}$ are parameters to be learned by minimizing the loss function or, equivalently,to maximize the satisfiability of the clause $P\left( {x,y}\right)  \rightarrow  A\left( y\right)$ .
其中 ${W}_{P}^{\left\lbrack  1 : k\right\rbrack  }$ 是在 ${\mathbb{R}}^{{mn} \times  {mn} \times  k},{V}_{P}$ 的一个三维张量，${\mathbb{R}}^{k \times  {mn}}$ 是一个矩阵，${B}_{P}$ 是一个向量，${\mathbb{R}}^{k}$，并且 $\sigma$ 是 sigmoid 函数。利用这种编码，可以由神经网络首先计算子句中文字的 grounding（即真值），然后使用特定的 s-范数将它们组合来确定子句的 grounding。一个关于 $\neg P\left( {x,y}\right)  \rightarrow  A\left( y\right)$ 的张量网络示例如图 1 所示。这一体系结构是对 [26] 提出结构的推广，在知识压缩任务中，甚至在存在简单逻辑约束的情况下也被证明相当有效。在上述张量网络的形式化中，${W}_{ * },{V}_{ * },{B}_{ * }$ 与 ${u}_{ * }$ 连同 $*  \in  \{ P,A\}$ 是需要通过最小化损失函数来学习的参数，或者等价地，为最大化子句 $P\left( {x,y}\right)  \rightarrow  A\left( y\right)$ 的可满足性。 


<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/02/2026_02_18__20_16_14_ea68d5.jpg"/>



Fig. 1. Tensor net for $P\left( {x,y}\right)  \rightarrow  A\left( y\right)$ ,with $\mathcal{G}\left( x\right)  = \mathbf{v}$ and $\mathcal{G}\left( y\right)  = \mathbf{u}$ and $k = 2$ .
图 1。$P\left( {x,y}\right)  \rightarrow  A\left( y\right)$ 的张量网，带有 $\mathcal{G}\left( x\right)  = \mathbf{v}$、$\mathcal{G}\left( y\right)  = \mathbf{u}$ 与 $k = 2$。


## 5 An Example of Knowledge Completion
## 5 知识完成的一个示例


Logic Tensor Networks have been implemented as a Python library called 1tn using Google's TENSORFLOW ${}^{\mathrm{{TM}}}$ . To test our idea,in this section we use the well-known friends and smokers ${}^{5}$ example [24] to illustrate the task of knowledge completion in 1tn. There are 14 people divided into two groups $\{ a,b,\ldots ,h\}$ and $\{ i,j,\ldots ,n\}$ . Within each group of people we have complete knowledge of their smoking habits. In the first group, we have complete knowledge of who has and does not have cancer. In the second group, this is not known for any of the persons. Knowledge about the friendship relation is complete within each group only if symmetry of friendship is assumed. Otherwise, it is imcomplete in that it may be known that, e.g., a is a friend of $b$ ,but not known whether $b$ is a friend of $a$ . Finally,there is also general knowledge about smoking, friendship and cancer, namely, that smoking causes cancer, friendship is normally a symmetric and anti-reflexive relation, everyone has a friend, and that smoking propagates (either actively or passively) among friends. All this knowledge can be represented by the knowledge-bases shown in Figure 2.
逻辑张量网络已实现为名为 1tn 的 Python 库，使用 Google 的 TENSORFLOW ${}^{\mathrm{{TM}}}$。为测试我们的想法，本节我们用著名的朋友与吸烟者 ${}^{5}$ 示例 [24] 来说明在 1tn 中进行知识完成的任务。共有 14 个人，分为两组 $\{ a,b,\ldots ,h\}$ 与 $\{ i,j,\ldots ,n\}$。在每组人中，我们对他们的吸烟习惯有完整的知识。在第一组，我们对谁有癌症和谁没有癌症有完整知识。在第二组，对于任何人，这些信息并不为人知。关于友谊关系的知识在每组内只有在假设友谊对称时才是完整的。否则，它在某些情况下是不完整的，如可能已知 a 是 $b$ 的朋友，但尚不清楚 $b$ 是否是 $a$ 的朋友。最后，还有关于吸烟、友谊和癌症的一般性知识，即吸烟会导致癌症、友谊通常是对称且自反性为假、每人都有一个朋友，以及吸烟在朋友之间传播（主动或被动）。所有这些知识都可以通过图 2 所示的知识库来表示。


The facts contained in the knowledge-bases should have different degrees of truth, and this is not known. Otherwise, the combined knowledge-base would be inconsistent (it would deduce e.g. $S\left( b\right)$ and $\neg S\left( b\right)$ ). Our main task is to complete the knowledge-base (KB), that is: (i) find the degree of truth of the facts contained in KB, (ii) find a truth-value for all the missing facts,e.g. $C\left( i\right)$ ,(iii) find the grounding of each constant symbol $a,\ldots ,n{.}^{6}$ To answer (i)-(iii),we use 1tn to find a grounding that best approximates the complete KB. We start by assuming that all the facts contained in the knowledge-base are true (i.e. have degree of truth 1). To show the role of background knolwedge in the learning-inference process, we run two experiments. In the first (exp1),we seek to complete a KB consisting of only factual knowledge: ${\mathcal{K}}_{\text{ exp1 }} = \; {\mathcal{K}}_{a\ldots h}^{SFC} \cup  {\mathcal{K}}_{i\ldots n}^{SF}$ . In the second (exp1),we also include background knowledge,that is: ${\mathcal{K}}_{exp2} = {\mathcal{K}}_{exp1} \cup  {\mathcal{K}}^{SFC}.$
知识库中的事实应具有不同程度的真值，这一点目前尚不清楚。否则，合成的知识库将不一致（例如会推得 $S\left( b\right)$ 与 $\neg S\left( b\right)$）。我们的主要任务是完善知识库（KB），即：（i）找出 KB 中事实的真值程度，（ii）为所有缺失的事实给出真值，例如 $C\left( i\right)$，（iii）为每个常量符号 $a,\ldots ,n{.}^{6}$ 找到其基底。为回答（i）-（iii），我们使用 1tn 找到最能逼近完整 KB 的基底。我们先假设知识库中的所有事实都为真（即真值度为 1）。为展示背景知识在学习-推理过程中的作用，我们进行两组实验。在第一组（exp1）中，旨在完成仅包含事实性知识的 KB：${\mathcal{K}}_{\text{ exp1 }} = \; {\mathcal{K}}_{a\ldots h}^{SFC} \cup  {\mathcal{K}}_{i\ldots n}^{SF}$。在第二组（exp1）中，我们也包含背景知识，即：${\mathcal{K}}_{exp2} = {\mathcal{K}}_{exp1} \cup  {\mathcal{K}}^{SFC}.$。


---



${}^{5}$ Normally,a probabilistic approach is taken to solve this problem,and one that requires instantiating all clauses to remove variables, essentially turning the problem into a propositional one; 1tn takes a different approach.
${}^{5}$ 通常采取概率方法来解决这个问题，并且需要实例化所有子句以消除变量，基本上将问题转化为命题式；1tn 采取了不同的方法。


${}^{6}$ Notice how no grounding is provided about the signature of the knowledge-base.
${}^{6}$ 注意没有给出知识库签名的基底。


---



<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/02/2026_02_18__20_16_14_a4bcc4.jpg"/>



Fig.2. Knowledge-bases for the friends-and-smokers example.
图2。朋友与吸烟者示例的知识库。


We confgure the network as follows: each constant (i.e. person) can have up to 30 real-valued features. We set the number of layers $k$ in the tensor network to 10,and the regularization parameter ${}^{7}\lambda  = {1}^{-{10}}$ . For the purpose of illustration,we use the Lukasiewicz t-norm with s-norm $\mu \left( {a,b}\right)  = \min \left( {1,a + b}\right)$ ,and use the harmonic mean as aggregation operator. An estimation of the optimal grounding is obtained after 5,000 runs of the RMSProp learning algorithm [27] available in TENSORFLOW ${}^{\mathrm{{TM}}}$ .
我们将网络配置如下：每个常量（即人）可以具有最多 30 个实值特征。我们将张量网络的层数 $k$ 设为 10，正则化参数 ${}^{7}\lambda  = {1}^{-{10}}$ 。为了便于说明，我们使用 Lukasiewicz t-范数及其 s-范数 $\mu \left( {a,b}\right)  = \min \left( {1,a + b}\right)$，并使用调和平均作为聚合算子。经过 5,000 轮 RMSProp 学习算法的运行后得到对最优基底的估计值，该算法在 TensorFlow ${}^{\mathrm{{TM}}}$ 中可用。


The results of the two experiments are reported in Table 1. For readability, we use boldface for truth-values greater than 0.5 . The truth-values of the facts listed in a knowledge-base are highlighted with the same background color of the knowledge-base in Figure 2. The values with white background are the result of the knowledge completion produced by the LTN learning-inference procedure. To evaluate the quality of the results, one has to check whether (i) the truth-values of the facts listed in a KB are indeed close to 1.0, and (ii) the truth-values associated with knowledge completion correspond to expectation. An initial analysis shows that the LTN associated with ${\mathcal{K}}_{\text{ exp1 }}$ produces the same facts as ${\mathcal{K}}_{\text{ exp1 }}$ itself. In other words,the LTN fits the data. However,the LTN also learns to infer additional positive and negative facts about $F$ and $C$ not derivable from ${\mathcal{K}}_{\exp 1}$ by pure logical reasoning; for example: $F\left( {c,b}\right) ,F\left( {g,b}\right)$ and $\neg F\left( {b,a}\right)$ . These facts are derived by exploiting similarities between the groundings of
两组实验的结果在表1中给出。为便于阅读，对真值大于 0.5 的结果使用粗体。知识库中事实的真值在图2的知识库背景色中高亮显示。白色背景的数值是通过 LT N 学习-推理过程得到的知识完成结果。要评估结果的质量，需要检查（i）知识库中列出事实的真值是否确实接近 1.0，以及（ii）与知识完成相关的真值是否符合预期。初步分析显示，与 ${\mathcal{K}}_{\text{ exp1 }}$ 相关的 LT N 产生的事实与 ${\mathcal{K}}_{\text{ exp1 }}$ 本身相同。换言之，LTN 拟合了数据。然而，LTN 还学会推断出关于 $F$ 与 $C$ 的额外正向和负向事实，这些事实不能仅通过纯逻辑推理从 ${\mathcal{K}}_{\exp 1}$ 导出；例如：$F\left( {c,b}\right) ,F\left( {g,b}\right)$ 与 $\neg F\left( {b,a}\right)$。这些事实通过利用基底的相似性来推断。


---



${}^{7}$ A smoothing factor $\lambda \parallel \mathbf{\Omega }{\parallel }_{2}^{2}$ is added to the loss function to create a preference for learned parameters with a lower absolute value.
${}^{7}$ 在损失函数中加入平滑因子 $\lambda \parallel \mathbf{\Omega }{\parallel }_{2}^{2}$，以偏好学得参数的绝对值较小。


---



<table><tr><td></td><td rowspan="2"></td><td rowspan="2">$S$</td><td rowspan="2">$C$</td><td colspan="8">$F$</td><td rowspan="2"></td><td rowspan="2">$S$</td><td rowspan="2">$C$</td><td colspan="6">$F$</td><td></td></tr><tr><td></td><td>$a$</td><td>$b$</td><td>$C$</td><td>$d$</td><td>$e$</td><td>$f$</td><td>$g$</td><td>$h$</td><td>$i$</td><td>$j$</td><td>$k$</td><td>$l$</td><td>$m$</td><td>$n$</td><td></td></tr><tr><td></td><td>$a$</td><td>1.00</td><td>1.00</td><td>0.00</td><td>1.00</td><td>0.00</td><td>0.00</td><td>1.00</td><td>1.00</td><td>1.00</td><td>0.00</td><td>$i$</td><td>1.00</td><td>0.00</td><td>0.00</td><td>1.00</td><td>0.00</td><td>0.00</td><td>1.00</td><td>0.00</td><td></td></tr><tr><td></td><td>$b$</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>1.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>$j$</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td></td></tr><tr><td></td><td>$c$</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.82</td><td>0.00</td><td>1.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>$k$</td><td>0.00</td><td>0.00</td><td>0.10</td><td>1.00</td><td>0.00</td><td>1.00</td><td>0.00</td><td>0.00</td><td></td></tr><tr><td></td><td>$d$</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.06</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>$l$</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.02</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td></td></tr><tr><td></td><td>$e$</td><td>1.00</td><td>1.00</td><td>0.00</td><td>0.33</td><td>0.21</td><td>0.00</td><td>0.00</td><td>1.00</td><td>0.00</td><td>0.00</td><td>$m$</td><td>0.00</td><td>0.03</td><td>1.00</td><td>1.00</td><td>0.12</td><td>1.00</td><td>0.00</td><td>1.00</td><td></td></tr><tr><td></td><td>$f$</td><td>1.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.05</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>$n$</td><td>1.00</td><td>0.01</td><td>0.00</td><td>0.98</td><td>0.00</td><td>0.01</td><td>0.02</td><td>0.00</td><td></td></tr><tr><td></td><td>$g$</td><td>1.00</td><td>0.00</td><td>0.03</td><td>1.00</td><td>1.00</td><td>1.00</td><td>0.11</td><td>1.00</td><td>0.00</td><td>1.00</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>$h$</td><td>0.00</td><td>0.00</td><td>0.00 <br> 二</td><td>0.23</td><td>0.01 <br> 0</td><td>0.14</td><td>0.00</td><td>0.02</td><td>0.00</td><td>0.00 <br> -</td><td></td><td></td><td>IGERO</td><td></td><td>IGSR</td><td></td><td></td><td></td><td></td><td></td></tr></table>
<table><tbody><tr><td></td><td rowspan="2"></td><td rowspan="2">$S$</td><td rowspan="2">$C$</td><td colspan="8">$F$</td><td rowspan="2"></td><td rowspan="2">$S$</td><td rowspan="2">$C$</td><td colspan="6">$F$</td><td></td></tr><tr><td></td><td>$a$</td><td>$b$</td><td>$C$</td><td>$d$</td><td>$e$</td><td>$f$</td><td>$g$</td><td>$h$</td><td>$i$</td><td>$j$</td><td>$k$</td><td>$l$</td><td>$m$</td><td>$n$</td><td></td></tr><tr><td></td><td>$a$</td><td>1.00</td><td>1.00</td><td>0.00</td><td>1.00</td><td>0.00</td><td>0.00</td><td>1.00</td><td>1.00</td><td>1.00</td><td>0.00</td><td>$i$</td><td>1.00</td><td>0.00</td><td>0.00</td><td>1.00</td><td>0.00</td><td>0.00</td><td>1.00</td><td>0.00</td><td></td></tr><tr><td></td><td>$b$</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>1.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>$j$</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td></td></tr><tr><td></td><td>$c$</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.82</td><td>0.00</td><td>1.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>$k$</td><td>0.00</td><td>0.00</td><td>0.10</td><td>1.00</td><td>0.00</td><td>1.00</td><td>0.00</td><td>0.00</td><td></td></tr><tr><td></td><td>$d$</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.06</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>$l$</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.02</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td></td></tr><tr><td></td><td>$e$</td><td>1.00</td><td>1.00</td><td>0.00</td><td>0.33</td><td>0.21</td><td>0.00</td><td>0.00</td><td>1.00</td><td>0.00</td><td>0.00</td><td>$m$</td><td>0.00</td><td>0.03</td><td>1.00</td><td>1.00</td><td>0.12</td><td>1.00</td><td>0.00</td><td>1.00</td><td></td></tr><tr><td></td><td>$f$</td><td>1.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.05</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>$n$</td><td>1.00</td><td>0.01</td><td>0.00</td><td>0.98</td><td>0.00</td><td>0.01</td><td>0.02</td><td>0.00</td><td></td></tr><tr><td></td><td>$g$</td><td>1.00</td><td>0.00</td><td>0.03</td><td>1.00</td><td>1.00</td><td>1.00</td><td>0.11</td><td>1.00</td><td>0.00</td><td>1.00</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>$h$</td><td>0.00</td><td>0.00</td><td>0.00 <br/> 二</td><td>0.23</td><td>0.01 <br/> 0</td><td>0.14</td><td>0.00</td><td>0.02</td><td>0.00</td><td>0.00 <br/> -</td><td></td><td></td><td>IGERO</td><td></td><td>IGSR</td><td></td><td></td><td></td><td></td><td></td></tr></tbody></table>


Learning and reasoning on ${\mathcal{K}}_{\text{ exp1 }} = {\mathcal{K}}_{a\ldots h}^{SFC} \cup  {\mathcal{K}}_{i\ldots n}^{SF}$
学习与推理在 ${\mathcal{K}}_{\text{ exp1 }} = {\mathcal{K}}_{a\ldots h}^{SFC} \cup  {\mathcal{K}}_{i\ldots n}^{SF}$


<table><tr><td rowspan="2"></td><td rowspan="2">$S$</td><td rowspan="2">$C$</td><td colspan="8">$F$</td></tr><tr><td>$a$</td><td>$b$</td><td>$C$</td><td>$d$</td><td>$e$</td><td>$f$</td><td>$g$</td><td>$h$</td></tr><tr><td>$a$</td><td>0.84</td><td>0.87</td><td>0.02</td><td>0.95</td><td>0.01</td><td>0.03</td><td>0.93</td><td>0.97</td><td>0.98</td><td>0.01</td></tr><tr><td>$b$</td><td>0.13</td><td>0.16</td><td>0.45</td><td>0.01</td><td>0.97</td><td>0.04</td><td>0.02</td><td>0.03</td><td>0.06</td><td>0.03</td></tr><tr><td>$C$</td><td>0.13</td><td>0.15</td><td>0.02</td><td>0.94</td><td>0.11</td><td>0.99</td><td>0.03</td><td>0.16</td><td>0.15</td><td>0.15</td></tr><tr><td>$d$</td><td>0.14</td><td>0.15</td><td>0.01</td><td>0.06</td><td>0.88</td><td>0.08</td><td>0.01</td><td>0.03</td><td>0.07</td><td>0.02</td></tr><tr><td>$e$</td><td>0.84</td><td>0.85</td><td>0.32</td><td>0.06</td><td>0.05</td><td>0.03</td><td>0.04</td><td>0.97</td><td>0.07</td><td>0.06</td></tr><tr><td>$f$</td><td>0.81</td><td>0.19</td><td>0.34</td><td>0.11</td><td>0.08</td><td>0.04</td><td>0.42</td><td>0.08</td><td>0.06</td><td>0.05</td></tr><tr><td>$g$</td><td>0.82</td><td>0.19</td><td>0.81</td><td>0.26</td><td>0.19</td><td>0.30</td><td>0.06</td><td>0.28</td><td>0.00</td><td>0.94</td></tr><tr><td>$h$</td><td>0.14</td><td>0.17</td><td>0.05</td><td>0.25</td><td>0.26</td><td>0.16</td><td>0.20</td><td>0.14</td><td>0.72</td><td>0.01</td></tr></table>
<table><tr><td rowspan="2"></td><td rowspan="2">$S$</td><td rowspan="2">$C$</td><td colspan="8">$F$</td></tr><tr><td>$a$</td><td>$b$</td><td>$C$</td><td>$d$</td><td>$e$</td><td>$f$</td><td>$g$</td><td>$h$</td></tr><tr><td>$a$</td><td>0.84</td><td>0.87</td><td>0.02</td><td>0.95</td><td>0.01</td><td>0.03</td><td>0.93</td><td>0.97</td><td>0.98</td><td>0.01</td></tr><tr><td>$b$</td><td>0.13</td><td>0.16</td><td>0.45</td><td>0.01</td><td>0.97</td><td>0.04</td><td>0.02</td><td>0.03</td><td>0.06</td><td>0.03</td></tr><tr><td>$C$</td><td>0.13</td><td>0.15</td><td>0.02</td><td>0.94</td><td>0.11</td><td>0.99</td><td>0.03</td><td>0.16</td><td>0.15</td><td>0.15</td></tr><tr><td>$d$</td><td>0.14</td><td>0.15</td><td>0.01</td><td>0.06</td><td>0.88</td><td>0.08</td><td>0.01</td><td>0.03</td><td>0.07</td><td>0.02</td></tr><tr><td>$e$</td><td>0.84</td><td>0.85</td><td>0.32</td><td>0.06</td><td>0.05</td><td>0.03</td><td>0.04</td><td>0.97</td><td>0.07</td><td>0.06</td></tr><tr><td>$f$</td><td>0.81</td><td>0.19</td><td>0.34</td><td>0.11</td><td>0.08</td><td>0.04</td><td>0.42</td><td>0.08</td><td>0.06</td><td>0.05</td></tr><tr><td>$g$</td><td>0.82</td><td>0.19</td><td>0.81</td><td>0.26</td><td>0.19</td><td>0.30</td><td>0.06</td><td>0.28</td><td>0.00</td><td>0.94</td></tr><tr><td>$h$</td><td>0.14</td><td>0.17</td><td>0.05</td><td>0.25</td><td>0.26</td><td>0.16</td><td>0.20</td><td>0.14</td><td>0.72</td><td>0.01</td></tr></table>


<table><tr><td rowspan="2"></td><td rowspan="2">$S$</td><td rowspan="2">$C$</td><td colspan="6">$F$</td></tr><tr><td>$i$</td><td>$j$</td><td>$k$</td><td>l</td><td>$m$</td><td>$n$</td></tr><tr><td>$i$</td><td>0.83</td><td>0.86</td><td>0.02</td><td>0.91</td><td>0.01</td><td>0.03</td><td>0.97</td><td>0.01</td></tr><tr><td>$j$</td><td>0.19</td><td>0.22</td><td>0.73</td><td>0.03</td><td>0.00</td><td>0.04</td><td>0.02</td><td>0.05</td></tr><tr><td>$k$</td><td>0.14</td><td>0.34</td><td>0.17</td><td>0.07</td><td>0.04</td><td>0.97</td><td>0.04</td><td>0.02</td></tr><tr><td>$l$</td><td>0.16</td><td>0.19</td><td>0.11</td><td>0.12</td><td>0.15</td><td>0.06</td><td>0.05</td><td>0.03</td></tr><tr><td>$m$</td><td>0.14</td><td>0.17</td><td>0.96</td><td>0.07</td><td>0.02</td><td>0.11</td><td>0.00</td><td>0.92</td></tr><tr><td>$n$</td><td>0.84</td><td>0.86</td><td>0.13</td><td>0.28</td><td>0.01</td><td>0.24</td><td>0.69</td><td>0.02</td></tr></table>
<table><tr><td rowspan="2"></td><td rowspan="2">$S$</td><td rowspan="2">$C$</td><td colspan="6">$F$</td></tr><tr><td>$i$</td><td>$j$</td><td>$k$</td><td>l</td><td>$m$</td><td>$n$</td></tr><tr><td>$i$</td><td>0.83</td><td>0.86</td><td>0.02</td><td>0.91</td><td>0.01</td><td>0.03</td><td>0.97</td><td>0.01</td></tr><tr><td>$j$</td><td>0.19</td><td>0.22</td><td>0.73</td><td>0.03</td><td>0.00</td><td>0.04</td><td>0.02</td><td>0.05</td></tr><tr><td>$k$</td><td>0.14</td><td>0.34</td><td>0.17</td><td>0.07</td><td>0.04</td><td>0.97</td><td>0.04</td><td>0.02</td></tr><tr><td>$l$</td><td>0.16</td><td>0.19</td><td>0.11</td><td>0.12</td><td>0.15</td><td>0.06</td><td>0.05</td><td>0.03</td></tr><tr><td>$m$</td><td>0.14</td><td>0.17</td><td>0.96</td><td>0.07</td><td>0.02</td><td>0.11</td><td>0.00</td><td>0.92</td></tr><tr><td>$n$</td><td>0.84</td><td>0.86</td><td>0.13</td><td>0.28</td><td>0.01</td><td>0.24</td><td>0.69</td><td>0.02</td></tr></table>


<table><tr><td></td><td colspan="2">$a,\ldots ,h,i,\ldots ,n$</td></tr><tr><td>$\forall x\neg F\left( {x,x}\right)$</td><td colspan="2">0.98</td></tr><tr><td>$\forall {xy}\left( {F\left( {x,y}\right)  \rightarrow  F\left( {y,x}\right) }\right)$</td><td></td><td>0.90</td></tr><tr><td>$\forall x\left( {S\left( x\right)  \rightarrow  C\left( x\right) }\right)$</td><td colspan="2">0.77</td></tr><tr><td>$\forall x\left( {S\left( x\right)  \land  F\left( {x,y}\right)  \rightarrow  S\left( y\right) }\right)$</td><td>0.96</td><td>0.92</td></tr><tr><td>$\forall x\exists y\left( {F\left( {x,y}\right) }\right)$</td><td colspan="2">1.0</td></tr></table>
<table><tr><td></td><td colspan="2">$a,\ldots ,h,i,\ldots ,n$</td></tr><tr><td>$\forall x\neg F\left( {x,x}\right)$</td><td colspan="2">0.98</td></tr><tr><td>$\forall {xy}\left( {F\left( {x,y}\right)  \rightarrow  F\left( {y,x}\right) }\right)$</td><td></td><td>0.90</td></tr><tr><td>$\forall x\left( {S\left( x\right)  \rightarrow  C\left( x\right) }\right)$</td><td colspan="2">0.77</td></tr><tr><td>$\forall x\left( {S\left( x\right)  \land  F\left( {x,y}\right)  \rightarrow  S\left( y\right) }\right)$</td><td>0.96</td><td>0.92</td></tr><tr><td>$\forall x\exists y\left( {F\left( {x,y}\right) }\right)$</td><td colspan="2">1.0</td></tr></table>


Learning and reasoning on ${\mathcal{K}}_{exp2} = {\mathcal{K}}_{a\ldots h}^{SFC} \cup  {\mathcal{K}}_{i\ldots n}^{SF} \cup  {\mathcal{K}}^{SFC}$
学习与推理在 ${\mathcal{K}}_{exp2} = {\mathcal{K}}_{a\ldots h}^{SFC} \cup  {\mathcal{K}}_{i\ldots n}^{SF} \cup  {\mathcal{K}}^{SFC}$


Table 1. the constants generated by the LTN. For instance, $\mathcal{G}\left( c\right)$ and $\mathcal{G}\left( g\right)$ happen to present a high cosine similarity measure. As a result,facts about the friendship relations of $c$ affect the friendship relations of $g$ and vice-versa,for instance $F\left( {c,b}\right)$ and $F\left( {g,b}\right)$ . The level of satisfiability associated with ${\mathcal{K}}_{\text{ exp1 }} \approx  1$ ,which indicates that ${\mathcal{K}}_{\text{ exp1 }}$ is classically satisfiable.
表1. 由 LTN 生成的常数。例如，$\mathcal{G}\left( c\right)$ 与 $\mathcal{G}\left( g\right)$ 的余弦相似度恰好很高。因此，关于 $c$ 的友谊关系的事实会影响 $g$ 的友谊关系，反之亦然，例如 $F\left( {c,b}\right)$ 与 $F\left( {g,b}\right)$ 。与 ${\mathcal{K}}_{\text{ exp1 }} \approx  1$ 相关的可满足性水平表明 ${\mathcal{K}}_{\text{ exp1 }}$ 在经典意义上可满足。


The results of the second experiment show that more facts can be learned with the inclusion of background knowledge. For example,the LTN now predicts that $C\left( i\right)$ and $C\left( n\right)$ are true. Similarly,from the symmetry of the friendship relation,the LTN concludes that $m$ is a friend of $i$ ,as expected. In fact,all the axioms in the generic background knowledge ${\mathcal{K}}^{SFC}$ are satisfied with a degree of satisfiability higher than ${90}\%$ , apart from the smoking causes cancer axiom - which is responsible for the classical inconsistency since in the data $f$ and $g$ smoke and do not have cancer -,which has a degree of satisfiability of 77%.
第二次实验的结果表明，若加入背景知识，可以学到更多事实。例如，LTN 现在预测 $C\left( i\right)$ 和 $C\left( n\right)$ 为真。类似地，由朋友关系的对称性，LTN 得出 $m$ 是 $i$ 的朋友，这符合预期。事实上，通用背景知识 ${\mathcal{K}}^{SFC}$ 中的所有公理在可满足度上均高于 ${90}\%$，但除了“吸烟导致癌症”的公理—它导致经典的不一致，因为在数据中 $f$ 和 $g$ 吸烟且没有癌症—该公理的可满足度为 77%。


## 6 Related work
## 6 相关工作


In his recent note, [15], Guha advocates the need for a new model theory for distributed representations (such as those based on embeddings). The note sketches a proposal, where terms and (binary) predicates are all interpreted as points/vectors in an $n$ -dimensional real space. The computation of the truth-value of the atomic formulae $P\left( {{t}_{1},\ldots ,{t}_{n}}\right)$ is obtained by comparing the projections of the vector associated to each ${t}_{i}$ with that associated to ${P}_{i}$ . Real logic shares with [15] the idea that terms must be interpreted in a geometric space. It has, however, a different (and more general) interpretation of functions and predicate symbols. Real logic is more general because the semantics proposed in [15] can be implemented within an Itn with a single layer $\left( {k = 1}\right)$ ,since the operation of projection and comparison necessary to compute the truth-value of $P\left( {{t}_{1},\ldots ,{t}_{m}}\right)$ can be encoded within an ${nm} \times  {nm}$ matrix $W$ with the constraint that ${\left\langle  \mathcal{G}\left( {t}_{1}\right) ,\ldots ,\mathcal{G}\left( {t}_{n}\right) \right\rangle  }^{T}W\left\langle  {\mathcal{G}\left( {t}_{1}\right) ,\ldots ,\mathcal{G}\left( {t}_{n}\right) }\right\rangle   \leq  \delta$ ,which can be encoded easily in 1tn.
在他最近的笔记中，[15]，Guha 主张需要一种新的分布式表示的模型理论（如基于嵌入的表示）。该笔记勾勒了一个提案，其中术语和（二元）谓词都在一个 $n$-维的实数空间中被解释为点/向量。原子公式 $P\left( {{t}_{1},\ldots ,{t}_{n}}\right)$ 真值的计算是通过比较与 ${t}_{i}$ 相关的向量的投影与与 ${P}_{i}$ 相关的投影来获得。实数逻辑与 [15] 共享的想法是，术语必须在几何空间中解释。然而，它对函数与谓词符号有不同（更一般）的解释。实数逻辑更通用，因为 [15] 中提出的语义可以在一个单层 $\left( {k = 1}\right)$ 的 Itn 内实现，因为计算 $P\left( {{t}_{1},\ldots ,{t}_{m}}\right)$ 真值所需的投影与比较的操作可以编码在一个 ${nm} \times  {nm}$ 矩阵 $W$ 中，且限制 ${\left\langle  \mathcal{G}\left( {t}_{1}\right) ,\ldots ,\mathcal{G}\left( {t}_{n}\right) \right\rangle  }^{T}W\left\langle  {\mathcal{G}\left( {t}_{1}\right) ,\ldots ,\mathcal{G}\left( {t}_{n}\right) }\right\rangle   \leq  \delta$ ，这一点可以很容易地在 1tn 中编码。


Real logic is orthogonal to the approach taken by (Hybrid) Markov Logic Networks (MLNs) and its variations [24,29,22]. In MLNs, the level of truth of a formula is determined by the number of models that satisfy the formula: the more models, the higher the degree of truth. Hybrid MLNs introduce a dependency from the real features associated to constants, which is given, and not learned. In real logic, instead, the level of truth of a complex formula is determined by (fuzzy) logical reasoning, and the relations between the features of different objects is learned through error minimization. Another difference is that MLNs work under the closed world assumption, while Real Logic is open domain. Much work has been done also on neuro-fuzzy approaches [19]. These are essentially propositional while real logic is first-order.
实数逻辑与（混合）马尔可夫逻辑网络（MLNs）及其变体的方法正交。在 MLNs 中，一个公式的真值水平由满足该公式的模型数量决定：模型越多，真度越高。混合 MLN 引入了对常数所关联的实数特征之间的依赖性，这种依赖是给定的、而非学习的。相比之下，在实数逻辑中，复杂公式的真值水平由（模糊）逻辑推理决定，不同对象特征之间的关系通过误差最小化来学习。另一个区别在于 MLNs 在封闭世界假设下工作，而实数逻辑是开放域。对神经模糊方法也做了大量工作 [19]。这些本质上是命题的，而实数逻辑是一阶的。


Bayesian logic (BLOG) [20] is open domain, and in this respect similar to real logic and LTNs. But, instead of taking an explicit probabilistic approach, LTNs draw from the efficient approach used by tensor networks for knowledge graphs, as already discussed. LTNs can have a probabilistic interpretation but this is not a requirement. Other statistical AI and probabilistic approaches such as lifted inference fall into this category, including probabilistic variations of inductive logic programming (ILP) [23], which are normally restricted to Horn clauses. Metainterpretive ILP [21], together with BLOG, seem closer to LTNs in what concerns the knowledge representation language, but do not explore the benefits of tensor networks for computational efficiency.
贝叶斯逻辑（BLOG）[20] 属于开放领域，在这一点上与真实逻辑和 LTNs 相似。但 LTNs 并非采用显式的概率方法，而是借用张量网络在知识图谱中的高效方法，正如前文所述。LTNs 可以有概率解释，但这并非必需。其他统计 AI 与概率方法，如提升推断，亦属于此范畴，包括 inductive logic programming（ILP）的概率变体[23]，通常限定于 Horn 子句。元解释性 ILP[21] 与 BLOG 一同，在知识表示语言方面似乎更接近 LTNs，但并未利用张量网络在计算效率方面的优势。


An approach for embedding logical knowledge onto data for the purpose of relational learning, similar to Real Logic, is presented in [25]. Real Logic and [25] share the idea of interpreting a logical alphabet in an $n$ -dimensional real space. Terminologically, the term "grounding" in Real Logic corresponds to "embeddings" in [25]. However, there are several differences. First, [25] uses function-free langauges, while we provide also groundings for functional symbols. Second, the model used to compute the truth-values of atomic formulas adopted in [25] is a special case of the more general model proposed in this paper (as described in Eq. (2)). Finally, the semantics of the universal and existential quantifiers adopted in [25] is based on the closed-world assumption (CWA), i.e. universally (respectively, existentially) quantified formulas are reduced to the finite conjunctions (respectively, disjunctions) of all of their possible instantiations; Real Logic does not make the CWA. Furthermore, Real Logic does not assume a specific t-norm.
一种将逻辑知识嵌入数据以进行关系学习的办法，类似 Real Logic，发表于 [25]。Real Logic 与 [25] 共享在实数空间的 $n$ 维度内对逻辑字母表进行解释的思路。从术语上讲，Real Logic 中的“ grounding ”对应于 [25] 中的“ embeddings ”。然而，两者存在若干差异。首先，[25] 使用无函数语言，而我们也为函数符号提供了 grounding。其次，[25] 用于计算原子公式真值的模型是本文所提出更一般模型的特例（如式(2)所述）。最后，[25] 采用的全称与存在量词的语义基于封闭世界假设（CWA），即对任意的全称/存在性量化公式，均化为其所有可能实例的有限合取（或有限析取）；Real Logic 不采用 CWA，且不假设特定的 t-范数。


As in [11], LTN is a framework for learning in the presence of logical constraints. LTNs share with [11] the idea that logical constraints and training examples can be treated uniformly as supervisions of a learning algorithm. LTN introduces two novelties: first, in LTN existential quantifiers are not grounded into a finite disjunction, but are scolemized. In other words, CWA is not required, and existentially quantified formulas can be satisfied by "new individuals". Second, LTN allows one to generate data for prediction. For instance,if a grounded theory contains the formula $\forall x\exists {yR}\left( {x,y}\right)$ ,LTN generates a real function (corresponding to the grounding of the Skolem function introduced by the formula) which for every vector $\mathbf{v}$ returns the feature vector $f\left( \mathbf{v}\right)$ ,which can be intuitively interpreted as being the set of features of a typical object which takes part in relation $R$ with the object having features equal to $\mathbf{v}$ .
与 [11] 一样，LTN 是一个在存在逻辑约束条件下进行学习的框架。LTN 与 [11] 共享的理念是：逻辑约束与训练样本可以被同等对待，作为学习算法的监督。LTN 引入了两点新意：第一，在 LTN 中存在量化不被 grounding 为有限的析取，而是采用 scolemized（注：此处原文为“skeletalized/scolemized”可能为拼写错误，保留原文含义）成分。换言之，不要求 CWA，存在量化公式可以被“新个体”满足。第二，LTN 允许生成用于预测的数据。例如，如果一个 grounding 理论包含公式 $\forall x\exists {yR}\left( {x,y}\right)$，LTN 会生成一个实数函数（对应公式引入的 Skolem 函数的 grounding），对于每一个向量 $\mathbf{v}$ 返回特征向量 $f\left( \mathbf{v}\right)$，可直观理解为参与关系 $R$ 的典型对象的特征集合，其对象的特征等于 $\mathbf{v}$。


Finally, related work in the domain of neural-symbolic computing and neural network fibring [10] has sought to combine neural networks with ILP to gain efficiency [14] and other forms of knowledge representation, such as propositional modal logic and logic programming. The above are more tightly-coupled approaches. In contrast, LTNs use a richer FOL language, exploit the benefits of knowledge compilation and tensor networks within a more loosely- coupled approach, and might even offer an adequate representation of equality in logic. Experimental evaluations and comparison with other neural-symbolic approaches are desirable though, including the latest developments in the field, a good snapshot of which can be found in [1].
最后，神经符号计算与神经网络纤维化领域的相关工作[10] 已尝试将神经网络与 ILP 结合以提升效率[14]及实现其他形式的知识表示，如命题模态逻辑与逻辑程序设计。这些方法相对更紧耦合。相比之下，LTNs 使用更丰富的 FOL 语言，在更松耦合的方式中利用知识编译和张量网络的优势，甚至可能提供对逻辑中等号的恰当表示。尽管如此，仍有必要进行实验评估并与其他神经符号方法进行比较，包括领域的最新发展，其良好概览见 [1]。


## 7 Conclusion and future work
## 7 结论与未来工作


We have proposed Real Logic: a uniform framework for learning and reasoning. Approximate satisfiability is defined as a learning task with both knowledge and data being mapped onto real-valued vectors. With an inference-as-learning approach, relational knowledge constraints and state-of-the-art data-driven approaches can be integrated. We showed how real logic can be implemented in deep tensor networks, which we call Logic Tensor Networks (LTNs), and applied efficiently to knowledge completion and data prediction tasks. As future work, we will make the implementation of LTN available in TENSORFLOW ${}^{\mathrm{{TM}}}$ and apply it to large-scale experiments and relational learning benchmarks for comparison with statistical relational learning, neural-symbolic computing, and (probabilistic) inductive logic programming approaches.
我们提出了 Real Logic：一个用于学习与推理的统一框架。近似可满足性被定义为一个学习任务，其中知识与数据都映射到实值向量。通过推理即学习的方式，可以将关系知识约束与最先进的数据驱动方法整合。我们展示了如何在深度张量网络中实现 Real Logic，我们称之为 Logic Tensor Networks（LTNs），并高效应用于知识完成与数据预测任务。未来工作中，我们将把 LTNs 的实现开放给 TensorFlow ${}^{\mathrm{{TM}}}$，并将其应用于大规模实验及关系学习基准，与统计关系学习、神经符号计算及（概率性）归纳逻辑编程等方法进行比较。


## References
## 参考文献


1. Cognitive Computation: Integrating Neural and Symbolic Approaches, Workshop at NIPS 2015, Montreal, Canada, April 2016. CEUR-WS 1583.
1. Cognitive Computation: Integrating Neural and Symbolic Approaches, Workshop at NIPS 2015, Montreal, Canada, April 2016. CEUR-WS 1583.


2. Knowledge Representation and Reasoning: Integrating Symbolic and Neural Approaches, AAAI Spring Symposium, Stanford University, CA, USA, March 2015.
2. Knowledge Representation and Reasoning: Integrating Symbolic and Neural Approaches, AAAI Spring Symposium, Stanford University, CA, USA, March 2015。


3. Dimitris Achlioptas. Random satisfiability. In Handbook of Satisfiability, pages 245-270. 2009.
3. Dimitris Achlioptas. 随机可满足性。收录于《可满足性手册》，第245-270页。2009。


4. Leon Barrett, Jerome Feldman, and Liam MacDermed. A (somewhat) new solution to the variable binding problem. Neural Computation, 20(9):2361-2378, 2008.
4. Leon Barrett, Jerome Feldman, and Liam MacDermed. 对变量绑定问题的（有些）新解法。神经计算，20(9):2361-2378，2008。


5. Yoshua Bengio. Learning deep architectures for ai. Found. Trends Mach. Learn., 2(1):1-127, January 2009.
5. Yoshua Bengio. 为 AI 学习深度结构。Found. Trends Mach. Learn., 2(1):1-127，2009年1月。


6. M. Bergmann. An Introduction to Many-Valued and Fuzzy Logic: Semantics, Algebras, and Derivation Systems. Cambridge University Press, 2008.
6. M. Bergmann. 多值与模糊逻辑导论：语义、代数与推导系统。剑桥大学出版社，2008。


7. David M. Blei, Andrew Y. Ng, and Michael I. Jordan. Latent dirichlet allocation. J. Mach. Learn. Res., 3:993-1022, March 2003.
7. David M. Blei, Andrew Y. Ng, and Michael I. Jordan. 潜在狄利克雷分配。J. Mach. Learn. Res., 3:993-1022，2003年3月。


8. Léon Bottou. From machine learning to machine reasoning. Technical report, arXiv.1102.1808, February 2011.
8. Léon Bottou. 从机器学习到机器推理。技术报告，arXiv.1102.1808，2011年2月。


9. Artur S. d'Avila Garcez, Marco Gori, Pascal Hitzler, and Luís C. Lamb. Neural-symbolic learning and reasoning (dagstuhl seminar 14381). Dagstuhl Reports, 4(9):50-84, 2014.
9. Artur S. d'Avila Garcez, Marco Gori, Pascal Hitzler, and Luís C. Lamb. 神经-符号学习与推理（Dagstuhl 研讨会 14381）。Dagstuhl Reports, 4(9):50-84，2014。


10. Artur S. d'Avila Garcez, Luís C. Lamb, and Dov M. Gabbay. Neural-Symbolic Cognitive Reasoning. Cognitive Technologies. Springer, 2009.
10. Artur S. d'Avila Garcez, Luís C. Lamb, and Dov M. Gabbay. 神经-符号认知推理。Cognitive Technologies. Springer，2009。


11. Michelangelo Diligenti, Marco Gori, Marco Maggini, and Leonardo Rigutini. Bridging logic and kernel machines. Machine Learning, 86(1):57-88, 2012.
11. Michelangelo Diligenti, Marco Gori, Marco Maggini, and Leonardo Rigutini. 架桥逻辑与核方法。Machine Learning, 86(1):57-88，2012。


12. David Silver et al. Mastering the game of go with deep neural networks and tree search. Nature, 529:484-503, 2016.
12. David Silver 等. 使用深度神经网络与树搜索征服围棋。Nature, 529:484-503，2016。


13. Martín Abadi et al. TensorFlow: Large-scale machine learning on heterogeneous systems, 2015. Software available from tensorflow.org.
13. Martín Abadi 等. TensorFlow：在异构系统上的大规模机器学习，2015。软件可从 tensorflow.org 获取。


14. Manoel V. M. França, Gerson Zaverucha, and Artur S. d'Avila Garcez. Fast relational learning using bottom clause propositionalization with artificial neural networks. Machine Learning, 94(1):81-104, 2014.
14. Manoel V. M. França, Gerson Zaverucha, and Artur S. d'Avila Garcez. 使用底部子句命题化进行快速关系学习的人工神经网络。Machine Learning, 94(1):81-104，2014。


15. Ramanathan Guha. Towards a model theory for distributed representations. In 2015 AAAI Spring Symposium Series, 2015.
15. Ramanathan Guha. 面向分布表示的模型理论。见 2015 年 AAAI 春季研讨会系列，2015。


16. Michael Huth and Mark Ryan. Logic in Computer Science: Modelling and Reasoning About Systems. Cambridge University Press, New York, NY, USA, 2004.
16. Michael Huth and Mark Ryan. 计算机科学中的逻辑：建模与系统推理。剑桥大学出版社，纽约，USA，2004。


17. Jeffrey O. Kephart and David M. Chess. The vision of autonomic computing. Computer, 36(1):41-50, January 2003.
17. Jeffrey O. Kephart and David M. Chess. 自主计算的愿景。Computer, 36(1):41-50，2003年1月。


18. Douwe Kiela and Léon Bottou. Learning image embeddings using convolutional neural networks for improved multi-modal semantics. In Proceedings of EMNLP 2014, Doha, Qatar, 2014.
18. Douwe Kiela and Léon Bottou. 使用卷积神经网络学习图像嵌入以提升多模态语义。EMNLP 2014 会议论文，卡塔尔多哈，2014。


19. Bart Kosko. Neural Networks and Fuzzy Systems: A Dynamical Systems Approach to Machine Intelligence. Prentice-Hall, Inc., Upper Saddle River, NJ, USA, 1992.
19. Bart Kosko. 神经网络与模糊系统：一种对机器智能的动力系统方法。Prentice-Hall, Inc., Upper Saddle River, NJ, USA, 1992.


20. Brian Milch, Bhaskara Marthi, Stuart J. Russell, David Sontag, Daniel L. Ong, and Andrey Kolobov. BLOG: probabilistic models with unknown objects. In IJCAI-05, Proceedings of the Nineteenth International Joint Conference on Artificial Intelligence, Edinburgh, Scotland, UK, July 30-August 5, 2005, pages 1352-1359, 2005.
20. Brian Milch, Bhaskara Marthi, Stuart J. Russell, David Sontag, Daniel L. Ong, and Andrey Kolobov. BLOG: probabilistic models with unknown objects. In IJCAI-05, Proceedings of the Nineteenth International Joint Conference on Artificial Intelligence, Edinburgh, Scotland, UK, July 30-August 5, 2005, pages 1352-1359, 2005.


21. Stephen H. Muggleton, Dianhuan Lin, and Alireza Tamaddoni-Nezhad. Meta-interpretive learning of higher-order dyadic datalog: predicate invention revisited. Machine Learning, 100(1):49-73, 2015.
21. Stephen H. Muggleton, Dianhuan Lin, and Alireza Tamaddoni-Nezhad. Meta-interpretive learning of higher-order dyadic datalog: predicate invention revisited. Machine Learning, 100(1):49-73, 2015.


22. Aniruddh Nath and Pedro M. Domingos. Learning relational sum-product networks. In Proceedings of the Twenty-Ninth AAAI Conference on Artificial Intelligence, January 25-30, 2015, Austin, Texas, USA., pages 2878-2886, 2015.
22. Aniruddh Nath and Pedro M. Domingos. Learning relational sum-product networks. In Proceedings of the Twenty-Ninth AAAI Conference on Artificial Intelligence, January 25-30, 2015, Austin, Texas, USA., pages 2878-2886, 2015.


23. Luc De Raedt, Kristian Kersting, Sriraam Natarajan, and David Poole. Statistical Relational Artificial Intelligence: Logic, Probability, and Computation. Synthesis Lectures on Artificial Intelligence and Machine Learning. Morgan & Claypool Publishers, 2016.
23. Luc De Raedt, Kristian Kersting, Sriraam Natarajan, and David Poole. Statistical Relational Artificial Intelligence: Logic, Probability, and Computation. Synthesis Lectures on Artificial Intelligence and Machine Learning. Morgan & Claypool Publishers, 2016.


24. Matthew Richardson and Pedro Domingos. Markov logic networks. Mach. Learn., 62(1- 2):107-136, February 2006.
24. Matthew Richardson and Pedro Domingos. Markov logic networks. Mach. Learn., 62(1- 2):107-136, February 2006.


25. Tim Rocktaschel, Sameer Singh, and Sebastian Riedel. Injecting logical background knowledge into embeddings for relation extraction. In Annual Conference of the North American Chapter of the Association for Computational Linguistics (NAACL), June 2015.
25. Tim Rocktaschel, Sameer Singh, and Sebastian Riedel. Injecting logical background knowledge into embeddings for relation extraction. In Annual Conference of the North American Chapter of the Association for Computational Linguistics (NAACL), June 2015.


26. Richard Socher, Danqi Chen, Christopher D. Manning, and Andrew Y. Ng. Reasoning With Neural Tensor Networks For Knowledge Base Completion. In Advances in Neural Information Processing Systems 26. 2013.
26. Richard Socher, Danqi Chen, Christopher D. Manning, and Andrew Y. Ng. Reasoning With Neural Tensor Networks For Knowledge Base Completion. In Advances in Neural Information Processing Systems 26. 2013.


27. T. Tieleman and G. Hinton. Lecture 6.5 - RMSProp, COURSERA: Neural networks for machine learning. Technical report, 2012.
27. T. Tieleman and G. Hinton. Lecture 6.5 - RMSProp, COURSERA: Neural networks for machine learning. Technical report, 2012.


28. Leslie G. Valiant. Robust logics. In Proceedings of the Thirty-first Annual ACM Symposium on Theory of Computing, STOC '99, pages 642-651, New York, NY, USA, 1999. ACM.
28. Leslie G. Valiant. Robust logics. In Proceedings of the Thirty-first Annual ACM Symposium on Theory of Computing, STOC '99, pages 642-651, New York, NY, USA, 1999. ACM.


29. Jue Wang and Pedro M. Domingos. Hybrid markov logic networks. In Proceedings of the Twenty-Third AAAI Conference on Artificial Intelligence, AAAI 2008, Chicago, Illinois, USA, July 13-17, 2008, pages 1106-1111, 2008.
29. Jue Wang and Pedro M. Domingos. Hybrid markov logic networks. In Proceedings of the Twenty-Third AAAI Conference on Artificial Intelligence, AAAI 2008, Chicago, Illinois, USA, July 13-17, 2008, pages 1106-1111, 2008.