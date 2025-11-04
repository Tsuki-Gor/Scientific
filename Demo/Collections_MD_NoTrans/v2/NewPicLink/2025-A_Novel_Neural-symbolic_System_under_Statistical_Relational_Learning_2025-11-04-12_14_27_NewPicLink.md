

<!-- Meanless: 1 MSRON HIRAL NEONSSONEKE-->

# A Neural-symbolic Framework under Statistical Relational Learning

Dongran Yu, Xueyan Liu, Shirui Pan, Anchen Li and Bo Yang

Abstract-A key objective in the field of artificial intelligence is to develop cognitive models that can exhibit human-like intellectual capabilities. One promising approach to achieving this is through neural-symbolic systems, which combine the strengths of deep learning and symbolic reasoning. However, current methodologies in this area face limitations in integration, generalization, and interpretability. To address these challenges, we propose a neural-symbolic framework based on statistical relational learning, referred to as NSF-SRL. This framework effectively integrates deep learning models with symbolic reasoning in a mutually beneficial manner. In NSF-SRL, the results of symbolic reasoning are utilized to refine and correct the predictions made by deep learning models, while deep learning models enhance the efficiency of the symbolic reasoning process. Through extensive experiments, we demonstrate that our approach achieves high performance and exhibits effective generalization in supervised learning, weakly supervised and zero-shot learning tasks. Furthermore, we introduce a quantitative strategy to evaluate the interpretability of the model's predictions, visualizing the corresponding logic rules that contribute to these predictions and providing insights into the reasoning process. We believe that this approach sets a new standard for neural-symbolic systems and will drive future research in the field of general artificial intelligence.

Index Terms-Neural-symbolic systems, Deep learning, Statistical relational learning, Markov logic networks.

## 1 INTRODUCTION

HUMAN cognitive systems encompass both perception and reasoning. Specifically, perception is primarily responsible for recognizing information, while reasoning handles logical deduction and analytical thinking. When humans process information, they integrate both perception and reasoning to enhance their comprehension and decision-making capabilities. Current artificial intelligence systems typically specialize in either perception or reasoning. For instance, deep learning models excel in perception, achieving remarkable performance in tasks that involve inductive learning and computational efficiency. In contrast, symbolic logic is adept at logical reasoning, providing strong results in deductive reasoning tasks, generalization, and interpretability. However, both models have inherent limitations. Deep learning models often operate as black boxes, lacking interpretability, generalizing poorly, and requiring vast amounts of training data to perform optimally. On the other hand, symbolic logic relies on search algorithms to explore solution spaces, resulting in slow reasoning in large-scale environments. Therefore, integrating the strengths of both models offers a way to combine perception and reasoning into a unified framework that more effectively mimics human cognitive processes. As Leslie G. Valiant argues, reconciling the statistical nature of learning with the logical nature of reasoning to create cognitive computing models that integrate concept learning and manipulation is one of the three fundamental challenges in computer science [1].

The neural-symbolic system represents a promising approach for effectively integrating perception and reasoning into a unified framework [2], [3], [4]. Various neural-symbolic systems have been proposed, which can be broadly classified into three categories [5]: learning-for-reasoning methods, reasoning-for-learning methods, and learning-reasoning methods. Learning-for-reasoning methods [6], [7], [8] primarily focus on symbolic reasoning. In these methods, deep learning models transform unstructured inputs into symbolic representations, which are then processed by symbolic reasoning models to derive solutions. In some cases, deep learning models replace search algorithms, thus accelerating symbolic reasoning. Reasoning-for-learning approaches [9], [10], [11] focus more on deep learning. Symbolic knowledge is encoded into distributed representations and integrated into deep learning models to compute results. However, these methods often use deep learning to support symbolic reasoning or incorporate symbolic priors to enhance deep learning, without fully achieving complementary integration. Few studies explore learning-reasoning methods, which aim for more comprehensive integration [12], [13]. For example, Manhaeve et al. [14] combine a deep learning model with a probabilistic logic programming language, where the output of the deep learning model serves as input for symbolic reasoning. Techniques like arithmetic circuits and gradient semi-rings enable interaction between the deep learning model and symbolic reasoning. Zhou [13] integrates machine learning with logic reasoning based on the principle of abduction, using the machine learning model's output as input for logical reasoning. This reasoning process iteratively corrects the model's output through consistency optimization, and the refined output is then used as supervised information for further training. While these approaches represent significant progress in neural-symbolic systems, achieving full integration remains a challenging and open problem, necessitating further exploration and research.

---

<!-- Footnote -->

- B. Yang (corresponding author), X. Liu (corresponding author) and A. Li are with the Key Laboratory of Symbolic Computation and Knowledge Engineer, Ministry of Education, Jilin University, Changchun, Jilin 130012, China and the School of Computer Science and Technology, Jilin University, Changchun, Jilin 130012, China. E-mail: ybo@jlu.edu.cn; xueyanliu@jlu.edu.cn; liac20@mails.jlu.edu.cn

- D. Yu is with the Key Laboratory of Symbolic Computation and Knowledge Engineer, Ministry of Education, Jilin University, Changchun, Jilin 130012, China, and the School of Artificial Intelligence, Jilin University, Changchun, Jilin 130012, China.

E-mail: yudran@foxmail.com

- S. Pan is with School of Information and Communication Technology, Griffith University, Brisbane 4222, Queensland, Australia. E-mail: s.pan@griffith.edu.au

<!-- Footnote -->

---




<!-- Meanless: 2-->

This paper introduces a novel framework called the Neural Symbolic Framework under Statistical Relational Learning (NSF-SRL for short), which aims to integrate deep learning models with symbolic logic in a mutually beneficial manner. In NSF-SRL, symbolic logic enhances deep learning models by making their predictions more logical, consistent with common sense, and interpretable, thereby improving their generalization capabilities. In turn, deep learning enhances symbolic logic by increasing its efficiency and robustness to noise. However, a key challenge in constructing the NSF-SRL framework is determining how to effectively combine deep learning and symbolic logic to model a joint probability distribution.

Statistical Relational Learning (SRL) [15] serves as a bridge between statistical models, such as deep learning, and relational models, like symbolic logic, by integrating the two approaches. Inspired by this framework, we employs SRL techniques to address the challenge of model construction. In this approach, deep learning processes data according to specific tasks and generates corresponding outputs, while symbolic logic learns a joint probability distribution based on these outputs and symbolic knowledge, thus constraining deep learning's predictions to achieve mutual enhancement. It is important to note that in our framework, deep learning not only functions as a data processor for symbolic logic but also replaces traditional search algorithms to improve computational efficiency. In this study, symbolic knowledge is represented using First-Order Logic (FOL). During the training phase,the model learns the basic concepts 1 in FOL from the sample data, a process we term concept learning. In the testing phase, the model utilizes existing or newly acquired FOLs to combine and manipulate learned concepts, thereby generating new ones-a process referred to as concept manipulation.

In summary, our contributions can be characterized in threefold:

- In this study, we propose a general neural-symbolic system framework NSF-SRL and develop an end-to-end model.

- The model employs statistical relational learning techniques to integrate deep learning and symbolic logic, thereby achieving mutual enhancement of learning and reasoning. This integration improves the model's generalization ability and interpretability.

- Based on our experimental results, we demonstrate that NSF-SRL outperforms comparable methods in various reasoning tasks, including supervised, weakly supervised, and zero-shot learning scenarios, with respect to performance and generalization. Additionally, we emphasize the interpretability of our model by providing visualizations that enhance the understanding of the reasoning process.

In our previous conference paper [16], we initially presented and validated the proposed approach for visual relationship detection. However, this current study significantly extends that work by introducing new model designs, such as concept manipulation, incorporating new tasks like digit image addition and zero-shot image classification, and comparing against additional baseline approaches. Furthermore, we provide extensive experimental validations and comparisons to thoroughly evaluate the model's performance.

## 2 RELATED WORK

Neural-symbolic systems. In recent times, neural-symbolic reasoning has gained significant attention and can be classified into three main groups [5]. The first group consists of methods where deep neural networks assist symbolic reasoning. These methods replace traditional search algorithms in symbolic reasoning with deep neural networks to reduce the search space and improve computation speed [6], [7], [8], [17]. For example, Qu et al. [6] proposed probabilistic Logic Neural Networks (pLogicNet), which addresses the problem of reasoning in knowledge graphs (triplet completion) as an inference problem involving hidden variables in a probabilistic graph model. The pLogicNet employs a combination of variational EM and neural networks to approximate the inference. Building on the idea of pLogicNet, Zhang et al. [7] introduced ExpressGNN, which leverages Graph Neural Networks (GNNs) as approximate inference methods for posterior calculation in the variable EM algorithm. Marra et al. [18] proposed NMLN, which reparametrizes the MLN through a neural network that is evaluated based on input features. The second group focuses on symbolic reasoning aiding deep learning models during the learning process. These methods incorporate symbolic knowledge into the training of deep learning models to enhance performance and interpretability [10], [11], [19], [20], [21]. Symbolic knowledge is often used as a regularizer during training. For instance, Xie et al. [10] encode symbolic knowledge into neural networks by designing a regularization term in the loss function for a specific task. The third group consists of models that strike a balance between deep learning models and symbolic reasoning, allowing both paradigms to contribute to problem-solving [12], [13], [22], [23]. Zhou [13] establishes a connection between machine learning and symbolic reasoning frameworks based on the characteristics of symbolic reasoning, such as abduction. Duan et al. [24] proposed a framework for joint learning of neural perception and logical reasoning, where the two components are mutually supervised and jointly optimized. Pryor et al. [25] introduced NeuPSL, where the neural network learns the predicates for logical reasoning, while logical reasoning imposes constraints on the neural network. Yang et al., [26] proposed NeurASP, which leverages a pre-trained neural network in symbolic computation and enhances the neural network's performance by applying symbolic reasoning. In contrast to the aforementioned methods, our approach takes a different route to bridge the gap between deep learning models and symbolic logic through statistical relational learning. By leveraging statistical relational learning, our method retains the full capabilities of both probabilistic reasoning and deep learning, offering a unique and powerful integration of the two paradigms.

Markov Logic Networks. To handle complexity and uncertainty of the real world, intelligent systems require a unified representation that combines first-order logic (FOL) and probabilistic graphical models. Markov Logic Networks (MLNs) achieve this by providing a unified framework that combines FOL and probabilistic graphical models into a single representation. MLN has been extensively studied and proven effective in various reasoning tasks, including knowledge graph reasoning [6], [7], semantic parsing [27], [28], and social network analysis [29]. MLN is capable of capturing complexity and uncertainty inherent in relational data. However, performing inference and learning in MLN can be computationally expensive due to the exponential cost of constructing the ground MLN and NP-complete optimization problem. This limitation hinders the practical application of MLN in large-scale scenarios. To address these challenges, many works have been proposed to improve accuracy and efficiency of MLN. For instance, some studies [30], [31] have focused on enhancing the accuracy of MLN, while others [6], [7], [32], [33], [34] have aimed to improve its efficiency. In particular, two studies [6], [7] have replaced traditional inference algorithms in MLN with neural networks. By leveraging neural networks, these approaches offer a more efficient alternative for performing inference in MLN. This integration of neural networks and MLN allows for more scalable and effective reasoning in large-scale applications.

---

<!-- Footnote -->

1. In this paper, concepts refer to predicates in FOL.

<!-- Footnote -->

---




<!-- Meanless: 3-->

## 3 PRELIMINARIES

In this section, we first introduce the neural-symbolic model definition and notations in this paper. Then, we will introduce the basic knowledge about statistic relational learning.

### 3.1 Model Description

The primary task in developing the model NSF-SRL is to formulate and maximize the posterior probability $P\left( {Y \mid  X,R;{\theta }_{1},{\theta }_{2},w}\right)$ , where $X = \left\{  {{x}_{1},{x}_{2},\ldots ,{x}_{n}}\right\}$ represents the observed data, $Y =$ $\left\{  {{y}_{1},{y}_{2},\ldots ,{y}_{n}}\right\}$ is the label set corresponding to data $X$ ,and $R = \left\{  {{r}_{1},{r}_{2},\ldots ,{r}_{m}}\right\}$ is the first-order logic rule set, ${\theta }_{1},{\theta }_{2}$ and $w$ denote the parameters of NSF-SRL. $n$ is the number of the instance of raw data,and $m$ is the number of rules. Given the training dataset $D = \left\{  {\left( {{x}_{1},{y}_{1}}\right) ,\left( {{x}_{2},{y}_{2}}\right) ,\ldots ,\left( {{x}_{n},{y}_{n}}\right) }\right\}$ and the first-order logic rules $R$ ,the learning process of NSF-SRL can be expressed as maximizing the posterior probability, formally defined as:

$$
\forall D\mathop{\max }\limits_{{{\theta }_{1},{\theta }_{2},w}}P\left( {Y \mid  X,R;{\theta }_{1},{\theta }_{2},w}\right) , \tag{1}
$$

For example,in image classification tasks,the input data $D$ represents images,while the output $y$ corresponds to the labels of the objects within those images. To enhance understanding of this paper, symbolic descriptions are provided in Table 1 These descriptions clarify the symbolic representations used throughout the study and facilitate comprehension of the concepts and methodologies discussed.

### 3.2 Statistical Relational Learning

Many tasks in real-world application domains are characterized by the presence of both uncertainty and complex relational structures. Statistical learning addresses the former, while relational learning focuses on the latter. Statistical Relational Learning (SRL) aims to harness the strengths of both approaches [15].

In this study, we leverage SRL to integrate first-order logic (FOL,rule body $\Rightarrow$ rule head) with probabilistic graphical models, creating a unified framework that facilitates probabilistic inference for reasoning problems. FOL represents a type of commonsense (symbolic) knowledge that is easily understood by humans. In this paper, we treat the FOL language as a means to describe knowledge in the form of logic rules, which provides strong expressive capability [35]. For instance, FOL allows for the definition of predicates and the description of various relations.

To achieve this integration, we employ Markov Logic Networks (MLNs), a well-known statistical relational learning model, to represent FOL as undirected graphs. In the constructed undirected graph, nodes are generated based on all ground atoms which are logical predicates with their arguments replaced by specific constants. In this paper, ${a}_{r}$ denotes assignments of variables to the arguments of an FOL $r$ ,and all consistent assignments are captured in the set ${A}_{r} = \left\{  {{a}_{r}^{1},{a}_{r}^{2},\ldots }\right\}$ . For instance, if we have a constant set $C = \left\{  {{c}_{1},{c}_{2}}\right\}$ and an FOL $r \in  R$ such as catlike $\left( x\right)  \land  \operatorname{tawn}y\left( x\right)  \land  \operatorname{spot}\left( x\right)  \Rightarrow$ leopard(x), the corresponding ground atoms ${A}_{r}$ can be generated such as \{catlike $\left( {c}_{1}\right)$ ,catlike $\left( {c}_{2}\right)$ ,tawny $\left( {c}_{1}\right)$ ,tawny $\left( {c}_{2}\right)$ ,spot $\left( {c}_{1}\right)$ ,spot $\left( {c}_{2}\right)$ ,leopard $\left. {\left( {c}_{1}\right) ,\text{leopard}\left( {c}_{2}\right) }\right\}$ . Furthermore,an edge is established between two nodes if the corresponding ground atoms co-occur in at least one ground FOL in the MLN. Consequently, a ground MLN can be formulated as a joint probability distribution, capturing the dependencies and correlations among the ground atoms. This joint probability distribution is expressed as:

$$
P\left( A\right)  = \frac{1}{Z\left( w\right) }\exp \left\{  {\mathop{\sum }\limits_{{r \in  R}}{w}_{r}\mathop{\sum }\limits_{{{a}_{r} \in  {A}_{r}}}\phi \left( {a}_{r}\right) }\right\}  , \tag{2}
$$

<!-- Media -->

TABLE 1

Important notations and their descriptions.

<table><tr><td>Notations</td><td>Descriptions</td></tr><tr><td>$D$</td><td>Set of input data</td></tr><tr><td>$Y$</td><td>Set of ground truths</td></tr><tr><td>$\widehat{y}$</td><td>Pseudo-label</td></tr><tr><td>$R$</td><td>Set of logical rules</td></tr><tr><td>$r$</td><td>A logic rule</td></tr><tr><td>${T}_{r}$</td><td>Triggered logic rule</td></tr><tr><td>$A$</td><td>Ground atom sets in knowledge base</td></tr><tr><td>${A}_{r}$</td><td>Ground atom sets in a logic rule</td></tr><tr><td>${a}_{r}$</td><td>A ground atom</td></tr><tr><td>$\phi$</td><td>Potential function</td></tr><tr><td>${\theta }_{1}$</td><td>Parameters of neural reasoning module</td></tr><tr><td>${\theta }_{2}$</td><td>Parameters of concept network</td></tr><tr><td>$W$</td><td>Weight sets of the logic rules</td></tr><tr><td>${w}_{r}$</td><td>Weight of a logic rule</td></tr></table>

<!-- Media -->

where $Z\left( w\right)  = \mathop{\sum }\limits_{A}\mathop{\sum }\limits_{{{A}_{r} \in  A,{a}_{r} \in  {A}_{r}}}\phi \left( {a}_{r}\right)$ is the partition function that sums over all ground atoms. $A$ represents all ground atoms in the knowledge base,while $\phi$ is a potential function reflecting the number of times a FOL statement is true. The variable $w$ denotes the weight sets of all FOLs,and ${w}_{r}$ refers to the weight of a specific FOL.

## 4 OUR METHOD: NSF-SRL

The goal of the NSF-SRL framework is to achieve a mutual integration of deep learning and symbolic logic. In this framework, deep learning can take the form of any task-related neural network, primarily responsible for feature extraction and result prediction. Symbolic logic, on the other hand, is grounded in probabilistic graphical models and is responsible for logical reasoning. In this section, we first provide an overview of our NSF-SRL in Section 4.1. We then present concept learning in Section 4.2, followed by a description of concept manipulation in Section 4.3

---

<!-- Footnote -->

2. Ground atom is a replacement of all of its arguments by constants. In this paper, we refer to the process of replacement as "grounding".

<!-- Footnote -->

---




<!-- Meanless: 4-->

<!-- Media -->

<!-- figureText: Concept learning Concept manipulation Transductive concept manipulation Inductive concept manipulation Test data original logic rules R1: catlike $\left( x\right)  \land$ tawny $\left( x\right)  \land$ spot(x) $\Rightarrow$ leopard(x) R2: horselike $\left( x\right)  \land$ white&black(x) $\land$ stripe $\left( x\right)  \Rightarrow$ zebra(x) new logic rules R3: $\operatorname{catlike}\left( x\right)  \land  \operatorname{tawn}y\left( x\right)  \land  \operatorname{stripe}\left( x\right)$ $\Rightarrow  \operatorname{tiger}\left( x\right)$ concept grounding stripe catlike Train data Learned concepts leopard catlike tawny spot spot tawny catlike R1: $\mathit{{catlike}}\left( x\right)  \land  \mathit{{tawny}}\left( x\right)$ $\land  \operatorname{spot}\left( x\right)  \Rightarrow$ leopard(x) zebra horselike white&black stripe white & black R2: horselike $\left( x\right)  \land$ white $\&$ black(x) $\land$ stripe $\left( x\right)  \Rightarrow$ zebra(x) -->

<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2025/11/2025_11_04__12_34_43_5b8b97.jpg"/>

Fig. 1. Overview of NSF-SRL. The concept learning phase acquires basic concepts such as "cattike", "tawny" and "spot" from the training data. In transductive concept manipulation, the learned concepts and toriginal rules are applied to test data whose labels were present in the training sets. This integration of learned concepts enhances the interpretability of NSF-SRL by providing insights into how predictions are made based on these concepts and the accompanying rules. Conversely, in inductive concept manipulation, the learned concepts serve as the rule body, and new rules are introduced to reason about samples with labels that have never appeared in the training set.

<!-- Media -->

### 4.1 Overview of NSF-SRL

An overview of the NSF-SRL framework, consisting of two key phases-concept learning and concept manipulation-is presented in Fig. 1

Concept learning focuses on acquiring fundamental concepts from training data. For instance, we can learn essential concepts such as "catlike", "tawny" and "spot" from images of leopards, and "horselike", "white&black" and "stripe" from images of zebras,utilizing the rules $\mathrm{R}1 : \operatorname{catlike}\left( x\right)  \land  \operatorname{tawny}\left( x\right)  \land  \operatorname{spot}\left( x\right)  \Rightarrow$ leopard(x)and R2: horselike $\left( x\right)  \land$ white&black $\left( x\right)  \land$ stripe $\left( x\right)  \Rightarrow$ $\operatorname{zebra}\left( x\right)$ .

Concept manipulation is used for reasoning and interpreting results, employing existing or newly acquired symbolic knowledge to combine established concepts and generate new ones. In this paper, we identify two types of conceptual operations: transduc-tive concept manipulation and inductive concept manipulation. In transductive concept manipulation, the learned concepts and original rules are utilized to test data whose labels have appeared in the training set. Incorporating these learned concepts enhances the interpretability of the NSF-SRL, providing insights into how prediction results are derived in conjunction with the rules. For example, the predicted label "leopard" can be attributed to rule R1. Conversely, in inductive concept manipulation, the learned concepts and the new rules are applied to test data whose label has never appeared in the training set. Specifically, the learned concepts serve as the rule body of a new rule, which is used to reason the rule head as the output when testing a new sample. For instance, when an image containing a tiger is fed into the well-trained model, it can trigger the new rule R3 and generate corresponding ground atoms such as "catlike", "tawny" and "stripe" via concept grounding. By leveraging R3 and the ground atoms, the model infers the new concept "tiger". Inductive concept manipulation enables the application of previously learned concepts to new tasks, facilitating the generation of new concepts through inference and realizing adaptation and generalization to new tasks. In summary, through the process of concept manipulation, the NSF-SRL effectively learns, reasons, and produces explainable results by leveraging learned concepts.

### 4.2 Concept Learning

Concept learning involves a Neural Reasoning Module (NRM) and a Symbolic Reasoning Module (SRM), as illustrated in Fig. 2 These two modules engage in end-to-end joint learning to produce a trained model. Specifically, the NRM functions as a task network, generating pseudo-labels and feature vectors. In contrast, the SRM operates as a probabilistic graphical model responsible for deriving reasoning outcomes. During the training process, the SRM constrains the parameter learning of the NRM, enhancing the accuracy and interpretability of its predictions. After $N$ iterations and corresponding parameter updates,the trained model is achieved.

#### 4.2.1 Neural Reasoning Module

The Neural Reasoning Module (NRM) is a versatile deep neural network whose architecture can vary according to the specific task at hand. This adaptability enables the NRM to accommodate diverse tasks and to be implemented with various network architectures. For instance, in the digital image addition task, the NRM may utilize a Convolutional Neural Network (CNN) to process image data, whereas in object detection, it may adopt a network structure incorporating ResNet to enhance detection performance. This capability to dynamically adjust the network architecture based on task requirements allows the NRM to effectively meet the needs of different applications. The objective to be maximized in terms of log-likelihood is formalized as follows:


<!-- Meanless: 5-->

$$
{O}_{\text{task }} = \log {P}_{{\theta }_{1}}\left( {\widehat{y} \mid  D}\right) , \tag{3}
$$

<!-- Media -->

<!-- figureText: The neural reasoning module (NRM) Concept learning Feature vectors $O = {O}_{\text{task }} + {O}_{\text{logic }} + {O}_{\text{cro }}$ Updating ${O}_{\text{task }} = \log {P}_{{\theta }_{1}}\left( {\widehat{y} \mid  D}\right)$ (y) Data $\left( {{\theta }_{1},{\theta }_{2},w}\right)$ ${O}_{\text{logic }} = \log {P}_{{\theta }_{2},w}\left( {\widehat{y},R}\right)$ Raw data ... Input(D) Task network $\left( {\mathbf{\theta }}_{\mathbf{1}}\right)$ Pseudo-Labels The symbolic reasoning module (SRM) $\mathrm{N}$ iterations by examples Grounding Concept networks $\left( {\mathbf{\theta }}_{2}\right)$ Logic rules MLN (R) Probabilistic graphical model(w) -->

<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2025/11/2025_11_04__12_34_43_a141c9.jpg"/>

Fig. 2. Illustration of concept learning. The NRM aims to predict labels for raw data, generating pseudo-labels and feature vectors as outputs. The SRM is a probabilistic graphical model that incorporates both the pseudo-labels from the NRM and the ground atoms from the MLN. The entire model is trained end-to-end, using backpropagation to iteratively refine the pseudo-labels.

<!-- Media -->

where ${\theta }_{1}$ is the learnable parameter of the NRM. At the beginning of the model training, the NRM may produce predictions with substantial errors due to insufficient training. Consequently, in this paper,we refer to these predictions as pseudo-labels $\widehat{y}$ .

#### 4.2.2 Symbolic Reasoning Module

The Symbolic Reasoning Module (SRM) plays a critical role in supporting the NRM by facilitating learning and employing reasoning to generate predictive outcomes and provide evidence for result interpretation. Specifically, the SRM operates as follows: when presented with a training sample $\left( {{x}_{i},{y}_{i}}\right)$ ,it is responsible for deducing the outcome ${y}_{i}$ based on the predicted label ${\widehat{y}}_{i}$ ,the feature vector output by the NRM, and first-order logic rules. If $\widehat{{y}_{i}}$ is incorrect,the SRM adjusts the NRM parameters through backpropagation to correct the prediction. To achieve this, we leverage SRL to construct a probabilistic graphical model within the SRM, as depicted in Fig. 2. The primary objective of the SRM is to utilize SRLs for learning variables and guiding the NRM's reasoning in the correct direction, effectively serving as an error corrector. In this study, the probabilistic graphical model is instantiated using a MLN that encompasses all tasks discussed in the validations.

When using MLNs to model logical rules, various structures can be adopted depending on the task, including single-layer and double-layer configurations. For instance, in the case of Visual Relationship Detection (VRD), we employed a double-layer structure, as detailed in Section 5.4 and illustrated in Fig. 8. In other scenarios, we utilized a single-layer structure, with its joint probability distribution taking the form presented in Eq. (2). However, if the MLN incorporates multiple types of nodes and potential functions, the joint probability distribution will consist of multiple components. In this study, obtaining the nodes of the MLN requires performing grounding of the FOL statements. Grounding all FOLs in the database can lead to an excessively large number of variables, significantly increasing model complexity. Therefore, during training, the model identifies FOLs that are strongly related to the data, such as predicates that share the same labels as the data in a FOL. The optimization goal of the SRM is defined as ${O}_{\text{logic }}$ in Eq. (4),which aims to maximize the joint probability distribution over all variables in terms of log-likelihood,

$$
{O}_{\text{logic }} = \log {P}_{{\theta }_{2},w}\left( {\widehat{y},R}\right)  = \log \left\{  {\frac{1}{Z\left( w\right) }\exp \left\{  {\mathop{\sum }\limits_{{r \in  R}}{w}_{r}\mathop{\sum }\limits_{{{a}_{r} \in  {A}_{r}}}\phi \left( {a}_{r}\right)  + \mathbb{C}}\right\}  }\right\}  ,
$$

(4)

where $\mathbb{C}$ represents a custom term that may include potential functions ${\phi }_{1},{\phi }_{2},\ldots$ ,and should be designed according to task requirements.

#### 4.2.3 Optimization

The NSF-SRL model comprises two neural networks and a probabilistic graphical model, where the neural networks consist of a NRM and a concept network. The NRM is responsible for learning the features of concepts, while the concept network aims to infer the labels of query variables to approximate the posterior distribution. The symbolic reasoning module is responsible for learning a joint probability distribution to facillitate outcome inference.


<!-- Meanless: 6-->

<!-- Media -->

<!-- figureText: ${e}_{1}$ 曲 ${P}_{\text{binary }}\left( {{A}_{i}\left( {{e}_{1},{e}_{2}}\right) }\right)$ ${P}_{\text{unary }}\left( {{A}_{j}\left( {e}_{j}\right) }\right)$ ${e}_{2}$ 曲 ${e}_{j}$ 曲 Tensor layer Concept network -->

<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2025/11/2025_11_04__12_34_43_fe5494.jpg"/>

Fig. 3. Concept network. The inputs are feature vectors of object pairs (e.g., ${e}_{1}$ and ${e}_{2}$ ) or objects (e.g., ${e}_{j}$ ),and outputs are probabilities of affiliation relationship labels (e.g., ${P}_{\text{binary }}\left( {{A}_{i}\left( {{e}_{1},{e}_{2}}\right) }\right)$ ) or object labels (e.g., ${P}_{\text{unary }}\left( {{A}_{j}\left( {e}_{j}\right) }\right) ).k$ represents tensor layer and each layer is a predicate.

<!-- Media -->

The objective function $\log {P}_{{\theta }_{1}}$ of the neural reasoning module is typically differentiable and can be optimized using gradient descent. In this paper, the discrete logical knowledge within the symbolic reasoning module,represented as $\log {P}_{{\theta }_{2},w}$ ,is transformed into a probabilistic graphical form, making symbolic reasoning also differentiable through the introduction of a concept network for posterior inference. The model aims to minimize the objective function to facilitate end-to-end joint training of both modules. Specifically, during the E-step, the posterior distribution of the query variables is inferred, while in the M-step, the weights of the rules are learned. The training phase continues until the model reaches convergence. The parameters of the neural reasoning module, the concept network, and the symbolic reasoning module are denoted as ${\theta }_{1},{\theta }_{2}$ ,and $w$ ,respectively.

To train the symbolic reasoning module, we need to maximize ${O}_{\text{logic }}$ . However,the computation of the partition function $Z\left( w\right)$ in ${P}_{{\theta }_{2},w}\left( {\widehat{y},R}\right)$ makes it intractable to optimize this objective function directly. Consequently, we introduce the variational EM algorithm and optimize the variational evidence lower bound (ELBO):

$$
{ELBO} = {E}_{Q}\left\lbrack  {\log {P}_{{\theta }_{2},w}\left( {\widehat{y},R}\right) }\right\rbrack   - {E}_{Q}\left\lbrack  {\log Q\left( {\widehat{y} \mid  R}\right) }\right\rbrack  , \tag{5}
$$

where $Q\left( {\widehat{y} \mid  R}\right)$ is the variational posterior distribution.

In general, we utilize the variational EM algorithm to optimize the ELBO. Specifically, we minimize the Kullback-Leibler (KL) divergence between the variational posterior distribution $Q\left( {\widehat{y} \mid  R}\right)$ and the true posterior distribution ${P}_{w}\left( {\widehat{y} \mid  R}\right)$ during the E-step. Due to the complex graphical structure among variables, the exact inference becomes computationally intractable. Therefore, we adopt a mean-field distribution to approximate the true posterior, inferring the variables independently as follows:

$$
Q\left( {\widehat{y} \mid  R}\right)  = \mathop{\prod }\limits_{{{A}_{i} \in  A}}Q\left( {A}_{i}\right) . \tag{6}
$$

For computational convenience, traditional variational methods typically require a predefined distribution, such as the Dirichlet distribution, and then utilize traditional search algorithms to solve the problem. In contrast, we employ neural networks (concept networks in this paper) to parameterize the variational calculation in Eq. (6). Consequently, the variational process transforms into a parameter learning process for the neural networks. As illustrated in Fig. 3, the neural network is called the concept network and is used to compute the posterior $Q\left( {A}_{i}\right)$ . Thus, $Q\left( {A}_{i}\right)$ is rewritten as ${Q}_{{\theta }_{2}}\left( {A}_{i}\right)$ .

Based on the above analysis, combined with Eq. 4 and Eq. 6), Eq. 5 is rewritten as:

$$
{ELBO} = \mathop{\sum }\limits_{{r \in  R}}{w}_{r}\mathop{\sum }\limits_{{{a}_{r} \in  {A}_{r}}}{E}_{{Q}_{{\theta }_{2}}}\left\lbrack  {\phi \left( {a}_{r}\right) }\right\rbrack   - \log Z\left( w\right)  \tag{7}
$$

$$
 - {E}_{{Q}_{{\theta }_{2}}}\left\lbrack  {\mathop{\sum }\limits_{{{A}_{i} \in  A}}{Q}_{{\theta }_{2}}\left( {A}_{i}\right) }\right\rbrack   + \mathbb{C}\text{.}
$$

In Fig. 3, to attain predicate labels of the hidden variables, we first feed feature vectors into concept network, such as feature vector of an object pair $\left( {{e}_{1},{e}_{2}}\right)$ or the feature vector of a single object ${e}_{j}$ . Then,the concept network outputs a binary predicate label if provided with feature vectors of an object pair; otherwise, it outputs a unary predicate label. For example, when we input the feature vector of an image of a zebra into the concept network, it can output the predicate "zebra". Furthermore, to enhance the performance of the concept network through supervised information, we introduce a cross-entropy loss for optimization, which serves as a log-likelihood,

$$
{O}_{cro} =  - \mathop{\sum }\limits_{{{A}_{i} \in  A}}{Q}_{{\theta }_{2}}\left( {A}_{i}\right) \log {\widehat{y}}_{i} =  - \log {\Pi }_{{A}_{i} \in  A}{\widehat{y}}_{i}{}^{{Q}_{{\theta }_{2}}\left( {A}_{i}\right) }. \tag{8}
$$

Thus, the overall E-step objective function becomes:

$$
O = \alpha {O}_{\text{task }} + \beta {O}_{\text{logic }} - \gamma {O}_{\text{cro }}, \tag{9}
$$

where $\alpha ,\beta$ and $\gamma$ are the hyperparameter to control the weight. We maximize Eq. 9) to learn model parameters, the details are as follows:

$$
\left\{  {{\theta }_{1}^{ * },{\theta }_{2}^{ * }}\right\}   = \underset{{\theta }_{1},{\theta }_{2}}{\arg \max }O. \tag{10}
$$

In the M-step, the model learns the weights of the first-order logic rules. As we optimize these weights, the partition function $Z\left( w\right)$ in Eq. (4) is no longer constant,while ${Q}_{\theta 2}$ remains fixed. The partition function $Z\left( w\right)$ consists of an exponential number of terms, rendering direct optimization of the ELBO intractable. To solve this issue, we employ pseudo-log-likelihood [36] to approximate the ELBO, which is defined as follows:

$$
{P}_{w}\left( {\widehat{y},R}\right)  \simeq  {E}_{{Q}_{{\theta }_{2}}}\left\lbrack  {\mathop{\sum }\limits_{{{A}_{i} \in  A}}\log {P}_{w}\left( {{A}_{i} \mid  M{B}_{{A}_{i}}}\right) }\right\rbrack  , \tag{11}
$$

where $M{B}_{{A}_{i}}$ represents Markov blanket of the ground atom ${A}_{i}$ . For each rule $r$ that connects ${A}_{i}$ to its Markov blanket,we optimize weights ${w}_{r}$ using gradient descent,and derivative is given by the following:

$$
{ \bigtriangledown  }_{{w}_{r}}{E}_{{Q}_{{\theta }_{2}}}\left\lbrack  {\log {P}_{w}\left( {{A}_{i} \mid  M{B}_{{A}_{i}}}\right) }\right\rbrack   \simeq  {\widehat{y}}_{i} - {P}_{w}\left( {{A}_{i} \mid  M{B}_{{A}_{i}}}\right) , \tag{12}
$$

where ${\widehat{y}}_{i} = 0$ or 1 if ${A}_{i}$ is an observed variable,and ${\widehat{y}}_{i} = {Q}_{{\theta }_{2}}\left( {A}_{i}\right)$ otherwise.

### 4.3 Concept Manipulation

As mentioned in the overview of NSF-SRL, concept manipulation includes transductive and inductive concept manipulation methods. Consequently, we designed two corresponding approaches, as illustrated in Fig. 4 When the test data intersects with the training data, transductive concept manipulation employs the trained task network to predict results and utilizes probability graphical model to derive the FOLs corresponding to these predictions, providing explanations, as shown in Fig. 4 (a). In contrast, when the test data is disjoint from the training data, inductive concept manipulation uses the trained task network to extract data features. By introducing new FOLs to generalize the model for addressing new tasks, fuzzy logic reasoning is then applied to deduce the prediction results, as depicted in Fig. 4 (b).


<!-- Meanless: 7-->

<!-- Media -->

<!-- figureText: (1) Feedforwarding Transductive concept manipulation (4) Inputting Concept networks (trained) (4) Probabilistic reasoning P(C Evidence chains (a) Inductive concept manipulation Concept networks (trained) A Solution candidate rules Fuzzy logic (b) New data Feature vectors Input Prediction labels Task network (trained) (2) Abstracting (3) Matching MLN (trained) New data Prediction labels Input Task network(trained) Feature vectors (2) Grounding (3) Logical reasoning New logic rules (1) Rewriting New task -->

<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2025/11/2025_11_04__12_34_43_b75e3e.jpg"/>

Fig. 4. Illustration of concept manipulation. (a) Transductive concept manipulation. The trained neural reasoning module predicts results, while the symbolic reasoning module provides interpretability. (b) Inductive concept manipulation. The trained neural reasoning module generates feature vectors, which are used by the symbolic reasoning module for reasoning.

<!-- Media -->

In the scenario depicted in Fig. 4 (a), the categories of the training set and the test set overlap. As illustrated in Fig. 1, the training data includes "zebra", which is also present in the test data. The steps of transductive manipulation are as follows: (1) Feedforwarding: input new data and obtain prediction labels and features through the trained task network; (2) Abstracting: derive partial nodes as observed variables in the probabilistic graphical model from the predicted labels; (3) Matching: match these partial nodes with first-order logic rules in the Markov logic network to identify candidate rules; (4) Inputting: feed feature vectors into the concept network, retrieve the scores of the concepts, and apply probabilistic reasoning (Eq. (13)) and fuzzy logic reasoning to obtain the probability score of each rule being true. Rules with high scores are selected as the evidence chain, interpreting the prediction labels. In this paper, we match the prediction results with ground atoms of the logic rules to achieve interpretability. A successful match indicates that the logic rules containing those ground atoms are triggered, and the corresponding clique composed of those nodes is selected. To quantify the likelihood that a candidate rule is true, we calculate the probability using t-norm fuzzy logic [37]. This process allows us to obtain evidence in the form of logic rules supporting the reasoning outcomes. To enhance interpretability, we select the most prominent piece of evidence in terms of a specific rule based on the posterior probability $P\left( {r \mid  \widehat{y}}\right)$ as follows:

$$
P\left( {r \mid  \widehat{y}}\right)  = \mathop{\prod }\limits_{{{A}_{i} \in  {T}_{r}}}p\left( {{A}_{i} \mid  \widehat{y}}\right) , \tag{13}
$$

where ${T}_{r}$ is the candidate rule here. Here, ${A}_{i}$ is the ground atom sets in ${T}_{r}$ .

In the scenario depicted in Fig. 4 (b), the categories of the training data and the test data do not overlap. As shown in Fig. 1 the training data does not include "tiger," whereas the test data does. Specifically, there are three steps for inductive manipulation: (1) Rewriting: rewriting logic rules based on the new task to accommodate specific requirements; (2) Grounding: grounding the logic rules using feature vectors from the task network; (3) Logic reasoning: inputting feature vectors of the concepts mentioned in the rule body of the candidate rules into the concept network to obtain the labels of the concepts. Subsequently, we reason the solution for the new task based on both the rule head and the rule body. This process can be seen as reprogramming for a new problem, utilizing the learned concepts from the previous step to tackle more complex problem scenarios. For instance, the model is trained on single-digit image addition and tested on multi-digit image addition tasks. By adopting this approach, the model can adapt its knowledge and reasoning capabilities to address new problems, thereby demonstrating the generalization capabilities of our method.


<!-- Meanless: 8-->

## 5 EXPERIMENTS

In this section, we conduct experiments on various tasks, including supervised task (transductive concept manipulation), weakly supervised task (transductive concept manipulation and inductive concept manipulation), and zero-shot learning task (inductive concept manipulation), using classic datasets for validation. We first describe the datasets and evaluation metrics. Then, we report the empirical results, including the performance, generalization, and interpretability across different tasks. Finally, we present ablation studies and hyperparameter analysis. The code is available at https://github.com/Dongranyu/NSF-SRL.

### 5.1 Experimental Setup

Tasks and datasets: For the supervised task, we validate our approach on visual relationship detection task. The corresponding datasets are Visual Relationship Detection(VRD) [38] and VG200 [39]. For the weakly supervised task, we conduct experiments on a digit image addition task, utilizing the handwritten digit dataset MNIST. For the zero-shot learning task, we employ image classification for validation, using the AwA2 [40] and CUB [41] datasets.

The VRD contains 5,000 images, with 4,000 images as training data and 1,000 images as testing data. There are 100 object classes and 70 predicates (relations). The VRD includes 37,993 relation annotations with 6,672 unique relations and 24.25 relationships per object category. This dataset contains 1,877 relationships in test set never occur in training set, thus allowing us to evaluate the generalization of our model in zero-shot prediction.

The VG200 contains 150 object categories and 50 predicates. Each image has a scene graph of around 11.5 objects and 6.2 relationships. ${70}\%$ of the images is used for training and the remaining ${30}\%$ is used for testing.

The MNIST is a handwritten digit dataset and includes 0-9 digit images. In this paper, the task is to learn the "single-digit addition" formula given two MNIST images and a "addition" label. To implement the experiment on single-digit image addition, we randomly choose the initial feature of two digits to concat a tuple and take their addition as their labels. MNIST has 60,000 train sets and 10,000 test sets.

The AwA2 consists of 50 animal classes with 37,322 images. Training data contains 40 classes with 30,337 images, and test data has 10 classes with 6,985 images. Additionally, AwA2 provides 85 numeric attribute values for each class.

The CUB comprises 11,788 images spanning 200 bird classes, each associated with 312 attributes. Among these classes, 150 classes are designated as seen during training, while the remaining 50 are unseen and used for evaluation.

The logic rules. In this paper, logic rules encode relationships between a subject and multiple objects for visual relationship detection. Here, we build logic rules in an artificial way for VRD and VG200 datsets. That is, we take relationship annotations together with their subjects and objects to construct a logic rule according to the annotation file in the dataset. For example,we can obtain a logic rule as $\operatorname{laptop}\left( x\right)  \land$ next $\operatorname{to}\left( {x,y}\right)  \Rightarrow$ keyboard $\left( y\right)  \vee$ mouse(y)by the above method. As a result,the numbers of logic rules are 1,642. Unlike VRD datasets, MNIST has no relationship annotation. To adapt to our weakly supervised task, we define corresponding logic rules, e.g., combining two single-digit labels and their addition label as logic rule. For example, $\operatorname{digit}\left( {x,{d}_{1}}\right)  \land  \operatorname{digit}\left( {y,{d}_{2}}\right)  \Rightarrow$ addition $\left( {{d}_{1} + {d}_{2},z}\right)$ ,where the rule head is the addition label, and the rule body is two single-digit labels. In zero-shot image classification, we design logic rules for the AwA2 and CUB datasets, where the rule head is animal categories and the rule body consists of their attributes. For instance,catlike $\left( x\right)  \land  \operatorname{tawny}\left( x\right)  \land  \operatorname{spot}\left( x\right)  \Rightarrow$ leopard(x).

Metrics: For VRD, we adopt evaluation metrics same as [42], which runs Relationship detection (ReD) and Phrase detection (PhD) and shows recall rates (Recall@) for the top 50/100 results, with $k = 1,{70}$ candidate relations per relationship proposal (or $k$ relationship predictions for per object box pair) before taking the top 50/100 results. ReD is inputting an image and outputting labels of triples and boxes of the objects. PhD is inputting an image and output labels and boxes of triples.

For VG200, we use the same evaluation metrics used in [42], including 1) Scene Graph Classification (SGCLS), which is to predict labels of the subject, object, and predicate given ground truth subject and object boxes; 2) Predicate Classification(PCLS), where predict predicate labels are given ground truth subject and object boxes and labels. Recall@ under the top ${20}/{50}/{100}$ predictions are reported.

For MNIST, AwA2 and CUB, we adopt accuracy(Acc) to evaluate the performance of the model. They are defined as Eq. (14).

$$
{Acc} = \frac{{TP} + {TN}}{{TP} + {TN} + {FP} + {FN}}, \tag{14}
$$

where ${TP}$ denotes true positive, ${TN}$ denotes true negative, ${FP}$ indicates false positive,and ${FN}$ is false negative.

For the logic rule, we compute the probability of a logic rule that is true as an evaluation of logic rules. Here, we adopt Łukaseiwicz of t-norm fuzzy logic [37].

### 5.2 Digit Image Addition Task

In the context of neural-symbolic studies, digit image addition serves as a benchmark task, and MNIST dataset is recognized as a benchmark dataset. We evaluate the performance of our NSF-SRL model by comparing it against several neural-symbolic approaches and convolutional neural networks (CNNs). The neural-symbolic approaches considered include DeepPSL [43], DeepProbLog [12], and NeurASP [26]. This paper assesses the model's performance specifically on the single-digit image addition task, where two single-digit images are input into NSF-SRL, and the output is the predicted addition result. Furthermore, to verify the model's generalization capability, we also perform the multi-digit image addition task in Section 5.5.

In the digit image addition task, the neural reasoning module first extracts image features using a CNN. These features are then processed through two fully connected layers to produce a 10-dimensional output vector. The activation functions employed in this neural network structure are ReLU and Softmax. We set the learning rate to 1e-4 and train the model for 15,000 epochs. Additionally, we utilize a batch size of 64 during training and a batch size of 1,000 during testing.


<!-- Meanless: 9-->

<!-- Media -->

<!-- figureText: 1.0 Digit image addition Zero-shot image classification Zero-shot image classification 0.800 0.775 0.778 0.761 0.750 0.699 0.692 0.725 0.72 0.684 0.679 0.700 0.675 0.676 0.66 0.650 0.625 0.600 (b) AwA2 (c) CUB 0.981 0.972 0.973 0.74 0.93 0.9 0.72 0.70 0.8 Acc 0.68 0.678 0.679 0.7 0.66 0.64 0.6 0.62 0.5 (a) MNIST -->

<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2025/11/2025_11_04__12_34_43_912d9a.jpg"/>

Fig. 5. Performance of NSF-SRL and comparison methods on digit image addition and zero-shot image classification tasks: (a) MNIST ; (b) AwA2 ; (c) CUB.

<!-- Media -->

Fig. 5 (a) presents the results of NSF-SRL alongside comparison methods for the digit image addition task. By comparing the performance of NSF-SRL with that of the other methods, we observe that NSF-SRL achieves decent performance. This finding underscores the feasibility of NSF-SRL in circumventing the reliance on strong supervised information typically required in conventional deep learning approaches. By integrating symbolic knowledge, NSF-SRL effectively leverages additional supervisory signals, such as data labels and relationships between data, resulting in improved model performance.

### 5.3 Zero-shot Image Classification Task

In contrast to the digit image addition task, the zero-shot image classification task is inherently more complex. This task involves training a model on images of seen classes, enabling it to recognize images of unseen classes. The objective of the neural reasoning module in this context is to learn a mapping function from the visual space to the semantic space, thereby extracting image features of the objects. The symbolic reasoning module first receives these image features from the neural reasoning module, then models the logic rules using a MLN to learn the joint probability distribution. Finally, it employs a concept network to calculate the posterior probability of the joint distribution, predicting attribute labels and combining these labels according to the established logic rules.

In the zero-shot image classification task, the neural reasoning module is a CNN initialized with a pre-trained GoogleNet. Given an input image, we first use the CNN to extract initial visual features. These features are then fed into an attention network to attain discriminative image features. To enhance data augmentation, images undergo random cropping before being input into the model. For optimization, we employ the Adam optimizer with the following configurations: 15 epochs, a batch size of 64, and a learning rate of $1\mathrm{e} - 4$ . The specific neural architecture is illustrate in the Fig 6

The symbolic reasoning module is implemented as a MLN, which integrates the neural reasoning module with FOL to extract discriminative image features. Additionally, this module enables the trained model to adapt from recognizing seen classes to unseen classes. Specifically, the symbolic reasoning module employs the MLN to learn the joint probability distribution between symbolic discriminative features and classes, predicting the labels of these features by calculating the posterior probability. Consequently, the symbolic reasoning module effectively combines the image features extracted by the neural reasoning module with FOL to perform fuzzy logic reasoning and derive class labels. The introduction of the MLN provides an efficient method for integrating visual features and symbolic discriminative features, thereby enhancing the model's generalization capability to unseen classes. This joint modeling approach captures the associations between image features and attributes, ultimately improving the model's performance in zero-shot image classification.

<!-- Media -->

<!-- figureText: mages (seen class) Class attribute labels scores Image features Attribute features 20.0/12.0、 Image features FC Attention network -->

<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2025/11/2025_11_04__12_34_43_24ed21.jpg"/>

Fig. 6. The neural reasoning module on zero-shot image classification.

<!-- Media -->

Zero-shot image classification is a complex reasoning task that current neural-symbolic methods struggle to address effectively. Consequently, we primarily adopted deep learning-based contrastive approaches. Fig. 5 (b) presents the results for the AwA2 dataset, comparing our method against baseline methods such as SABR [44], MLSE [45], AREN [46], LFGAA [47], DAZLE [48], APN [49], CF-ZSL [50], DUET [51], and MSDN [52]. Fig. 5(c) presents the comparative results on the CUB dataset. The methods included in this comparison are Composer [53], LFGAA [47], DAZLE [48], APN [49], MSDN [52], TransZero [54], and DUET [51].

From Fig. 5, it is evident that our NSF-SRL achieves optimal performance across different datasets, validating the effectiveness of the model. This success can be attributed to the logical rules that model the relationships between attribute features, seen categories, and unseen categories, including co-occurrence relationships. Such rules facilitate the model in capturing these relationships, thereby enhancing classification performance. Additionally, this experiment highlights that incorporating symbolic reasoning with FOL enhances the robustness of the model.


<!-- Meanless: 10-->

### 5.4 Visual Relationship Detection Task

Visual relationship detection, similar to zero-shot image classification, is a complex task that aims to identify objects within an image and the relationships between them. These relationships can be represented as triplets (subject, predicate, object). In this context, the neural reasoning module serves as a deep learning-based specifically designed for visual relationship detection, extracting label concepts of both objects and their relationships from the input image. Conversely, the symbolic reasoning module functions as a two-layer probabilistic graphical model, intended to integrate the learned object and relationship labels while guiding the learning process of the visual reasoning module.

For the visual relationship detection task, our neural reasoning module is based on the architecture described in [42]. It consists of two components: a visual module and a semantic module. The visual module primarily extracts visual features using a CNN, specifically employing layers conv1_1 to conv5_3 of VGG16 to generate a global feature map of the image. Subsequently, the subject, relation, and object features are region-of-interest (ROI) pooled and processed through two fully connected layers to produce three intermediate hidden features. The semantic module, on the other hand, processes word vectors corresponding to the subject, relation, and object labels via a multilayer perceptron (MLP) to generate embeddings. Before training, we initialize each branch using pre-trained weights from the COCO dataset [55] and adopt word2vec [56] for the word vectors in our experiments. Specifically, we train our model for 7 epochs with a learning rate set to 1e-4, and the dimension of the object feature is established at 512. The specific neural architecture is illustrated in Fig. 7

<!-- Media -->

<!-- figureText: CNN RPN object score matrix $\mathbf{{FC}}$ relation score matrix Objects features Input -->

<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2025/11/2025_11_04__12_34_43_563155.jpg"/>

Fig. 7. The neural reasoning module on visual relationship detection task.

<!-- Media -->

As illustrated in the Fig. 8, the symbolic reasoning module is structured as a bi-level probabilistic graphical model, where the high-level layer represents the prediction results (pseudo-labels) generated by the neural reasoning module. In contrast, the low-level layer consists of the ground atoms of MLN. This module consists of two types of nodes (random variables) and cliques (potential functions): the prediction labels from the neural reasoning module in the high-level layer nodes and the ground atoms of the MLN in the low-level layer nodes. Let $\widehat{y} = \left\{  {\widehat{{y}_{1}},\widehat{{y}_{2}},\ldots }\right\}$ denote the set of high-level nodes (pseudo-labels),and let $A = \left\{  {{A}_{1},{A}_{2},\ldots }\right\}$ represent the set of low-level nodes, comprising the ground atoms in the FOLs. A clique $\left\{  {{\widehat{y}}_{i},{A}_{j}}\right\}$ signifies the correlation between these levels,while another clique ${A}_{r}$ represents the ground atoms of a FOL. Consequently,the custom term $\mathbb{C}$ can be defined as $\mathop{\sum }\limits_{{\widehat{{y}_{i}} \in  \widehat{y},{A}_{j} \in  A}}{\phi }_{1}\left( {\widehat{{y}_{i}},{A}_{j}}\right)$ in Eq. 4).

<!-- Media -->

<!-- figureText: person( 2000 person $\left( {\cdot ,t}\right)  \rightarrow$ motorcycle( FOL: person(x) $\land$ on(x,y) $\Rightarrow$ motorcycle(y) High-level person( Low-level MLN on( -->

<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2025/11/2025_11_04__12_34_43_2e9179.jpg"/>

Fig. 8. The symbolic reasoning module on visual relationship detection task.

<!-- Media -->

Existing neural-symbolic methods, such as DeepProbLog, have not been validated on complex tasks like visual relationship detection. Therefore, our comparative methods are restricted to those based solely on deep learning. The experimental results of NSF-SRL and several comparative methods are presented in Table 2 for VRD dataset. As not all comparative methods specified $k$ in their experiment,we report results as "free $k$ " when treating $k$ as a hyperparameter. The results indicate that our NSF-SRL outperforms the comparative methods in most cases. The enhancements offered by the symbolic reasoning module can be attributed to two key factors. First, the symbolic reasoning module is structured as a probabilistic graphical model that effectively captures dependencies between variables, facilitating a more accurate modeling of complex relationships. Second, our logic rules are constructed based on the co-occurrence relationships among predicates, suggesting that when one object is present, another is likely to appear as well. By maximizing the joint probability of the probabilistic graphical model, we effectively enhance the co-occurrence probability during the training phase.

Table 3 presents the results on the VG200 dataset. Notably, the state-of-the-art methods do not specify a clear value for $k$ in this context. Therefore, we report the performance of our NSF-SRL model with $k = 1$ . Our results demonstrate that NSF-SRL outperforms existing methods across two metrics in Recall@20/50/100, highlighting the advantages of leveraging symbolic knowledge through logic rules. Furthermore, while PCLS emphasizes relationship recognition, NSF-SRL achieves a superior score on the PCLS evaluation metric, indicating that the incorporation of logic rules enhances relationship recognition capabilities within the model.

### 5.5 Generalization

Evaluating a model's generalization ability is essential, as it reflects its adaptability and robustness across diverse scenarios. In this study, generalization refers to the model's predictive performance on unseen samples. For example, the model is initially trained on a single-digit image addition task and subsequently tested on a multi-digit image addition task. Zero-shot image classification serves as an experiment that validates the model's generalization capabilities. Consequently, we only focus our experimental validation on visual relationship detection and digit image addition tasks.


<!-- Meanless: 11-->

<!-- Media -->

TABLE 2

Test performance of visual relationship detection. The recall results for the top 50/100 in "ReD" and "PhD" are reported, respectively. The best result is highlighted in bold. "-" denotes the corresponding result is not provided.

<table><tr><td>Methods</td><td colspan="2">ReD</td><td colspan="2">PhD</td><td colspan="4">ReD</td><td colspan="4">PhD</td></tr><tr><td/><td colspan="4">free $k$</td><td colspan="2">$k = 1$</td><td colspan="2">$k = {70}$</td><td colspan="2">$k = 1$</td><td colspan="2">$k = {70}$</td></tr><tr><td>Recall@</td><td>50</td><td>100</td><td>50</td><td>100</td><td>50</td><td>100</td><td>50</td><td>100</td><td>50</td><td>100</td><td>50</td><td>100</td></tr><tr><td>Lk distilation [57]</td><td>22.7</td><td>31.9</td><td>26.5</td><td>29.8</td><td>19.2</td><td>21.3</td><td>22.7</td><td>31.9</td><td>23.1</td><td>24.0</td><td>26.3</td><td>29.4</td></tr><tr><td>Zoom-Net [58]</td><td>21.4</td><td>27.3</td><td>29.1</td><td>37.3</td><td>18.9</td><td>21.4</td><td>21.4</td><td>27.3</td><td>28.8</td><td>28.1</td><td>29.1</td><td>37.3</td></tr><tr><td>CAI+SCA-M [58]</td><td>22.3</td><td>28.5</td><td>29.6</td><td>38.4</td><td>19.5</td><td>22.4</td><td>22.3</td><td>28.5</td><td>25.2</td><td>28.9</td><td>29.6</td><td>38.4</td></tr><tr><td>MF-URLN [59]</td><td>23.9</td><td>26.8</td><td>31.5</td><td>36.1</td><td>23.9</td><td>26.8</td><td>-</td><td>-</td><td>23.9</td><td>26.8</td><td>-</td><td>-</td></tr><tr><td>LS-VRU [42]</td><td>27.0</td><td>32.6</td><td>32.9</td><td>39.6</td><td>23.7</td><td>26.7</td><td>27.0</td><td>32.6</td><td>28.9</td><td>32.9</td><td>32.9</td><td>39.6</td></tr><tr><td>GPS-Net [60]</td><td>27.8</td><td>31.7</td><td>33.8</td><td>39.2</td><td>-</td><td>-</td><td>27.8</td><td>31.7</td><td>-</td><td>-</td><td>33.8</td><td>39.2</td></tr><tr><td>UVTransE [61]</td><td>27.4</td><td>34.6</td><td>31.8</td><td>40.4</td><td>25.7</td><td>29.7</td><td>27.3</td><td>34.1</td><td>30.0</td><td>36.2</td><td>31.5</td><td>39.8</td></tr><tr><td>NMP [62]</td><td>21.5</td><td>27.5</td><td>-</td><td>-</td><td>20.2</td><td>24.0</td><td>21.5</td><td>27.5</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>NSF-SRL</td><td>29.4</td><td>35.3</td><td>36.2</td><td>43.0</td><td>26.2</td><td>29.4</td><td>29.4</td><td>35.3</td><td>32.3</td><td>36.4</td><td>36.2</td><td>43.0</td></tr></table>

<!-- Media -->

#### 5.5.1 Visual Relationship Detection

We evaluated the performance of our NSF-SRL model against the baseline LS-VRU in a zero-shot learning scenario. In this context, the training and testing data comprise disjoint sets of relationships from the VRD dataset, as illustrated in Fig. 9 (a). The results demonstrate that NSF-SRL outperforms LS-VRU across various recall metrics, highlighting LS-VRU's limitations in handling sparse relationships. In contrast, NSF-SRL effectively incorporates symbolic knowledge and language priors, making it less susceptible to the challenges posed by sparse relationships.

#### 5.5.2 Digit image Addition

We validate the generalization capability of NSF-SRL in multi-digit task by comparing it to the baseline. In multi-digit image addition, the input consists of two lists of images, each representing a digit, with each list corresponding to a multi-digit number. The label reflects the sum of these two numbers. In our experiment, a CNN is trained on the multi-digit image addition dataset to test the multi-digit image addition task, while we apply the learned model from the single-digit image addition task to this scenario. As shown in Fig. 9 (b), the results illustrate the enhanced prediction accuracy in the multi-digit image addition task by leveraging concepts acquired during the single-digit task.Our findings indicate a significant improvement compared to other methods, underscoring the flexibility of our model, which can generalize from simpler tasks to more complex ones by adapting its logic rules. Notably, this generalization is facilitated by the shared learnable concepts between the two tasks.

### 5.6 Interpretibility

We employ visual relationship detection and zero-shot image classification tasks to demonstrate the interpretability of our results. In the context of visual relationship detection, Fig. 10 (a) illustrates the reasoning behind the identified relationship "next to" between "laptop" and either the "keyboard" or "mouse". According to Eq. (13), when the subject is a "laptop" and the object is either "keyboard" or "mouse", the relationship "next to" is assigned the highest confidence by the logic rule laptop $\left( x\right)  \land$ next $\operatorname{to}\left( {x,y}\right)  \Rightarrow$ keyboard $\left( y\right)  \vee$ mouse(y).

<!-- Media -->

TABLE 3

Comparative results for top 50/100 in "SGCLS" and "PCLS" respectively on the VG200 dataset. The best result is highlighted in bold.

<table><tr><td>Metrics Recall@</td><td colspan="3">SGCLS</td><td colspan="3">PCLS</td></tr><tr><td>Methods</td><td>20</td><td>50</td><td>100</td><td>20</td><td>50</td><td>100</td></tr><tr><td>VRD [38]</td><td>-</td><td>11.8</td><td>14.1</td><td>-</td><td>27.9</td><td>35.0</td></tr><tr><td>Ass-Embedding [63</td><td>18.2</td><td>21.8</td><td>22.6</td><td>47.9</td><td>54.1</td><td>55.4</td></tr><tr><td>Mess-Passing [39</td><td>31.7</td><td>34.6</td><td>35.4</td><td>52.7</td><td>59.3</td><td>61.3</td></tr><tr><td>Graph-RCNN [64</td><td>-</td><td>29.6</td><td>31.6</td><td>-</td><td>54.2</td><td>59.1</td></tr><tr><td>Per-Invariant [65</td><td>-</td><td>36.5</td><td>38.8</td><td>-</td><td>65.1</td><td>66.9</td></tr><tr><td>Motifnet | 66</td><td>32.9</td><td>35.8</td><td>36.5</td><td>58.5</td><td>65.2</td><td>67.1</td></tr><tr><td>LS-VRU [42]</td><td>36.0</td><td>36.7</td><td>36.7</td><td>66.8</td><td>68.4</td><td>68.4</td></tr><tr><td>GPS-Net 60</td><td>36.1</td><td>39.2</td><td>40.1</td><td>60.7</td><td>66.9</td><td>68.8</td></tr><tr><td>$\mathrm{{NSF}} - \mathrm{{SRL}}\left( {k = 1}\right)$</td><td>37.0</td><td>39.3</td><td>39.3</td><td>67.8</td><td>69.1</td><td>70.0</td></tr></table>

<!-- figureText: Visual relationship detection 1.0 Digit image addition 0.9 0.8 0.6 0.5 OW (b) NSF-SRL top100 NSF-SRL top50 LS-VRU_top50 ReD k=1 $k = {10}$ k=70 (a) -->

<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2025/11/2025_11_04__12_34_43_c60cf2.jpg"/>

Fig. 9. Generalization of NSF-SRL and comparison methods on visual relationship detection and digit image addition tasks. (a) Visual relationship detection. Larger ReD indicates better results. (b) Multi-digit image addition.

<!-- Media -->

In zero-shot image classification, we used heatmaps to visualize the discriminative image features. As shown in Fig. 10 (b), the highlighted regions represent the discriminative features captured by our model. By combining the predicted discriminative feature labels with the logic rules, the model can infer class labels. This transparent reasoning process facilitates easy understanding of the model's decision-making when presented with an image. For instance, when the model identifies an image as black_billed_vuckoo, it justifies its prediction by highlighting features such as a curved_bill, tapered_wing and pointed_tail in the image, and logically deduces that the object possessing these features belongs to the black_billed_vuckoo class, based on the applied rule.


<!-- Meanless: 12-->

<!-- Media -->

<!-- figureText: next to ( "An relationship that subject is laptop and object is keyboard or mouse is next to." mouse (a) Tapered_wing Pointed tail (b) laptop( A next to Black Billed Cuckoo Curved bill Rule: curved_bill $\left( x\right)  \land$ tapered_wing $\left( x\right)  \land$ pointed_tail $\left( x\right)  \Rightarrow$ black_billed_cuckoo(x) -->

<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2025/11/2025_11_04__12_34_43_2d8e8c.jpg"/>

Fig. 10. Interpretability analysis. (a) An example illustrating the interpretability of NSF-SRL. For example, why is the relationship "next to" detected between a "laptop" and a "keyboard" or "mouse" in an image? According to Eq. (13), the model identifies the most confident logic rule: laptop(x) $\land$ next to $\left( {x,y}\right)  \Rightarrow$ keyboard $\left( y\right)  \vee$ mouse(y). This demonstrates that the reasoning results of NSF-SRL align with common sense. (b) Visualization of the learned discriminative image features by our model. Key features, such as the shape of the bill, wing, and tail, are highlighted, providing a visual explanation of the model's reasoning.

<!-- figureText: Digit image addition(ablation) Zero-shot image classification(ablation) 0.60 0.55 NSF-SRL-SRM NSF-SRL-NRM NSF-SRL-O NSF-SRL 0.50 (b) 0.9 0.7 Acc 0.5 NSF-SRL-SRM NSF-SRL-NRM NSF-SRL-OI NSF-SRL 0.3 single-digit multi-digit (a) -->

<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2025/11/2025_11_04__12_34_43_19df7f.jpg"/>

Fig. 11. Ablation results on digit image addition and zero-shot image classification tasks.

<!-- Media -->

### 5.7 Ablation Studies

During the training phase, we conduct an extensive analysis of various factors that may affect downstream task performance. These factors include the hyperparameters $\alpha ,\beta ,\gamma$ . This comprehensive evaluation framework provides deeper insights into the influence of these factors on model performance.

To investigate the impact of model trade-offs on reasoning, we designed three variants to assess the effect of individual components on NSF-SRL. Specifically, we derived these variants from the optimized objective in Eq. (9) by adjusting the values of the trade-off factors. The three variants are as follows: (1) NSF-SRL-SRM $\left( {\alpha  = 1,\beta  = 0,\gamma  = 0}\right)$ : excluding the symbolic reasoning module,(2) NSF-SRL-NRM $\left( {\alpha  = 1/2,\beta  = 1,\gamma  = 1}\right)$ : reducing the visual reasoning module by half, and (3) NSF-SRL-OI $\left( {\alpha  = 1,\beta  = 1,\gamma  = 0}\right)$ : omitting the cross-entropy of observed variables. We conducted experiments on digit image addition and zero-shot image classification tasks to evaluate performance of NSF-SRL and its variants. The results are presented in Fig. 11 (a) and Fig. 11(b), respectively.

In Fig. 11 (a), we observe that the performance of the NSF-SRL-NRM variant is higher compared to its NSF-SRL-SRM counterparts. This indicates that the symbolic reasoning module is crucial in weakly supervised tasks. This is likely due to the limited availability of supervised information in such tasks. Specifically, in weakly supervised tasks, the input images are not individually labeled but only labeled by the addition task. As a result, the NRM module may have a more restricted role in these tasks. Moreover, this finding highlights the importance of incorporating symbolic knowledge.

In Fig. 11 (b), we observe that the correlations among the components of SRM, VRM, and OI have a significantly positive impact on zero-shot image classification. Furthermore, the performance of our model is notably enhanced when SRM is applied, confirming the effectiveness of the symbolic knowledge integrated into the model. We conclude that symbolic knowledge helps the model adapt to new environments, specifically in recognizing unseen classes.


<!-- Meanless: 13-->

### 5.8 Hyperparameter Analysis

To analyze the robustness of our NSF-SRL framework and determine optimal hyperparameters, we conducted extensive experiments to evaluate the effects of epoch settings and loss weights (in Eq. 9)).

1) Effects of Epoch: In Fig. 12, we present the fine-tuning results for models trained with varying numbers epochs, evaluated based on accuracy (Acc) for both digit image addition and zero-shot image classification tasks. The figures clearly show that both NSF-SRL and the baseline models exhibit an upward trend as the number of iterations increases. This trend suggests that the models continue to benefit from longer training, indicating that extended training can further improve performance until convergence. Additionally, the baseline models converge faster than NSF-SRL, which may be due to differences in model architecture, such as CNN or LFGAA having fewer parameters to learn.

2) Effects of Loss Weights: In this section, we analyze the impact of the loss weights $\alpha ,\beta$ and $\gamma$ on their respective loss terms. We experimented with a range of values $\{ 0,{0.5},1,{1.5}$ , 2\} for these weights across digit image addition and zero-shot image classification tasks. The results are illustrated in Fig. 13 When $0 < \alpha  < {0.5}$ ,all evaluation metrics exhibit an upward trend, while for $\alpha  > {0.5}$ ,the performance across all evaluation strategies remains consistent. Additionally, NSF-SRL demonstrates relative insensitivity to $\beta$ and $\gamma$ when set to larger values (e.g.,greater than 0.5). Based on these observations,we set $\alpha ,\beta$ ,and $\gamma$ to 1,1,and 1 , respectively, in our experiments.

<!-- Media -->

<!-- figureText: 1.0 Digit image addition 0.8 Zero-shot image classification 0.7 0.6 0.5 0.4 0.3 0.2 NSF-SRL LFGAA 0.0 epoch 0.8 0.6 0.4 NSF-SRL CNN 2500 5000 7500 10000 12500 15000 epoch -->

<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2025/11/2025_11_04__12_34_43_6c23ba.jpg"/>

Fig. 12. Effects of different epochs for the NSF-SRL on digit image addition and zero-shot image classification.

<!-- figureText: (a) Digit image addition 0.8 1.0 0.8 0.4 0.2 0.70 0.65 0.55 0.45 0.2 0.5 1.5 0.80 0.70 0.65 0.55 0.45 -->

<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2025/11/2025_11_04__12_34_43_c955ff.jpg"/>

Fig. 13. Effects of loss weights that control their corresponding loss terms on digit image addition and zero-shot image classification tasks, i.e., $\alpha ,\beta$ and $\gamma$ .

<!-- Media -->

## 6 CONCLUSION

In this study, we introduce NSF-SRL, a general model in neural-symbolic systems. Our goal is to improve the model's performance and generalization, while also providing interpretability of the results. Additionally, we propose a novel evaluation metric to quantify the interpretability of the deep model. Our experimental results demonstrate that NSF-SRL outperforms state-of-the-art methods across various reasoning tasks, including supervised, weakly supervised, and zero-shot image classification scenarios, in terms of both performance and generalization. Furthermore, we highlight the interpretability of NSF-SRL by providing visualizations that clarify the model's reasoning process.

In practice, the NSF-SRL can find applications in diverse scenarios beyond the experimental tasks discussed in this paper. For instance, in healthcare, the model can be leveraged for medical image analysis and patient diagnosis. By amalgamating symbolic reasoning with deep learning capabilities, it can assist physicians in disease diagnosis and treatment planning while enhancing diagnostic reliability through interpretability. In the financial sector, the NSF-SRL can be instrumental in fraud detection and risk assessment by effectively managing complex data patterns with its hybrid approach.

In our NSF-SRL framework, the manual definition of logic rules may restrict the breadth of acquired rule knowledge and involves labor costs. On top of this foundational work, a potential enhancement would involve enabling the model to autonomously learn rules from data, leading to a more efficient and adaptive system.

## ACKNOWLEDGMENTS

This work was supported by the National Key R&D Program of China under Grant Nos. 2021ZD0112500; the National Natural Science Foundation of China under Grant Nos. U22A2098, 62172185, 62202200, and 62206105. REFERENCES

[1] L. G. Valiant, "Three problems in computer science," in JACM, vol. 50, no. 1, 2003, pp. 96-99.

[2] V. Belle, "Symbolic logic meets machine learning: A brief survey in infinite domains," in SUM, 2020, pp. 3-16.

[3] P. Hitzler and M. K. Sarker, "Neuro-symbolic artificial intelligence: The state of the art," in Neuro-Symbolic Artificial Intelligence, 2021.

[4] E. Curry, D. Salwala, P. Dhingra, F. A. Pontes, and P. Yadav, "Multimodal event processing: A neural-symbolic paradigm for the internet of multimedia things," IOTJ, vol. 9, no. 15, pp. 13705-13 724, 2022.

[5] D. Yu, B. Yang, D. Liu, H. Wang, and S. Pan, "A survey on neural-symbolic systems," NN, 2022.

[6] M. Qu and J. Tang, "Probabilistic logic neural networks for reasoning," NeurIPS, vol. 32, 2019.

[7] Y. Zhang, X. Chen, Y. Yang, A. Ramamurthy, B. Li, Y. Qi, and L. Song, "Efficient probabilistic logic reasoning with graph neural networks," ICLR, 2020.

[8] J. Mao, C. Gan, P. Kohli, J. B. Tenenbaum, and J. Wu, "The neuro-symbolic concept learner: Interpreting scenes, words, and sentences from natural supervision," in arXiv preprint arXiv:1904.12584, 2019.

[9] J. Xu, Z. Zhang, T. Friedman, Y. Liang, and G. Broeck, "A semantic loss function for deep learning with symbolic knowledge," in ICML, 2018, pp. 5502-5511.

[10] Y. Xie, Z. Xu, M. S. Kankanhalli, K. S. Meel, and H. Soh, "Embedding symbolic knowledge into deep networks," NeurIPS, 2019.

[11] R. Luo, N. Zhang, B. Han, and L. Yang, "Context-aware zero-shot recognition," in AAAI, vol. 34, no. 07, 2020, pp. 11709-11716.

[12] R. Manhaeve, S. Dumančić, A. Kimmig, T. Demeester, and L. De Raedt, "Neural probabilistic logic programming in deepproblog," AI, vol. 298, p. 103504, 2021.

[13] Z.-H. Zhou, "Abductive learning: towards bridging machine learning and logical reasoning," SCIS, vol. 62, no. 7, pp. 1-3, 2019.

[14] R. Manhaeve, S. Dumancic, A. Kimmig, T. Demeester, and L. De Raedt, "Deepproblog: Neural probabilistic logic programming," NeurIPS, vol. 31, 2018.


<!-- Meanless: 14-->

[15] L. Getoor and B. Taskar, Introduction to statistical relational learning, 2007.

[16] D. Yu, B. Yang, Q. Wei, A. Li, and S. Pan, "A probabilistic graphical model based on neural-symbolic reasoning for visual relationship detection," in CVPR, 2022, pp. 10609-10618.

[17] R. Abboud, I. Ceylan, and T. Lukasiewicz, "Learning to reason: Leveraging neural networks for approximate dnf counting," in AAAI, vol. 34, no. 04, 2020, pp. 3097-3104.

[18] G. Marra and O. Kuželka, "Neural markov logic networks," in Uncertainty in Artificial Intelligence, 2021, pp. 908-917.

[19] Z. Hu, X. Ma, Z. Liu, E. Hovy, and E. Xing, "Harnessing deep neural networks with logic rules," ${ACL},{2016}$ .

[20] Y. Sun, D. Tang, N. Duan, Y. Gong, X. Feng, B. Qin, and D. Jiang, "Neural semantic parsing in low-resource settings with back-translation and meta-learning," in AAAI, vol. 34, no. 05, 2020, pp. 8960-8967.

[21] A. Oltramari, J. Francis, F. Ilievski, K. Ma, and R. Mirzaee, "Generalizable neuro-symbolic systems for commonsense question answering," in Neuro-Symbolic Artificial Intelligence: The State of the Art, 2021, pp. 294-310.

[22] S. Badreddine, A. d. Garcez, L. Serafini, and M. Spranger, "Logic tensor networks," vol. 303, p. 103649, 2022.

[23] J. Tian, Y. Li, W. Chen, L. Xiao, H. He, and Y. Jin, "Weakly supervised neural symbolic learning for cognitive tasks," AAAI, 2022.

[24] X. Duan, X. Wang, P. Zhao, G. Shen, and W. Zhu, "Deeplogic: Joint learning of neural perception and logical reasoning," TPAMI, 2022.

[25] C. Pryor, C. Dickens, E. Augustine, A. Albalak, W. Y. Wang, and L. Getoor, "Neupsl: neural probabilistic soft logic," in IJCAI, 2023, pp. 4145-4153.

[26] Z. Yang, A. Ishay, and J. Lee, "Neurasp: embracing neural networks into answer set programming," in IJCAI, 2021, pp. 1755-1762.

[27] S. D. Tran and L. S. Davis, "Event modeling and recognition using markov logic networks," in ECCV,2008,pp. 610-623.

[28] H. Poon and P. Domingos, "Unsupervised semantic parsing," in EMNLP, 2009, pp. 1-10.

[29] W. Zhang, X. Li, H. He, and X. Wang, "Identifying network public opinion leaders based on markov logic networks," The scientific world journal, vol. 2014, 2014.

[30] P. Singla and P. Domingos, "Discriminative training of markov logic networks," in AAAI, 2005, pp. 868-873.

[31] L. Mihalkova and R. J. Mooney, "Bottom-up learning of markov logic network structure," in ${ML},{2007}$ ,pp. 625-632.

[32] P. Singla and P. Domingos, "Memory-efficient inference in relational domains," in AAAI, 2006, pp. 488-493.

[33] T. Khot, S. Natarajan, K. Kersting, and J. Shavlik, "Learning markov logic networks via functional gradient boosting," in ICDM, 2011, pp. 320-329.

[34] S. H. Bach, M. Broecheler, B. Huang, and L. Getoor, "Hinge-loss markov random fields and probabilistic soft logic," JLMR, 2017.

[35] H. B. Enderton, A mathematical introduction to logic, 2001.

[36] M. Richardson and P. Domingos, "Markov logic networks," ML, vol. 62, no. 1, pp. 107-136, 2006.

[37] V. Novák, I. Perfllieva, and J. Mockor, Mathematical principles of fuzzy logic, 2012, vol. 517.

[38] C. Lu, R. Krishna, M. Bernstein, and L. Fei-Fei, "Visual relationship detection with language priors," in ECCV, 2016, pp. 852-869.

[39] D. Xu, Y. Zhu, C. B. Choy, and L. Fei-Fei, "Scene graph generation by iterative message passing," in CVPR, 2017, pp. 5410-5419.

[40] Y. Xian, C. H. Lampert, B. Schiele, and Z. Akata, "Zero-shot learning-a comprehensive evaluation of the good, the bad and the ugly," in TPAMI, vol. 41, no. 9, 2019, pp. 2251-2265.

[41] P. Welinder, S. Branson, T. Mita, C. Wah, F. Schroff, S. Belongie, and P. Perona, "Caltech-ucsd birds 200," 2010.

[42] J. Zhang, Y. Kalantidis, M. Rohrbach, M. Paluri, A. Elgammal, and M. Elhoseiny, "Large-scale visual relationship understanding," in AAAI, vol. 33, no. 01, 2019, pp. 9185-9194.

[43] S. Dasaratha, S. A. Puranam, K. S. Phogat, S. R. Tiyyagura, and N. P. Duffy, "Deeppsl: end-to-end perception and reasoning," in IJCAI, 2023, pp. 3606-3614.

[44] A. Paul, N. C. Krishnan, and P. Munjal, "Semantically aligned bias reducing zero shot learning," in CVPR, 2019, pp. 7056-7065.

[45] Z. Ding and H. Liu, "Marginalized latent semantic encoder for zero-shot learning," in CVPR, 2019, pp. 6191-6199.

[46] G.-S. Xie, L. Liu, X. Jin, F. Zhu, Z. Zhang, J. Qin, Y. Yao, and L. Shao, "Attentive region embedding network for zero-shot learning," in CVPR, 2019, pp. 9384-9393.

[47] Y. Liu, J. Guo, D. Cai, and X. He, "Attribute attention for semantic disambiguation in zero-shot learning," in ECCV, 2019, pp. 6698-6707.

[48] D. Huynh and E. Elhamifar, "Fine-grained generalized zero-shot learning via dense attribute-based attention," in CVPR, 2020, pp. 4483-4493.

[49] W. Xu, Y. Xian, J. Wang, B. Schiele, and Z. Akata, "Attribute prototype network for zero-shot learning," NeurIPS, vol. 33, pp. 21969-21980, 2020.

[50] B. Yang, Y. Zhang, Y. Peng, c. Zhang, and J. Hang, "Collaborative filtering based zero-shot learning," Journal of Software, vol. 32, no. 9, pp. 2801-2815, 2021.

[51] Z. Chen, Y. Huang, J. Chen, Y. Geng, W. Zhang, Y. Fang, J. Z. Pan, and H. Chen, "Duet: Cross-modal semantic grounding for contrastive zero-shot learning," in AAAI, vol. 37, no. 1, 2023, pp. 405-413.

[52] S. Chen, Z. Hong, G.-S. Xie, W. Yang, Q. Peng, K. Wang, J. Zhao, and X. You, "Msdn: Mutually semantic distillation network for zero-shot learning," in CVPR, 2022, pp. 7612-7621.

[53] D. Huynh and E. Elhamifar, "Compositional zero-shot learning via fine-grained dense feature composition," NeurIPS, vol. 33, pp. 19849-19860, 2020.

[54] S. Chen, Z. Hong, Y. Liu, G.-S. Xie, B. Sun, H. Li, Q. Peng, K. Lu, and X. You, "Transzero: Attribute-guided transformer for zero-shot learning," in ${AAAI}$ ,vol. 36,no. 1,2022,pp. 330-338.

[55] T.-Y. Lin, M. Maire, S. Belongie, J. Hays, P. Perona, D. Ramanan, P. Dollár, and C. L. Zitnick, "Microsoft coco: Common objects in context," in ECCV, 2014, pp. 740-755.

[56] T. Mikolov, I. Sutskever, K. Chen, G. S. Corrado, and J. Dean, "Distributed representations of words and phrases and their compositionality," in NeurIPS, 2013, pp. 3111-3119.

[57] R. Yu, A. Li, V. I. Morariu, and L. S. Davis, "Visual relationship detection with internal and external linguistic knowledge distillation," in ECCV, 2017, pp. 1974-1982.

[58] G. Yin, L. Sheng, B. Liu, N. Yu, X. Wang, J. Shao, and C. C. Loy, "Zoom-net: Mining deep feature interactions for visual relationship recognition," in ${ECCV},{2018}$ ,pp. 322-338.

[59] Y. Zhan, J. Yu, T. Yu, and D. Tao, "On exploring undetermined relationships for visual relationship detection," in CVPR, 2019, pp. 5128-5137.

[60] X. Lin, C. Ding, J. Zeng, and D. Tao, "Gps-net: Graph property sensing network for scene graph generation," in CVPR, 2020, pp. 3746-3753.

[61] Z.-S. Hung, A. Mallya, and S. Lazebnik, "Contextual translation embedding for visual relationship detection and scene graph generation," TPAMI, vol. 43, no. 11, pp. 3820-3832, 2020.

[62] Y. Hu, S. Chen, X. Chen, Y. Zhang, and X. Gu, "Neural message passing for visual relationship detection," arXiv preprint arXiv:2208.04165, 2022.

[63] A. Newell and J. Deng, "Pixels to graphs by associative embedding," NeurIPS, 2017.

[64] J. Yang, J. Lu, S. Lee, D. Batra, and D. Parikh, "Graph r-cnn for scene graph generation," in ${ECCV},{2018}$ ,pp. 670-685.

[65] R. Herzig, M. Raboh, G. Chechik, J. Berant, and A. Globerson, "Mapping images to scene graphs with permutation-invariant structured prediction," NeurIPS, vol. 31, pp. 7211-7221, 2018.

[66] R. Zellers, M. Yatskar, S. Thomson, and Y. Choi, "Neural motifs: Scene graph parsing with global context," in CVPR, 2018, pp. 5831-5840.