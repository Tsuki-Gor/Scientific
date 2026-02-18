# Soft Prompt Generation for Domain Generalization
# 面向领域泛化的软提示生成


Shuanghao Bai ${}^{1 \star  }$ Yuedi Zhang ${}^{1 \star  }$ Wanqi Zhou ${}^{1}$ Zhirong Luan ${}^{2}$ Badong Chen ${}^{1 \dagger  }$
Shuanghao Bai ${}^{1 \star  }$ Yuedi Zhang ${}^{1 \star  }$ Wanqi Zhou ${}^{1}$ Zhirong Luan ${}^{2}$ Badong Chen ${}^{1 \dagger  }$


${}^{1}$ Institute of Artificial Intelligence and Robotics,Xi’an Jiaotong University,China ${}^{2}$ School of Electrical Engineering,Xi’an University of Technology,Xi’an,China \{baishuanghao, zyd993, zwq785915792\}@stu.xjtu.edu.cn, luanzhirong@xaut.edu.cn, chenbd@mail.xjtu.edu.cn
${}^{1}$ 人工智能与机器人研究所, 西安交通大学, 中国 ${}^{2}$ 电气工程学院, 西安理工大学, 西安, 中国 \{baishuanghao, zyd993, zwq785915792\}@stu.xjtu.edu.cn, luanzhirong@xaut.edu.cn, chenbd@mail.xjtu.edu.cn


Abstract. Large pre-trained vision language models (VLMs) have shown impressive zero-shot ability on downstream tasks with manually designed prompt. To further adapt VLMs to downstream tasks, soft prompt is proposed to replace manually designed prompt, which undergoes fine-tuning based on specific domain data. Prior prompt learning methods primarily learn a fixed prompt or residuled prompt from training samples. However, the learned prompts lack diversity and ignore information about unseen domains. In this paper, we reframe the prompt learning framework from a generative perspective and propose a simple yet efficient method for the Domain Generalization (DG) task, namely Soft Prompt Generation (SPG). Specifically, SPG consists of a two-stage training phase and an inference phase. During the training phase, we introduce soft prompt label for each domain, aiming to incorporate the generative model domain knowledge. During the inference phase, the generator of the generative model is employed to obtain instance-specific soft prompts for the unseen target domain. Extensive experiments on five domain generalization benchmarks of three DG tasks demonstrate that SPG achieves state-of-the-art performance. The code is available at https://github.com/renytek13/Soft-Prompt-Generation-with-CGAN.
摘要。大型预训练视觉-语言模型（VLMs）在通过手工设计提示的下游任务上展示了令人印象深刻的零样本能力。为进一步让 VLMs 适应下游任务，提出软提示来替代手工设计的提示，并基于特定领域数据进行微调。以往的提示学习方法主要从训练样本中学习一个固定提示或残差式提示。然而，所学的提示缺乏多样性，且忽略了未见领域的信息。本文将提示学习框架从生成视角重新构建，并提出一个简单而高效的领域泛化（DG）任务方法，即软提示生成（SPG）。具体来说，SPG 由两阶段训练和一个推理阶段组成。在训练阶段，我们为每个领域引入软提示标签，旨在将生成模型的领域知识融入其中。在推理阶段，使用生成模型的生成器来获取未见目标领域的实例特定软提示。对五个领域泛化基准数据集的五个 DG 任务的广泛实验表明，SPG 取得了最先进的性能。代码可在 https://github.com/renytek13/Soft-Prompt-Generation-with-CGAN 获取。


Keywords: Domain Generalization - Visual Language Models - Prompt Learning - Generative Models
关键词：领域泛化 - 视觉语言模型 - 提示学习 - 生成模型


## 1 Introduction
## 1 介绍


Large vision language models, such as CLIP 32 and ALIGN 16, have attracted significant attention owing to their effective adaptation to downstream tasks with manually designed prompts. However, manually designed prompts are not always optimal for domain-specific tasks. Instead of manually designed prompts, the soft prompt19,31,40can be optimized in a data-driven manner through back-propagation. The soft prompt, serving as a learning vector, is refined through fine-tuning with domain-specific data to better adapt to downstream tasks, including base-to-novel generalization [18, 40] and domain adaptation [2, 13, 35]. Despite the progress made in prompt learning, generalization performance tends to decline significantly when facing distribution shifts.
大型视觉语言模型，如 CLIP 32 和 ALIGN 16，由于通过手工设计的提示对下游任务的有效适应性而受到广泛关注。然而，手工设计的提示并不总是最适合领域特定任务。与其手工设计提示，不如通过反向传播以数据驱动的方式优化软提示19,31,40。软提示作为一个学习向量，通过基于领域数据的微调进行细化，以更好地适应下游任务，包括基础到新颖任务的泛化 [18, 40] 和领域自适应 [2, 13, 35]。尽管在提示学习方面取得了进展，但在面对分布偏移时，泛化性能往往显著下降。


---



* Equal contribution.
* 贡献等同。


† Corresponding author.
† 通信作者。


---



<img src="https://cdn.noedgeai.com/bo_d6aql6v7aajc739ardvg_1.jpg?x=413&y=333&w=982&h=821&r=0"/>



Fig. 1: The difference between previous work and our work. We reframe the prompt learning framework from a generative perspective. We exclusively rely on a generative model to directly produce soft prompts, ensuring their diversity. Essentially, we transfer the prompt's adaptability to the generative model by incorporating domain knowledge.
图1：以往工作与我们工作的区别。我们从生成角度重新构建提示学习框架。我们仅依赖生成模型直接产生软提示，以确保其多样性。实质上，我们通过结合领域知识，将提示的适应性转移到生成模型。


Many efforts have been made to address the distribution shift problem by domain generalization (DG) [25, 38]. The objective of DG is to train a model using data from one or multiple related but distinct source domains in a manner that enables the model to generalize effectively to any out-of-distribution (OOD) unseen target domain. Due to the extensive learning in various domains, vision language models like CLIP exhibit a generalization performance that can even rival state-of-the-art traditional supervised learning algorithms. Taking a step, many works adopt prompt learning techniques [6, 7, 29, 36, 39] to enhance the generalization performance of CLIP. Niu et al. 29 introduce domain bank to incorporate textual domain knowledge into soft prompts. Both Zhou et al. 39 and Zhang et al. 36 establish a lightweight neural network (i.e., multilayer perceptron) to generate soft prompts conditioned on images with a residual or concatenation manner. However, these methods either lack diversity and visibility into the information from the target domain or rely on simple models to approximate the relationship between images and soft prompts. Consequently, the learned prompts may still fall short of being transferable.
已经有许多努力通过领域泛化（DG）来解决分布偏移问题 [25, 38]。DG 的目标是使用来自一个或多个相关但不同的源领域的数据来训练模型，使模型能够对任意未分布外（OOD）的目标领域进行有效泛化。由于在各领域的广泛学习，像 CLIP 这样的视觉语言模型的泛化性能甚至可以与最先进的传统监督学习算法相媲美。进一步地，许多工作采用提示学习技术 [6, 7, 29, 36, 39] 以提升 CLIP 的泛化性能。Niu 等人 29 引入域知识库将文本域知识纳入软提示。Zhou 等人 39 与 Zhang 等人 36 构建了一个轻量级神经网络（即多层感知机），以在图像条件下生成软提示，采用残差或拼接的方式。然而，这些方法要么缺乏对目标领域信息的多样性和可视性，要么依赖简单模型来近似图像与软提示之间的关系。因此，所学的提示仍可能难以实现可迁移性。


As shown in Figure 1, unlike the previous prompt learning methods, such as unconditional prompt learning 1840 and conditional prompt learning 8 36 39 methods, which typically rely on fixed prompts obtained from the training set for downstream tasks, or employ a straightforward multi-layer perceptron to learn a residual vector to enhance the richness of the fixed prompts, our approach takes a different direction. We reframe the prompt learning framework by adopting a generative perspective, marking the first integration of generative models into VLMs for prompt learning. We propose a new prompt learning paradigm Soft Prompt Generation (SPG), which offers a straightforward yet effective solution for Domain Generalization. SPG is designed to exclusively harness a generative model for prompt generation, leveraging the model's inherent capability to encode domain and content knowledge directly into the generated prompts.
如图1所示，与先前的提示学习方法不同，如无条件提示学习1840和条件提示学习8 36 39方法，通常依赖于从训练集中获得的固定提示用于下游任务，或采用简单的多层感知机来学习残差向量以增强固定提示的丰富性，我们的方法走了一条不同的道路。我们通过采纳生成性视角重新框定提示学习框架，标志着生成模型首次被引入VLMs用于提示学习。我们提出一种新的提示学习范式Soft Prompt Generation（SPG），为领域泛化提供一个简单而有效的解决方案。SPG旨在专门利用生成模型进行提示生成，直接将模型固有的领域与内容知识编码到生成的提示中。


Our Proposed SPG: SPG method consists of a two-stage training phase and an inference phase. Specifically, in the initial training phase, we introduce domain prompt labels, representing optimal prompts for each domain. Images and corresponding domain prompt labels are then input into a simple yet effective generative model, Conditional Generative Adversarial Net (CGAN) 27, to incorporate domain and content knowledge into the prompt generator model in the second training phase. As a result, domain knowledge is stored not in soft prompts but in the generative model. During inference, the generator of the generative model is employed to obtain domain-specific soft prompts for the target domain data, enabling enhanced diversity and transferability for prompts.
我们提出的SPG：SPG方法包括两阶段训练和推理阶段。具体地，在初始训练阶段，我们引入领域提示标签，表示每个领域的最优提示。将图像及其相应的领域提示标签输入到一个简单但有效的生成模型Conditional Generative Adversarial Net（CGAN）27中，在第二阶段训练中将领域与内容知识并入提示生成模型。结果，领域知识并非存储在软提示中，而是存储在生成模型中。在推理时，使用生成模型的生成器获得目标领域数据的领域特定软提示，从而提升提示的多样性与可迁移性。


Our main contributions are as follows:
我们的主要贡献如下：


- To the best of our knowledge, we are the first to introduce the generative model into prompt learning in VLMs. Then, we propose a new paradigm of prompt tuning, namely Soft Prompt Generation (SPG).
- 据我们所知，我们首次在VLMs的提示学习中引入生成模型。随后，我们提出一种新的提示微调范式，即Soft Prompt Generation（SPG）。


- We design a two-stage training phase to align the generative model with domain prompt labels. It incorporates domain knowledge into the generated prompts, enhancing the transferability across unseen domains.
- 我们设计了两阶段训练以使生成模型与领域提示标签对齐。它将领域知识融入生成的提示，提升对未见领域的可迁移性。


- Extensive experiments on five datasets for three DG tasks demonstrate that the proposed SPG achieves state-of-the-art performance.
- 对五个数据集的三项DG任务的广泛实验表明，所提SPG达到了最先进的性能。


## 2 Related Work
## 2 相关工作


### 2.1 Domain Generalization
### 2.1 域泛化


Domain Generalization (DG) aims to train a model using data from one or multiple related yet different source domains, enabling the model to generalize effectively to any out-of-distribution (OOD) target domain. Existing works mainly focus on learning domain-invariant features across one or multiple source domains. One line of research focuses on domain alignment methods, primarily aiming to minimize moments [28], KL Divergence [23], and Maximum Mean Discrepancy 22 to learn domain-invariant features. Another line of work involves leveraging data augmentation to enrich images or their features, thereby contributing to the learning of invariant predictors [24, 34]. Additional strategies encompass adversarial learning [12], ensemble learning [26], meta-learning [20], gradient operations [4], and more. Recently, large vision language models (VLMs) such as CLIP 32 have been applied to various downstream tasks, demonstrating their potential for remarkable generalization performance. One of the most efficient fine-tuning paradigms for VLMs is prompt learning. Building upon the success of prompt learning, our work delves deeper into exploring this highly efficient paradigm. We propose a novel prompt learning framework that departs from the previous prompt learning methods of utilizing fixed prompts [40] or residualed prompts [39] directly during inference. Instead, our framework leverages a generative model to dynamically obtain soft prompts for the inference process, introducing a new paradigm of prompt tuning.
域泛化（DG）旨在使用来自一个或多个相关但不同的源领域的数据来训练模型，使其能够对任何分布外（OOD）目标领域进行有效泛化。现有工作主要关注在一个或多个源领域之间学习领域不变的特征。一个方向聚焦于领域对齐方法，主要目标是最小化矩 [28]、KL散度 [23] 和最大均值差异 22，以学习领域不变特征。另一方向利用数据增强来丰富图像或其特征，从而有助于学习不变预测器 [24, 34]。其他策略包括对抗学习 [12]、集成学习 [26]、元学习 [20]、梯度操作 [4] 等。最近，大型视觉语言模型（VLMs）如CLIP 32已应用于各种下游任务，展示了其显著的泛化潜力。在VLMs中最有效的微调范式之一是提示学习。在提示学习取得成功的基础上，我们的工作更深入地探索这一高效范式。我们提出一个新颖的提示学习框架，区别于以往在推理阶段直接使用固定提示 [40] 或残差提示 [39] 的方法。相反，我们的框架利用生成模型在推理过程中动态获取软提示，开辟了提示微调的新范式。


### 2.2 Prompt Learning for Vision Language Models
### 2.2 面向视觉语言模型的提示学习


Prompt learning for VLMs, also referred to as prompt tuning, aims to tune the model on domain-specific downstream tasks by only training a small set of parameters which might be a set of newly added parameters with the input. CoOp [40] firstly introduces soft prompt in VLMs and demonstrates a suitable prompt that can improve performance for the image recognition task. CoCoOp [39] solves the overfitting problem of CoOp with conditioned prompt learning, i.e., residualed prompt, ensuring the diversity of prompt. DAPL 13 and PDA 2 introduce domain labels and domain alignment into prompt learning for domain adaptation, respectively. For DG problem, CAE 29 aims to obtain domain-unified text representation with domain bank. DPL 36 generates prompts through MLP to add domain-specific features to the prompt template. Different from previous prompt learning methods in VLMs, we reframe the prompt learning framework from a generative perspective, i.e., exclusively relying on a generative model to dynamically produce soft prompts. For the DG setting, we further introduce a two-stage training paradigm and domain prompt labels to effectively embed domain knowledge within the generative model.
VLMs的提示学习，也称为提示微调，目标是在领域特定的下游任务上对模型进行微调，方法仅训练少量参数，可能是一组与输入一起新添加的参数。CoOp [40] 首次在VLMs中引入软提示，证明了一个可提升图像识别任务性能的合适提示。CoCoOp [39] 通过带条件的提示学习，即残差提示，解决了CoOp的过拟合问题，确保提示的多样性。DAPL 13 和 PDA 2 将领域标签和领域对齐引入提示学习用于领域自适应。对于DG问题，CAE 29旨在通过领域库获得域统一的文本表示。DPL 36 通过MLP生成提示，在提示模板中加入领域相关特征。与VLMs中以往的提示学习方法不同，我们从生成性角度重新构建提示学习框架，即完全依赖生成模型动态生成软提示。对于DG设置，我们进一步引入两阶段训练范式和领域提示标签，以在生成模型中有效嵌入领域知识。


## 3 Preliminaries
## 3 预备知识


### 3.1 Problem Setup of Domain Generalization
### 3.1 域泛化问题设置


In DG setting,there are $M$ source domains ${D}_{s} = \left\{  {{D}_{s}^{1},{D}_{s}^{2},\ldots ,{D}_{s}^{M}}\right\}$ and $N$ target domains ${D}_{t} = \left\{  {{D}_{t}^{1},{D}_{t}^{2},\ldots ,{D}_{t}^{N}}\right\}$ ,all of which follow different distributions. For each source domain, ${D}_{s}^{i} = {\left\{  \left( {x}_{j}^{i},{y}_{j}^{i}\right) \right\}  }_{j = 1}^{{n}_{i}}$ where ${n}_{i}$ denotes the size of samples in ${D}_{s}^{i}$ and $\left( {{x}_{j}^{i},{y}_{j}^{i}}\right)$ denotes the pair of input and target label for the $j$ th sample in the $i$ th domain. Then we denote the input space as $X$ and denote the label set as $Y$ .
在 DG 设置中，存在 $M$ 个源域 ${D}_{s} = \left\{  {{D}_{s}^{1},{D}_{s}^{2},\ldots ,{D}_{s}^{M}}\right\}$ 和 $N$ 个目标域 ${D}_{t} = \left\{  {{D}_{t}^{1},{D}_{t}^{2},\ldots ,{D}_{t}^{N}}\right\}$，它们都遵循不同的分布。对于每个源域，${D}_{s}^{i} = {\left\{  \left( {x}_{j}^{i},{y}_{j}^{i}\right) \right\}  }_{j = 1}^{{n}_{i}}$，其中 ${n}_{i}$ 表示 ${D}_{s}^{i}$ 中样本的大小，$\left( {{x}_{j}^{i},{y}_{j}^{i}}\right)$ 表示第 $j$ 个样本在第 $i$ 个域中的输入与目标标签对。于是我们把输入空间记为 $X$，把标签集合记为 $Y$。


Assuming that all domains share the same label space, previous methods mainly focus on learning a domain-invariant model to map $F : X \rightarrow  Y$ from images to labels, with the aspiration that this mapping can generalize to unseen target domains. With the advent of VLMs, prompt learning methods are proposed to incorporate soft prompts $V$ into the input,and the mapping is rephrased as $F : \{ X,V\}  \rightarrow  Y$ . In contrast to previous methods that involved fine-tuning the model,prompt learning methods center on the fine-tuning of the prompt $V$ . Unlike these methods above, we introduce a new prompt learning paradigm that solely relies on a generative model $G$ to produce soft prompts directly. Thus,the mapping is rephrased as $F : \{ X,G\left( X\right) \}  \rightarrow  Y$ . Our goal is to learn a generative model that can capture both domain-invariant and domain-specific features to produce generalized prompts for unseen target domains dynamically.
假设所有域具有相同的标签空间，先前的方法主要关注学习一个域不变模型，将 $F : X \rightarrow  Y$ 从图像映射到标签，期望该映射能够泛化到未见的目标域。随着 VLM 的出现，提出了提示学习方法，将软提示 $V$ 融入输入中，映射被改写为 $F : \{ X,V\}  \rightarrow  Y$。与以往需要对模型进行微调的方法相比，提示学习方法聚焦于对提示 $V$ 的微调。与上述方法不同，我们引入一种全新的提示学习范式，完全依赖生成模型 $G$ 直接产生软提示。因此，映射被改写为 $F : \{ X,G\left( X\right) \}  \rightarrow  Y$。我们的目标是学习一个生成模型，能够同时捕捉域不变和域特定特征，动态为未见的目标域生成通用提示。


### 3.2 Prompt Learning Methods in Generalization
### 3.2 泛化中的提示学习方法


Contrastive Language-Image Pre-Training (CLIP) 32 model is pre-trained on 400 million image-text pairs collected from the internet with contrastive learning. It consists of an image encoder $f$ and a text encoder $g$ ,which encodes images and corresponding natural language descriptions, respectively.
对比语言-图像预训练（CLIP）32 模型使用对比学习，在互联网上收集的 4 亿对图像-文本进行预训练。它由图像编码器 $f$ 和文本编码器 $g$ 组成，分别对图像和相应的自然语言描述进行编码。


Zero-shot CLIP directly incorporates a template text description as a prompt into the text encoder, such as "a photo of a [CLASS]" where [CLASS] denotes the class token. The image features and text features $w$ of manually designed prompts are extracted from the image encoder and text encoder, respectively. The prediction of the class of the image is $\widehat{y} = \arg \max \left\langle  {f\left( \mathbf{x}\right) ,{w}_{k}}\right\rangle$ ,where $\langle  \cdot  , \cdot  \rangle$ denotes the cosine similarity.
零样本 CLIP 直接将模板文本描述作为提示嵌入到文本编码器中，例如“[CLASS] 的一张照片”，其中 [CLASS] 表示类别令牌。手工设计提示的图像特征和文本特征 $w$ 分别从图像编码器和文本编码器提取。图像的类别预测为 $\widehat{y} = \arg \max \left\langle  {f\left( \mathbf{x}\right) ,{w}_{k}}\right\rangle$，其中 $\langle  \cdot  , \cdot  \rangle$ 表示余弦相似度。


CoOp [40] introduces a set of continuous learnable context vectors $v$ concatenated with the template prompt $c$ ,namely soft prompt,then the $i$ th class of text prompt ${t}^{i}$ can be defined as ${t}^{i} = \left\lbrack  {v,{c}^{i}}\right\rbrack$ . Therefore the whole framework can be updated through the frozen CLIP model via tuning these prompts with cross-entropy loss ${\mathcal{L}}_{ce} =  - {\mathbb{E}}_{y}\left\lbrack  {\mathop{\sum }\limits_{i}{y}_{i}\log \left( {{\widehat{y}}_{i} \mid  {\mathbf{x}}_{i},{t}_{i}}\right) }\right\rbrack$ ,where ${\widehat{y}}_{i}$ denotes $i$ th element of the model's predicted probability distribution. However, CoOp learns a fixed prompt from training samples, which may lead to overfitting to the training distribution and degraded performance on test distribution.
CoOp [40] 引入一组连续可学习的上下文向量 $v$，与模板提示 $c$ 连接起来，即软提示，然后第 $i$ 个文本提示类别 ${t}^{i}$ 可以定义为 ${t}^{i} = \left\lbrack  {v,{c}^{i}}\right\rbrack$。因此，整个框架可以通过对这些提示在冻结 CLIP 模型上的交叉熵损失 ${\mathcal{L}}_{ce} =  - {\mathbb{E}}_{y}\left\lbrack  {\mathop{\sum }\limits_{i}{y}_{i}\log \left( {{\widehat{y}}_{i} \mid  {\mathbf{x}}_{i},{t}_{i}}\right) }\right\rbrack$ 进行微调来更新，其中 ${\widehat{y}}_{i}$ 表示模型预测概率分布的第 $i$ 个元素。然而，CoOp 仅从训练样本学习固定的提示，可能导致对训练分布的过拟合，进而在测试分布上的性能下降。


CoCoOp 39 and DPL 36 attempt to overcome distribution shifts by learning an instance-specific continuous prompt that is conditioned on the input image with an MLP layer $\phi$ . Both of them learn an image-conditional vector $r = \; \phi \left( {f\left( \mathbf{x}\right) }\right)$ ,which is then either added for CoCoOp with the learnable context vectors $v$ or concatenated for DPL. Then the $i$ th class of text prompt ${t}^{i}$ can be defined as ${t}^{i} = \left\lbrack  {v + r,{c}^{i}}\right\rbrack$ or ${t}^{i} = \left\lbrack  {r,{c}^{i}}\right\rbrack$ . By tuning the prompts and MLP layer, the loss is formulated as ${\mathcal{L}}_{ce} =  - {\mathbb{E}}_{y}\left\lbrack  {\mathop{\sum }\limits_{i}{y}_{i}\log \left( {{\widehat{y}}_{i} \mid  {\mathbf{x}}_{i},{t}_{i},\phi }\right) }\right\rbrack$ .
CoCoOp 39 和 DPL 36 尝试通过学习一个针对输入图像的实例特定连续提示来克服分布偏移，该提示由一个 MLP 层 $\phi$ 条件化。它们都学习一个图像条件向量 $r = \; \phi \left( {f\left( \mathbf{x}\right) }\right)$，然后要么对 CoCoOp 与可学习上下文向量 $v$ 相加，要么对 DPL 进行拼接。然后可以将文本提示 ${t}^{i}$ 的 $i$ 级别定义为 ${t}^{i} = \left\lbrack  {v + r,{c}^{i}}\right\rbrack$ 或 ${t}^{i} = \left\lbrack  {r,{c}^{i}}\right\rbrack$。通过对提示和 MLP 层的微调，损失函数表示为 ${\mathcal{L}}_{ce} =  - {\mathbb{E}}_{y}\left\lbrack  {\mathop{\sum }\limits_{i}{y}_{i}\log \left( {{\widehat{y}}_{i} \mid  {\mathbf{x}}_{i},{t}_{i},\phi }\right) }\right\rbrack$。


## 4 Method
## 4 方法


Different from previous prompt learning methods, we abandon the training paradigm of the fixed prompt and residualed prompt. Instead, we introduce a novel generative perspective to prompt learning, namely Soft Prompt Generation (SPG). Specifically, our method includes a two-stage training phase to train a generative model incorporated with domain information and an inference phase to generate prompts directly. In our method, the generation of soft prompts is exclusively handled by a generative model. Notably, the domain knowledge is stored within the generative model, making it possible for each image to generate an instance-specific prompt. Our method ensures the diversity of prompts and allows for the incorporation of domain-specific information. We introduce our SPG method as follows.
与以往的提示学习方法不同，我们放弃固定提示和残差提示的训练范式。相反，我们引入了一种新的生成式提示学习视角，即 Soft Prompt Generation（SPG）。具体地，我们的方法包含一个两阶段的训练阶段，用以训练结合领域信息的生成模型，以及一个推断阶段直接生成提示。在我们的方法中，软提示的生成仅由生成模型来处理。值得注意的是，领域知识被存储在生成模型中，使得每张图像都能生成一个实例特定的提示。我们的方法确保了提示的多样性，并允许结合领域特定信息。下面介绍我们的 SPG 方法。 


<img src="https://cdn.noedgeai.com/bo_d6aql6v7aajc739ardvg_5.jpg?x=440&y=335&w=930&h=504&r=0"/>



Fig. 2: The design of the second stage of the training phase. The condition generative adversarial net is the backbone of the generative model. The generator is guided by images to produce prompts. Meanwhile, the discriminator evaluates the authenticity of the prompt labels and the generated prompts with image data.
图 2：训练阶段二阶段的设计。条件生成对抗网络是生成模型的骨架。生成器在图像的引导下产生提示。与此同时，判别器评估提示标签与带有图像数据的生成提示的真实性。


Training Stage I: Domain Prompt Labels Learning. To better adapt our SPG method to the DG problem, we introduce the concept of domain prompt labels. Each domain corresponds to a domain prompt label ${\mathbf{v}}^{{d}_{i}}$ where ${d}_{i}$ denotes $i$ th domain,derived from training on the data of each domain with cross-entropy:
训练阶段 I：领域提示标签学习。为更好地将 SPG 方法适配于 DG 问题，我们引入领域提示标签的概念。每个领域对应一个领域提示标签 ${\mathbf{v}}^{{d}_{i}}$，其中 ${d}_{i}$ 表示来自不同领域数据训练得到的第 $i$ 个领域，采用交叉熵进行训练：


$$
{\mathbf{v}}^{{d}_{i} * } = \arg \mathop{\min }\limits_{\mathbf{v}}{\mathbb{E}}_{{\mathbf{x}}_{j}^{{d}_{i}},{y}_{j}^{{d}_{i}}}\left\lbrack  {-\log p\left( {{y}_{j}^{{d}_{i}} \mid  {\mathbf{x}}_{j}^{{d}_{i}},{\mathbf{v}}^{{d}_{i}}}\right) }\right\rbrack  , \tag{1}
$$



where $\left( {{\mathbf{x}}_{j}^{{d}_{i}},{y}_{j}^{{d}_{i}}}\right)$ denotes images and labels of training samples in $i$ th domain. The prompt design follows Zhou et al. 40 with text prompt tuning. These domain prompt labels represent optimal prompts for each domain, which encapsulate rich domain information.
其中 $\left( {{\mathbf{x}}_{j}^{{d}_{i}},{y}_{j}^{{d}_{i}}}\right)$ 表示在 $i$ 域中的训练样本的图像和标签。提示设计遵循 Zhou 等人 40 的文本提示调优方法。这些领域提示标签代表每个领域的最优提示，蕴含丰富的领域信息。


Training Stage II: Generative Model Pre-training. We adopt a simple yet efficient generative model, conditional generative adversarial net (CGAN) [27], to demonstrate the effectiveness of our method. CGAN, an extension of the conventional generative adversarial network (GAN) framework, operates by conditioning the generation process on additional information, usually presented in the form of auxiliary input data or labels. The CGAN architecture includes a generator $G$ and a discriminator $D$ . The generator $G$ is guided by additional information, ensuring that the generated outputs align with the specified conditions. Meanwhile,the discriminator $D$ evaluates the authenticity of the real data and fake output, fostering a dynamic adversarial interplay that refines the overall generative capabilities of the model. In this work, the fake batch of images is randomly sampled from the dataset.
训练阶段 II：生成模型的预训练。我们采用一个简单但高效的生成模型，条件生成对抗网络（CGAN）[27]，以展示方法的有效性。CGAN，是对传统生成对抗网络（GAN）框架的扩展，通过以额外信息为条件来引导生成过程，通常以辅助输入数据或标签的形式呈现。CGAN 架构包含一个生成器 $G$ 和一个判别器 $D$。生成器 $G$ 在额外信息的引导下工作，确保生成输出与所给条件相一致。与此同时，判别器 $D$ 评估真实数据与伪输出的真实性，促进一种动态的对抗性互动，从而提升模型的整体生成能力。在本工作中，伪造的图像批次是从数据集随机抽取的。


As shown in Figure 2 we adapt the CGAN model to serve the DG task of prompt generation, aligning the generated prompts with their corresponding images. Our objective is to learn the generator’s distribution ${p}_{g}$ over soft prompt $\mathbf{v}$ , transferring the transferability from prompts to the generator. In the generator, we define a prior on input noise variables $\mathbf{z}$ ,and the joint hidden representation $\left\lbrack  {\mathbf{z},f\left( \mathbf{x}\right) }\right\rbrack$ combines these noise variables with image embeddings $f\left( \mathbf{x}\right)$ with concatenation operation. In the discriminator, domain prompt labels and the generated prompt with image embeddings $f\left( \mathbf{x}\right)$ serve as inputs to a discriminative function. Therefore,the objective function $V\left( {G,D}\right)$ of a two-player minimax game can be formulated as:
如图 2 所示，我们对 CGAN 模型进行改造，以用于提示生成的 DG 任务，使生成的提示与其对应图像对齐。我们的目标是学习生成器在软提示 $\mathbf{v}$ 上的分布 ${p}_{g}$，将可迁移性从提示转移到生成器。在生成器中，我们对输入噪声变量 $\mathbf{z}$ 设定先验，联合隐表示 $\left\lbrack  {\mathbf{z},f\left( \mathbf{x}\right) }\right\rbrack$ 将这些噪声变量与图像嵌入 $f\left( \mathbf{x}\right)$ 通过拼接运算结合。在判别器中，域提示标签及带有图像嵌入 $f\left( \mathbf{x}\right)$ 的生成提示作为输入进入判别函数。因此，两人博弈的目标函数 $V\left( {G,D}\right)$ 可以形式化为：


$$
\mathop{\min }\limits_{G}\mathop{\max }\limits_{D}V\left( {G,D}\right)  = {\mathbb{E}}_{\mathbf{v} \sim  {p}_{\mathbf{v}}\left( \mathbf{v}\right) }\left\lbrack  {\log D\left( {\mathbf{v} \mid  f\left( \mathbf{x}\right) }\right) }\right\rbrack \tag{2}
$$



$$
+ {\mathbb{E}}_{\mathbf{z} \sim  {p}_{\mathbf{z}}\left( \mathbf{z}\right) }\left\lbrack  {\log \left( {1 - D\left( {G\left( {\mathbf{z} \mid  f\left( \mathbf{x}\right) }\right) }\right) }\right) }\right\rbrack  .
$$



The pre-trained CGAN aims to capture domain-invariant and domain-specific features, ensuring consistency across diverse domains. Additionally, its generator maintains prompt diversity, enhancing the model's adaptability and generalization capabilities to handle varied inputs.
预训练的 CGAN 旨在捕捉域不变与域特征，确保在多样域间的一致性。此外，其生成器保持提示多样性，提升模型对不同输入的适应性与泛化能力。


Inference. The generator of CGAN is employed to produce a domain-specific soft prompt for each image of the target domain. The probability of the image belonging to the $i$ th class can be formulated as:
推理。CGAN 的生成器用于为目标域的每张图像生成一个域特定的软提示。图像属于第 $i$ 类的概率可表示为：


$$
p\left( {y = i \mid  \mathbf{x}}\right)  = \frac{\exp \left( {\left\langle  {{w}_{i},f\left( \mathbf{x}\right) }\right\rangle  /\tau }\right) }{\mathop{\sum }\limits_{{j = 1}}^{K}\exp \left( {\left\langle  {{w}_{j},f\left( \mathbf{x}\right) }\right\rangle  /\tau }\right) }, \tag{3}
$$



$$
\text{ s.t. }\;{w}_{i} = g\left( \left\lbrack  {G\left( {\mathbf{z} \mid  f\left( \mathbf{x}\right) }\right) ,{c}_{i}}\right\rbrack  \right) \text{ , } \tag{4}
$$



where $\tau$ denotes temperature parameter, $K$ denotes the number of classes, $g$ denotes the text encoder, $G\left( {\mathbf{z} \mid  f\left( \mathbf{x}\right) }\right)$ denotes the domain-specific soft prompt, ${c}_{i}$ denotes $i$ th tokenized class token,and $\left\lbrack  {\cdot , \cdot  }\right\rbrack$ denotes the concatenation operation. In this way, exclusively relying on a generative model for soft prompt production provides benefits in terms of adaptability and dynamic prompt generation. Our method fosters diversity in prompt generation and enhances generalization across various tasks and domains.
其中 $\tau$ 表示温度参数，$K$ 表示类别数，$g$ 表示文本编码器，$G\left( {\mathbf{z} \mid  f\left( \mathbf{x}\right) }\right)$ 表示域特定的软提示，${c}_{i}$ 表示第 $i$ 个标记的类别标记，以及 $\left\lbrack  {\cdot , \cdot  }\right\rbrack$ 表示拼接运算。通过这种方式，仅依赖生成模型来生成软提示，在适应性和动态提示生成方面具有优势。我们的方法促进了提示生成的多样性，并提升了对各类任务与域的泛化能力。


## 5 Experiments
## 5 实验


### 5.1 Experimental Setting
### 5.1 实验设置


Datasets. Experiments are conducted on five popular benchmark datasets of DG, namely PACS [21], VLCS [11], OfficeHome [33], TerraIncognita (Ter-raInc.) [3] and DomainNet [30]. The dataset details can be seen in the appendix.
数据集。实验在五个广泛使用的 DG 基准数据集上进行，分别是 PACS [21]、VLCS [11]、OfficeHome [33]、TerraIncognita (Ter-raInc.) [3] 和 DomainNet [30]。数据集细节可见附录。


Baselines. We consider two types of methods for comparisons. For the traditional DG methods, we report the performance of ERM [14], SWAD [4], MIRO [5]. For CLIP-based methods, we compare SPG method with zero-shot CLIP (ZS-CLIP) 32, linear probing of CLIP (Lin. Prob.) 32, CoOp 40, Co-CoOp 39, VPT 17, VP 11, MaPLe 18, DPL 36.
基线。我们考虑两种类型的方法进行比较。对于传统的 DG 方法，报告 ERM [14]、SWAD [4]、MIRO [5] 的性能。对于基于 CLIP 的方法，比较 SPG 方法与零-shot CLIP(ZS-CLIP) 32、CLIP 线性探测（Lin. Prob.）32、CoOp 40、Co-CoOp 39、VPT 17、VP 11、MaPLe 18、DPL 36。


Experimental Setup. ResNet50 (RN50) 15 and ViT-B/16 10 are adopted as our backbones. For our SPG method, in the first stage, we train the domain prompt labels using the SGD optimizer with a batch size of 32 and the context length of the prompt label is 4. In the second stage, we train the CGAN model using the AdamW optimizer with weight decay 1e-4 and betas (0.9, 0.999), starting with a learning rate of 2e-3 for PACS, VLCS and TerraIncognita dataset, and a learning rate of 2e-4 for OfficeHome and DomainNet dataset. To enhance the stability of the CGAN, we incorporate a gradient clipping strategy to impose conditional constraints on the learnable parameter. For other CLIP-based methods, we set the learning rate initially to around 2e-3 and decay it using a cosine annealing rule with 20 epochs. We also set the batch size to 32. Moreover, the context tokens length is set to 2 for MaPLe, 10 for VPT and VP, and 16 for CoOp and CoCoOp. We employ the training-domain validation set method for model selection for all methods, selecting the model that achieves the highest accuracy on the overall validation set.
实验设置。采用 ResNet50（RN50）15 和 ViT-B/16 10 作为骨干网络。对于我们的 SPG 方法，第一阶段使用 SGD 优化器训练域提示标签，批量大小为 32，提示标签的上下文长度为 4。第二阶段使用 AdamW 优化器训练 CGAN 模型，权重衰减为 1e-4，betas 为 (0.9, 0.999)，PACS、VLCS 和 TerraIncognita 数据集起始学习率为 2e-3，OfficeHome 和 DomainNet 数据集为 2e-4。为提升 CGAN 的稳定性，我们引入梯度裁剪策略，对可学习参数施加条件约束。对于其他基于 CLIP 的方法，初始学习率设为约 2e-3，并按余弦退火规则衰减 20 个 epoch。批量大小也设为 32。此外，MaPLe 的上下文标记长度为 2，VPT 和 VP 为 10，CoOp 和 CoCoOp 为 16。我们对所有方法采用在训练域上进行验证的模型选择策略，选择在整体验证集上达到最高准确率的模型。


### 5.2 Comparison with SOTA Methods
### 5.2 与 SOTA 方法的比较


Multi-source Domain Generalization. We follow the leave-one-domain-out evaluation protocol 14 for multi-source domain generalization. In this protocol, the model excludes one domain from the training set in each evaluation round and is then tested on the excluded domain. This iterative process continues until each domain has been excluded once.
多源域泛化。我们遵循通用的多源域泛化的“留一域评估”协议 14。在该协议中，每轮评估时模型从训练集中排除一个域，然后在被排除的域上进行测试。该迭代过程持续进行，直到每个域都被排除一次。


In Table 1, we report the mean leave-one-domain-out performance of PACS, VLCS, Office-Home, TerraIncognita, and DomainNet for both ResNet50 and ViT-B/16 backbones. We can observe that SPG outperforms all other CLIP-based methods on five datasets with two backbones. For instance, with ResNet50 as the backbone, SPG can surpass zero-shot CLIP and the state-of-the-art (SOTA) CLIP-based method with a large margin of 13.7% and 3.4% for Ter-raIncognita. SPG also achieves a large improvement of around ${4.0}\%$ and 2.2% compared with zero-shot CLIP and the SOTA method CoCoOp for VLCS. SPG achieves a convincing improvement of approximately 1.5% on the averaged results of five benchmarks, establishing a new SOTA for the multi-source DG task. The results underscore the potential of generative models for prompt learning.
在表1中，我们报告了PACS、VLCS、Office-Home、TerraIncognita和DomainNet在ResNet50和ViT-B/16两种骨干网络下的平均“留一域外”性能。可以观察到，SPG在五个数据集、两种骨干网络上均超越所有其他基于CLIP的方法。例如，以ResNet50为骨干时，SPG在TerraIncognita上相对于零-shot CLIP和SOTA基于CLIP的方法分别有13.7%和3.4%的巨大提升。与VLCS相比，SPG在零-shot CLIP和SOTA方法CoCoOp之间也取得了约${4.0}\%$和2.2%的显著提升。SPG在五个基准的平均结果上实现了约1.5%的令人信服的提升，为多源DG任务确立了新的SOTA。结果凸显了生成模型在提示学习中的潜力。


Table 1: Comparisons with SOTA methods on five domain generalization benchmark datasets for multi-source DG in terms of mean leave-one-domain-out performance with ResNet50 and ViT-B/16 as the backbone (B.). The results marked by $\dagger$ are the reported numbers from the original papers. Average accuracies and standard errors are reported from three trials. Bold denotes the best score.
表1：在五个域泛化基准数据集上，使用ResNet50和ViT-B/16作为骨干的多源DG在平均留一域外性能方面的与SOTA方法的比较（B.）。标注为$\dagger$的结果为原文献中的报道数值。平均准确率和标准误差来自三次试验。粗体表示最佳分数。


<table><tr><td>Method</td><td>B.</td><td>PACS</td><td>VLCS</td><td>OfficeHome</td><td>TerraInc.</td><td>DomainNet</td><td>Avg</td></tr><tr><td>ERM ${}^{ \dagger  }$ 14</td><td rowspan="3">RN50</td><td>85.7±0.5</td><td>77.4±0.3</td><td>67.5±0.5</td><td>47.2±0.4</td><td>41.2±0.2</td><td>63.8</td></tr><tr><td>MIRO</td><td>85.4±0.4</td><td>79.0±0.0</td><td>70.5±0.4</td><td>50.4±1.1</td><td>44.3±0.2</td><td>65.9</td></tr><tr><td>SWAD ${}^{ \dagger  }$ 4</td><td>88.1±0.1</td><td>79.1±0.1</td><td>70.6±0.2</td><td>50.0±0.3</td><td>46.5±0.1</td><td>66.9</td></tr><tr><td>ZS-CLIP 32</td><td rowspan="7">CLIP RN50</td><td>90.7±0.0</td><td>80.0±0.0</td><td>70.8±0.0</td><td>23.8±0.0</td><td>46.4±0.0</td><td>62.3</td></tr><tr><td>Lin. Prob. 32</td><td>90.6±0.3</td><td>79.8±0.4</td><td>65.5±0.2</td><td>33.0±1.2</td><td>27.1±0.2</td><td>59.2</td></tr><tr><td>CoOp [40]</td><td>91.3±0.3</td><td>81.4±0.2</td><td>73.5±0.2</td><td>33.2±3.4</td><td>49.7±0.2</td><td>65.9</td></tr><tr><td>CoCoOp [39]</td><td>91.9±0.6</td><td>81.8±0.3</td><td>73.4±0.4</td><td>34.1±3.0</td><td>49.7±0.1</td><td>66.3</td></tr><tr><td>DPL 36</td><td>91.8±0.7</td><td>80.8±0.8</td><td>73.6±0.4</td><td>34.4±1.0</td><td>49.6±0.2</td><td>66.0</td></tr><tr><td>VP 1</td><td>90.2±0.1</td><td>80.5±0.3</td><td>70.2±0.2</td><td>25.6±1.0</td><td>45.8±0.1</td><td>62.4</td></tr><tr><td>SPG (ours)</td><td>92.8±0.2</td><td>84.0±1.1</td><td>73.8±0.5</td><td>37.5±1.8</td><td>50.1±0.2</td><td>67.5</td></tr><tr><td>ZS-CLIP 32</td><td rowspan="9">CLIP ViT-B/16</td><td>96.1±0.0</td><td>82.3±0.0</td><td>81.8±0.0</td><td>33.8±0.0</td><td>56.6±0.0</td><td>70.2</td></tr><tr><td>Lin. Prob. 32</td><td>94.9±1.4</td><td>77.5±0.7</td><td>79.3±0.2</td><td>44.6±2.1</td><td>48.2±0.2</td><td>68.9</td></tr><tr><td>CoOp 40</td><td>96.4±0.3</td><td>80.8±0.3</td><td>83.0±0.1</td><td>46.8±0.7</td><td>59.5±0.2</td><td>73.6</td></tr><tr><td>CoCoOp 39</td><td>96.7±0.2</td><td>80.3±0.3</td><td>83.4±0.2</td><td>45.3±2.4</td><td>59.4±0.2</td><td>73.2</td></tr><tr><td>DPL 36</td><td>96.4±0.3</td><td>80.9±0.5</td><td>83.0±0.3</td><td>46.6±0.8</td><td>59.5±0.3</td><td>73.6</td></tr><tr><td>VP 1</td><td>95.8±0.1</td><td>82.2±0.0</td><td>81.2±0.2</td><td>34.9±0.2</td><td>${56.5} \pm  {0.0}$</td><td>70.1</td></tr><tr><td>VPT 17</td><td>96.9±0.2</td><td>82.0±0.2</td><td>83.2±0.1</td><td>46.7±0.6</td><td>${58.5} \pm  {0.2}$</td><td>73.6</td></tr><tr><td>MaPLe 18</td><td>96.5±0.2</td><td>82.2±0.2</td><td>83.4±0.0</td><td>50.2±0.9</td><td>59.5±0.3</td><td>74.4</td></tr><tr><td>SPG (ours)</td><td>97.0±0.5</td><td>82.4±0.4</td><td>83.6±0.4</td><td>50.2±1.2</td><td>60.1±0.5</td><td>74.7</td></tr></table>
<table><tbody><tr><td>方法</td><td>B.</td><td>PACS</td><td>VLCS</td><td>OfficeHome</td><td>TerraInc.</td><td>DomainNet</td><td>平均</td></tr><tr><td>ERM ${}^{ \dagger  }$ 14</td><td rowspan="3">RN50</td><td>85.7±0.5</td><td>77.4±0.3</td><td>67.5±0.5</td><td>47.2±0.4</td><td>41.2±0.2</td><td>63.8</td></tr><tr><td>MIRO</td><td>85.4±0.4</td><td>79.0±0.0</td><td>70.5±0.4</td><td>50.4±1.1</td><td>44.3±0.2</td><td>65.9</td></tr><tr><td>SWAD ${}^{ \dagger  }$ 4</td><td>88.1±0.1</td><td>79.1±0.1</td><td>70.6±0.2</td><td>50.0±0.3</td><td>46.5±0.1</td><td>66.9</td></tr><tr><td>ZS-CLIP 32</td><td rowspan="7">CLIP RN50</td><td>90.7±0.0</td><td>80.0±0.0</td><td>70.8±0.0</td><td>23.8±0.0</td><td>46.4±0.0</td><td>62.3</td></tr><tr><td>线性概率 32</td><td>90.6±0.3</td><td>79.8±0.4</td><td>65.5±0.2</td><td>33.0±1.2</td><td>27.1±0.2</td><td>59.2</td></tr><tr><td>CoOp [40]</td><td>91.3±0.3</td><td>81.4±0.2</td><td>73.5±0.2</td><td>33.2±3.4</td><td>49.7±0.2</td><td>65.9</td></tr><tr><td>CoCoOp [39]</td><td>91.9±0.6</td><td>81.8±0.3</td><td>73.4±0.4</td><td>34.1±3.0</td><td>49.7±0.1</td><td>66.3</td></tr><tr><td>DPL 36</td><td>91.8±0.7</td><td>80.8±0.8</td><td>73.6±0.4</td><td>34.4±1.0</td><td>49.6±0.2</td><td>66.0</td></tr><tr><td>VP 1</td><td>90.2±0.1</td><td>80.5±0.3</td><td>70.2±0.2</td><td>25.6±1.0</td><td>45.8±0.1</td><td>62.4</td></tr><tr><td>SPG（ours）</td><td>92.8±0.2</td><td>84.0±1.1</td><td>73.8±0.5</td><td>37.5±1.8</td><td>50.1±0.2</td><td>67.5</td></tr><tr><td>ZS-CLIP 32</td><td rowspan="9">CLIP ViT-B/16</td><td>96.1±0.0</td><td>82.3±0.0</td><td>81.8±0.0</td><td>33.8±0.0</td><td>56.6±0.0</td><td>70.2</td></tr><tr><td>线性概率 32</td><td>94.9±1.4</td><td>77.5±0.7</td><td>79.3±0.2</td><td>44.6±2.1</td><td>48.2±0.2</td><td>68.9</td></tr><tr><td>CoOp 40</td><td>96.4±0.3</td><td>80.8±0.3</td><td>83.0±0.1</td><td>46.8±0.7</td><td>59.5±0.2</td><td>73.6</td></tr><tr><td>CoCoOp 39</td><td>96.7±0.2</td><td>80.3±0.3</td><td>83.4±0.2</td><td>45.3±2.4</td><td>59.4±0.2</td><td>73.2</td></tr><tr><td>DPL 36</td><td>96.4±0.3</td><td>80.9±0.5</td><td>83.0±0.3</td><td>46.6±0.8</td><td>59.5±0.3</td><td>73.6</td></tr><tr><td>VP 1</td><td>95.8±0.1</td><td>82.2±0.0</td><td>81.2±0.2</td><td>34.9±0.2</td><td>${56.5} \pm  {0.0}$</td><td>70.1</td></tr><tr><td>VPT 17</td><td>96.9±0.2</td><td>82.0±0.2</td><td>83.2±0.1</td><td>46.7±0.6</td><td>${58.5} \pm  {0.2}$</td><td>73.6</td></tr><tr><td>MaPLe 18</td><td>96.5±0.2</td><td>82.2±0.2</td><td>83.4±0.0</td><td>50.2±0.9</td><td>59.5±0.3</td><td>74.4</td></tr><tr><td>SPG（ours）</td><td>97.0±0.5</td><td>82.4±0.4</td><td>83.6±0.4</td><td>50.2±1.2</td><td>60.1±0.5</td><td>74.7</td></tr></tbody></table>


Meanwhile, SPG with ResNet50 as the backbone outperforms the SOTA traditional DG methods by a large margin of ${4.7}\% ,{4.9}\% ,{3.2}\%$ ,and ${3.6}\%$ for PACS, VLCS, OfficeHome, and DomainNet datasets, respectively. We also observe that traditional DG methods achieve the SOTA performance on the TerraIncognita dataset, surpassing both the zero-shot CLIP model and our method. This discrepancy may arise from the fact that CLIP was not pre-trained on data similar to TerraIncognita, highlighting the need for further exploration of VLMs on unseen domain data and other downstream tasks.
同时，使用 ResNet50 作为骨干的 SPG 在 PACS、VLCS、OfficeHome 和 DomainNet 数据集上分别以 ${4.7}\% ,{4.9}\% ,{3.2}\%$ 的巨大优势和 ${3.6}\%$ 超越了最先进的传统 DG 方法。我们还观察到，传统 DG 方法在 TerraIncognita 数据集上达到 SOTA 性能，超过了零-shot CLIP 模型和我们的方法。此差异可能源于 CLIP 未在与 TerraIncognita 相似的数据上进行预训练，凸显了在未见域数据及其他下游任务上进一步探索 VLM 的必要性。


Single-source Domain Generalization. The leave-all-but-one-domain-out evaluation protocol is adopted for single-source domain generalization. Under this protocol, all domains except one are included in the training set, and the model is then tested on the remaining domain.
单源域泛化。采用保留除一个域以外的所有域的评估协议进行单源域泛化。在该协议下，训练集中包含除一个域以外的所有域，然后在剩余的一个域上对模型进行测试。


Table 2 shows the experimental results of the leave-all-but-one-domain-out performance of PACS, VLCS, Office-Home, TerraIncognita and DomainNet for ResNet50 backbone. SPG outperforms other CLIP-based fine-tuning methods on five datasets. We observe averaged improvements of 19.0% and 1.4% compared with linear probing of CLIP and the SOTA method CoOp, respectively. This demonstrates that by leveraging the generative model for prompt generation, SPG is able to produce more domain-relevant and adaptive prompts, leading to improved performance across different domains. The domain-wise results for both multi-source DG and single-source DG are provided in the appendix.
表 2 给出 ResNet50 骨干下 PACS、VLCS、Office-Home、TerraIncognita 与 DomainNet 的留一域外全部域（leave-all-but-one-domain-out）性能的实验结果。SPG 在五个数据集上均优于其他基于 CLIP 的微调方法。相比 CLIP 的线性探针，和 SOTA 方法 CoOp，平均提升分别为 19.0% 和 1.4%。这表明通过利用生成模型进行提示生成，SPG 能产出更具领域相关性和自适应性的提示，从而在不同域上实现更好表现。多源 DG 与单源 DG 的域级结果在附录中给出。


Table 2: Comparisons with CLIP-based fine-tuning methods on five domain generalization benchmark datasets for single-source DG in terms of leave-all-but-one-domain-out performance with ResNet50 as the backbone. Bold denotes the best scores.
表 2：在五个域泛化基准数据集上，以 ResNet50 为骨干的单源 DG，进行留一域外全部域性能的比较，与 CLIP 为基础的微调方法对比。粗体表示最佳分数。


<table><tr><td>Method</td><td>PACS</td><td>VLCS</td><td>OfficeHome</td><td>TerraInc.</td><td>DomainNet</td><td>Avg</td></tr><tr><td>Lin. Prob. 32</td><td>77.3</td><td>65.5</td><td>46.4</td><td>23.3</td><td>6.3</td><td>43.8</td></tr><tr><td>CoOp 40</td><td>86.0</td><td>75.3</td><td>70.7</td><td>30.7</td><td>44.9</td><td>61.4</td></tr><tr><td>CoCoOp [39]</td><td>88.1</td><td>68.2</td><td>70.6</td><td>25.6</td><td>45.5</td><td>59.6</td></tr><tr><td>DPL 36</td><td>86.8</td><td>75.2</td><td>70.8</td><td>28.4</td><td>43.8</td><td>61.0</td></tr><tr><td>SPG (ours)</td><td>88.8</td><td>76.5</td><td>70.9</td><td>32.3</td><td>45.6</td><td>62.8</td></tr></table>
<table><tbody><tr><td>方法</td><td>PACS</td><td>VLCS</td><td>OfficeHome</td><td>TerraInc.</td><td>DomainNet</td><td>平均值</td></tr><tr><td>线性概率 32</td><td>77.3</td><td>65.5</td><td>46.4</td><td>23.3</td><td>6.3</td><td>43.8</td></tr><tr><td>CoOp 40</td><td>86.0</td><td>75.3</td><td>70.7</td><td>30.7</td><td>44.9</td><td>61.4</td></tr><tr><td>CoCoOp [39]</td><td>88.1</td><td>68.2</td><td>70.6</td><td>25.6</td><td>45.5</td><td>59.6</td></tr><tr><td>DPL 36</td><td>86.8</td><td>75.2</td><td>70.8</td><td>28.4</td><td>43.8</td><td>61.0</td></tr><tr><td>SPG (ours)</td><td>88.8</td><td>76.5</td><td>70.9</td><td>32.3</td><td>45.6</td><td>62.8</td></tr></tbody></table>


Table 3: Comparisons with CLIP-based methods on four domain generalization benchmark datasets for cross-dataset generalization performance with ViT-B/16 as the backbone. Bold denotes the best scores.
表 3：在四个域泛化基准数据集上基于 CLIP 的方法比较，跨数据集泛化性能以 ViT-B/16 作为骨架模型。加粗表示最佳分数。


<table><tr><td>Method</td><td>PACS</td><td>VLCS</td><td>OfficeHome</td><td>TerraInc.</td><td>Avg</td></tr><tr><td>ZS-CLIP 32</td><td>94.5</td><td>80.5</td><td>82.1</td><td>30.9</td><td>72.0</td></tr><tr><td>CoOp 40</td><td>96.2</td><td>72.8</td><td>83.1</td><td>32.1</td><td>71.1</td></tr><tr><td>CoCoOp 39</td><td>95.8</td><td>72.6</td><td>83.3</td><td>34.2</td><td>71.5</td></tr><tr><td>DPL 36</td><td>96.0</td><td>72.7</td><td>83.6</td><td>30.2</td><td>70.6</td></tr><tr><td>VPT 17</td><td>95.7</td><td>77.7</td><td>82.5</td><td>26.2</td><td>70.5</td></tr><tr><td>MaPLe 18</td><td>95.1</td><td>74.4</td><td>83.9</td><td>27.3</td><td>70.2</td></tr><tr><td>SPG (ours)</td><td>96.7</td><td>76.7</td><td>83.9</td><td>38.0</td><td>73.8</td></tr></table>
<table><tbody><tr><td>方法</td><td>PACS</td><td>VLCS</td><td>OfficeHome</td><td>TerraInc.</td><td>平均</td></tr><tr><td>ZS-CLIP 32</td><td>94.5</td><td>80.5</td><td>82.1</td><td>30.9</td><td>72.0</td></tr><tr><td>CoOp 40</td><td>96.2</td><td>72.8</td><td>83.1</td><td>32.1</td><td>71.1</td></tr><tr><td>CoCoOp 39</td><td>95.8</td><td>72.6</td><td>83.3</td><td>34.2</td><td>71.5</td></tr><tr><td>DPL 36</td><td>96.0</td><td>72.7</td><td>83.6</td><td>30.2</td><td>70.6</td></tr><tr><td>VPT 17</td><td>95.7</td><td>77.7</td><td>82.5</td><td>26.2</td><td>70.5</td></tr><tr><td>MaPLe 18</td><td>95.1</td><td>74.4</td><td>83.9</td><td>27.3</td><td>70.2</td></tr><tr><td>SPG (ours)</td><td>96.7</td><td>76.7</td><td>83.9</td><td>38.0</td><td>73.8</td></tr></tbody></table>


Cross-Dataset Generalization. We split the DomainNet subset into training and validation datasets. With training-domain validation set model selection, we train on training data of DomainNet and select the best model on validation data of DomainNet. Then we test the best model on four downstream datasets, i.e., PACS, VLCS, OfficeHome and TerraIncognita.
跨数据集泛化。我们将 DomainNet 子集划分为训练集和验证集。通过训练域验证集进行模型选择，然后在 DomainNet 的训练数据上训练，并在 DomainNet 的验证数据上选择最佳模型。随后在四个下游数据集上测试最佳模型，即 PACS、VLCS、OfficeHome 和 TerraIncognita。


As shown in Table 3, SPG outperforms all other CLIP-based methods, indicating the effectiveness of our generative prompt learning method in tackling distribution shifts. SPG achieves an average accuracy improvement of 1.8% compared with the zero-shot CLIP. It is also noteworthy that there is a stable improvement in the performance of baseline methods for the PACS and Office-Home datasets, such as CoOp, CoCoOp, etc. We attribute this to the similarity in the distribution between DomainNet and these downstream datasets. However, when encountering a significant distribution shift, such as observed with the VLCS and TerraIncognita, the performance of these methods may decline due to the learned prompts not being able to generalize to the new distribution. For TerraIncognita dataset, our SPG method can mitigate this challenge to some extent, the generative model is capable of dynamically generating prompts for each image of unseen distribution, thus enabling more adaptable prompt generation. Nevertheless, it remains particularly challenging for the VLCS dataset, which indicates a need for further exploration.
如表 3 所示，SPG 的表现优于所有其他基于 CLIP 的方法，表明我们在处理分布转移时的生成式提示学习方法的有效性。SPG 相较于零-shot CLIP 平均准确率提高了 1.8%。同样值得注意的是，在 PACS 和 Office-Home 数据集上，基线方法如 CoOp、CoCoOp 等也呈现出稳定的性能提升。这归因于 DomainNet 与这些下游数据集在分布上的相似性。然而，当遇到显著的分布转移时，例如在 VLCS 和 TerraIncognita 上观察到的，因学习到的提示无法泛化到新分布，这些方法的性能可能下降。对于 TerraIncognita 数据集，我们的 SPG 方法在一定程度上缓解了这一挑战，生成模型能够对未见分布的每张图像动态生成提示，从而实现更具适应性的提示生成。然而，对于 VLCS 数据集仍然尤为具有挑战性，这表明需要进一步探索。


<img src="https://cdn.noedgeai.com/bo_d6aql6v7aajc739ardvg_10.jpg?x=473&y=326&w=862&h=740&r=0"/>



Fig. 3: The t-sne visualization of the prompt embeddings for CoCoOp, DPL, and our SPG method. Multi-source domain generalization models on 3 tasks of PACS dataset are employed to obtain prompt embeddings. Different colors denote different classes. All the domains including the target domain are highly clustered in SPG.
Figure 3：CoCoOp、DPL 和我们 SPG 方法的提示嵌入的 t-SNE 可视化。对 PACS 数据集的3项多源域泛化模型进行应用，以获取提示嵌入。不同颜色表示不同类别。包括目标域在内的所有域在 SPG 中高度聚簇。


### 5.3 Visualization
### 5.3 可视化


The t-SNE visualization. In Figure 3, we qualitatively evaluate prompt em-beddings synthesized by baseline methods and our SPG method for three multi-source domain generalization tasks of PACS using t-SNE visualization [9]. We aim to generate a domain-specific prompt for each image, which ensures that prompt diversity and domain knowledge benefit the DG tasks. As illustrated in Figure 3 the prompts generated by our SPG method demonstrate a higher degree of clustering compared to prompts generated by alternative methods such as CoCoOp and DPL. This improved clustering indicates a higher degree of discriminative ability of the prompts, suggesting that our SPG method is more effective at capturing domain knowledge during prompt learning. The clustered domain prompts may also facilitate a better understanding of the semantic relationship between images and text, allowing the model to focus more on comprehending class concepts when aligning with the image encoder.
t-SNE 可视化。在图 3 中，我们使用 t-SNE 可视化对基线方法和我们 SPG 方法对 PACS 的三个多源域泛化任务所合成的提示嵌入进行了定性评估。我们的目标是为每张图像生成一个领域特定的提示，确保提示的多样性和领域知识有利于 DG 任务。如图 3 所示，我们的 SPG 方法生成的提示相较于 CoCoOp、DPL 等替代方法生成的提示，显示出更高的聚类度。这一更强的聚类性表明提示具有更高的判别能力，说明在提示学习期间 SPG 能更有效地捕捉领域知识。聚类的领域提示也可能促进对图像与文本之间语义关系的更好理解，使模型在与图像编码器对齐时更关注于理解类别概念。


<img src="https://cdn.noedgeai.com/bo_d6aql6v7aajc739ardvg_11.jpg?x=412&y=328&w=981&h=521&r=0"/>



Fig. 4: The image-prompt-image retrieval experiment is designed to demonstrate the correlation between the prompts and the images. We present the top-2 results of image retrieval conducted using CoOp, CoCoOp, DPL, and our SPG method. Images encased in red rectangles indicate instances where the query image and the retrieval image belong to the same domain.
Figure 4：图像-提示-图像检索实验旨在展示提示与图像之间的相关性。我们给出使用 CoOp、CoCoOp、DPL 和 SPG 方法进行的图像检索前 2 名结果。用红色框标注的图像表示查询图像和检索图像属于同一领域的实例。


Image retrieval. We designed an image-prompt-image retrieval experiment to demonstrate the correlation between the generated prompt and the images in each domain. Specifically,we first sampled $N$ -way $K$ -shot images from each domain in the dataset and combined them into an image library,where $N$ is 7 and $K$ is 10 for PACS. Then,we randomly select one image as a query from the remaining images. The generated prompt of each query image is concatenated with the class token, which is passed through a text encoder to calculate the probability distribution with the features of each image in the image library. Subsequently, we select the probability value corresponding to the class of the query image as the confidence score. These scores are then sorted in sequence, and the top-2 images are obtained as the results of the image retrieval.
图像检索。我们设计了一个图像-提示-图像检索实验，以展示生成的提示与各领域中图像之间的相关性。具体来说，我们先从数据集中各领域抽取 $N$-类 $K$--shot 的图像，然后将它们合并成一个图像库，其中 $N$ 在 PACS 中为 7，$K$ 为 10。随后，我们从剩余图像中随机选取一张作为查询图像。将每个查询图像生成的提示与类别标记拼接，并传入文本编码器，以计算查询图像在图像库中各图像的特征分布的概率分布。随后，我们将对应查询图像类别的概率值作为置信分数。将这些分数按序排序，选取前 2 张图像作为图像检索结果。


In Figure 4, we can observe that the prompts generated by our SPG method are capable of retrieving images that are more closely aligned with the query image, implying that the prompts generated by our model for the query image are more closely related to domain-specific information. It shows the effectiveness of our SPG method in encoding domain-relevant information within the prompts with a generative model. While previous methods have not emphasized the importance of domain knowledge, our SPG method stands out by explicitly integrating domain-specific information into the prompt generation process.
在图 4 中，我们可以看到由 SPG 方法生成的提示更能检索到与查询图像更一致的图像，意味着我们模型对查询图像所产生的提示与领域特定信息的相关性更高。这显示了我们通过生成模型将领域相关信息编码到提示中的有效性。尽管以往方法未强调领域知识的重要性，但我们的 SPG 方法通过在提示生成过程中显式整合领域特定信息而脱颖而出。


### 5.4 Ablation Study
### 5.4 消融研究


Ablation on the different design of prompt. We have summarized four characteristics of prompts: fixed, learnable, conditional, and generative. Based on these characteristics, we selected five methods for comparison: (1) Manual prompt: manually designed prompt as "a photo of a [CLS]", where [CLS] is the class token. (2) Mix-domain-prompt: weighted sum of domain prompts obtained from each source domain. We set equal weights for each domain. (3) All-domain-prompt: prompt obtained from all source domain data. (4) CoCoOp. (5) DPL.
对不同提示设计的消融。我们总结了提示的四个特性：固定、可学习、条件和生成。基于这些特性，我们选择了五种对比方法：(1) 手工提示：手工设计的提示为“某张 [CLS] 的照片”，其中 [CLS] 是类别标记。 (2) 混域提示：来自各源域的域提示的加权和。对每个域设定相等权重。 (3) 全域提示：来自所有源域数据的提示。 (4) CoCoOp。 (5) DPL。


Table 4: Comparisons with different design of prompt on five domain generalization benchmark datasets for multi-source DG performance with ResNet50 as the backbone. O.H. denotes OfficeHome, and Do.Net denotes DomainNet. Bold denotes the best scores.
表4：在五个域泛化基准数据集上对不同提示设计的对比，使用 ResNet50 作为骨干网的多源 DG 性能。O.H. 表示 OfficeHome，Do.Net 表示 DomainNet。粗体表示最佳分数。


<table><tr><td rowspan="2">Method</td><td colspan="4">Prompt Type</td><td colspan="5"></td></tr><tr><td>Fixed</td><td>Learnable</td><td>Conditional</td><td>Generative</td><td>PACS</td><td>VLCS</td><td>O.H.</td><td>TerraInc.</td><td>Do.Net</td></tr><tr><td>Manual Prompt</td><td>✓</td><td></td><td></td><td></td><td>90.7</td><td>80.0</td><td>70.8</td><td>23.8</td><td>46.4</td></tr><tr><td>Mix-domain-prompt</td><td>✓</td><td>✓</td><td></td><td></td><td>92.0</td><td>76.2</td><td>72.5</td><td>29.6</td><td>47.9</td></tr><tr><td>All-domain-prompt</td><td>✓</td><td>✓</td><td></td><td></td><td>91.3</td><td>81.4</td><td>73.2</td><td>33.2</td><td>49.7</td></tr><tr><td>CoCoOp 39</td><td></td><td>✓</td><td>✓</td><td></td><td>91.9</td><td>81.8</td><td>73.4</td><td>34.1</td><td>49.7</td></tr><tr><td>DPL 36</td><td></td><td>✓</td><td>✓</td><td></td><td>91.8</td><td>80.8</td><td>73.6</td><td>34.4</td><td>49.6</td></tr><tr><td>SPG (Ours)</td><td></td><td>✓</td><td></td><td>✓</td><td>92.8</td><td>84.0</td><td>73.8</td><td>37.5</td><td>50.1</td></tr></table>
<table><tbody><tr><td rowspan="2">方法</td><td colspan="4">提示类型</td><td colspan="5"></td></tr><tr><td>固定</td><td>可学习</td><td>条件</td><td>生成</td><td>PACS</td><td>VLCS</td><td>O.H.</td><td>TerraInc.</td><td>Do.Net</td></tr><tr><td>手动提示</td><td>✓</td><td></td><td></td><td></td><td>90.7</td><td>80.0</td><td>70.8</td><td>23.8</td><td>46.4</td></tr><tr><td>混域提示</td><td>✓</td><td>✓</td><td></td><td></td><td>92.0</td><td>76.2</td><td>72.5</td><td>29.6</td><td>47.9</td></tr><tr><td>全域提示</td><td>✓</td><td>✓</td><td></td><td></td><td>91.3</td><td>81.4</td><td>73.2</td><td>33.2</td><td>49.7</td></tr><tr><td>CoCoOp 39</td><td></td><td>✓</td><td>✓</td><td></td><td>91.9</td><td>81.8</td><td>73.4</td><td>34.1</td><td>49.7</td></tr><tr><td>DPL 36</td><td></td><td>✓</td><td>✓</td><td></td><td>91.8</td><td>80.8</td><td>73.6</td><td>34.4</td><td>49.6</td></tr><tr><td>SPG（Our）</td><td></td><td>✓</td><td></td><td>✓</td><td>92.8</td><td>84.0</td><td>73.8</td><td>37.5</td><td>50.1</td></tr></tbody></table>


Table 5: Ablation on the context length of prompt on PACS for multi-DG performance with ResNet50 as the backbone.
表 5：在 PACS 上对提示上下文长度进行消融分析，以 ResNet50 为骨干网络，评估多域任务性能。


<table><tr><td>Context Length</td><td>Art</td><td>Cartoon</td><td>Photo</td><td>Sketch</td><td>Avg</td></tr><tr><td>2</td><td>91.5</td><td>94.4</td><td>99.1</td><td>74.6</td><td>89.9</td></tr><tr><td>4</td><td>92.8</td><td>93.8</td><td>99.5</td><td>85.1</td><td>92.8</td></tr><tr><td>4 (random)</td><td>93.0</td><td>93.4</td><td>99.5</td><td>82.9</td><td>92.2</td></tr><tr><td>8</td><td>83.1</td><td>91.7</td><td>99.2</td><td>79.2</td><td>88.3</td></tr><tr><td>16</td><td>84.7</td><td>93.6</td><td>98.3</td><td>82.0</td><td>89.7</td></tr></table>
<table><tbody><tr><td>上下文长度</td><td>艺术</td><td>卡通</td><td>照片</td><td>素描</td><td>平均值</td></tr><tr><td>2</td><td>91.5</td><td>94.4</td><td>99.1</td><td>74.6</td><td>89.9</td></tr><tr><td>4</td><td>92.8</td><td>93.8</td><td>99.5</td><td>85.1</td><td>92.8</td></tr><tr><td>4（随机）</td><td>93.0</td><td>93.4</td><td>99.5</td><td>82.9</td><td>92.2</td></tr><tr><td>8</td><td>83.1</td><td>91.7</td><td>99.2</td><td>79.2</td><td>88.3</td></tr><tr><td>16</td><td>84.7</td><td>93.6</td><td>98.3</td><td>82.0</td><td>89.7</td></tr></tbody></table>


As shown in Table 4, in the multi-source DG setting, the manual prompt performs the worst, highlighting the necessity of fine-tuning for downstream tasks. For the learnable fixed prompt, the all-domain-prompt shows better performance than mix-domain-prompt. However, if the weights for mix-domain-prompt are more reasonable, there might be some improvement of performance, as some works have already attempted to explore this aspect 37 . Additionally, the conditional prompt methods outperform the fixed prompt methods, likely due to the integration of image information and a more diverse prompt. Finally, our generative prompt method performs the best, demonstrating its potential in prompt learning in VLMs.
如表4所示，在多源DG设置中，人工提示的表现最差，凸显了对下游任务进行微调的必要性。对于可学习的固定提示，全域提示的表现优于混域提示。然而，如果混域提示的权重更为合理，可能会有性能提升，因为一些工作已经尝试探索这一方面37。此外，条件提示方法的效果优于固定提示方法，可能是因为对图像信息的整合和提示更为多样化。最后，我们的生成提示方法表现最佳，展示了其在VLMs提示学习中的潜力。


Sensitivity of context length of prompt. As shown in Table 5, we evaluate the effects of different context lengths of prompt for multi-source DG on PACS using the ResNet50 backbone. We mainly initialize the soft prompt in two ways. One is to initialize it with the embedding of the text "a photo of a", namely word embeddings-based initialization. The other one is to sample from a zero-mean Gaussian distribution with a standard deviation of 0.02 (marked as random). In Table 5, we find that this initialization leads to slight improvement. Overall, we observe that the prompt context length of 4 provides the optimal performance for our SPG method.
提示的上下文长度敏感性。如下表5所示，我们在PACS数据集上以ResNet50骨干对多源DG的提示上下文长度 Effects 进行评估。我们主要以两种方式初始化软提示。一种是用“a photo of a”的文本嵌入来初始化，即基于词嵌入的初始化。另一种是从均值为0、标准差为0.02的零均值高斯分布中采样（标记为 random）。在表5中，我们发现这种初始化略有提升。总体而言，我们观察到提示上下文长度为4时为我们的SPG方法提供了最佳性能。


Table 6: Ablation on the training samples of the generative model (CGAN) on PACS for multi-DG performance with ResNet50 as the backbone.
表6：在以ResNet50为主干的PACS多DG性能中，对生成模型（CGAN）的训练样本进行消融分析。


<table><tr><td>Training Samples</td><td>Art</td><td>Cartoon</td><td>Photo</td><td>Sketch</td><td>Avg</td></tr><tr><td>20</td><td>30.1</td><td>34.9</td><td>23.4</td><td>34.5</td><td>30.7</td></tr><tr><td>40</td><td>51.2</td><td>55.3</td><td>34.3</td><td>65.5</td><td>51.6</td></tr><tr><td>60</td><td>73.8</td><td>76.7</td><td>56.8</td><td>81.1</td><td>72.1</td></tr><tr><td>80</td><td>89.2</td><td>93.5</td><td>84.8</td><td>86.4</td><td>88.5</td></tr><tr><td>full</td><td>92.8</td><td>93.8</td><td>99.5</td><td>85.1</td><td>92.8</td></tr></table>
<table><tbody><tr><td>训练样本</td><td>艺术</td><td>卡通</td><td>照片</td><td>素描</td><td>均值</td></tr><tr><td>20</td><td>30.1</td><td>34.9</td><td>23.4</td><td>34.5</td><td>30.7</td></tr><tr><td>40</td><td>51.2</td><td>55.3</td><td>34.3</td><td>65.5</td><td>51.6</td></tr><tr><td>60</td><td>73.8</td><td>76.7</td><td>56.8</td><td>81.1</td><td>72.1</td></tr><tr><td>80</td><td>89.2</td><td>93.5</td><td>84.8</td><td>86.4</td><td>88.5</td></tr><tr><td>完整</td><td>92.8</td><td>93.8</td><td>99.5</td><td>85.1</td><td>92.8</td></tr></tbody></table>


Sensitivity to the number of training samples. We sample different proportions of the training data for the generative model, including 20%, 40%, 60%, 80%, and all samples. Table 6 presents the results of our multi-source domain generalization experiments on the PACS dataset, and we observe a consistent increase in accuracy for all four tasks on the PACS as the amount of data for training the generative model increases. The performance reached its optimum when using all the data.
对训练样本数量的敏感性。我们对生成模型的训练数据按不同比例进行采样，包括 20%、40%、60%、80% 及全部样本。表 6 给出我们在 PACS 数据集上的多源领域泛化实验结果，随着用于训练生成模型的数据量增加，在 PACS 的四个任务中的准确率均呈现稳定提升。使用全部数据时，性能达到最优。


## 6 Conclusion
## 6 结论


In this paper, we reframe the prompt learning framework from a generative perspective, and are the first to introduce the generative model into prompt learning in Vision Language Models (VLMs). We propose a simple yet efficient Soft Prompt Generation (SPG) method for the Domain Generalization (DG). SPG is a new paradigm of prompt tuning, which consists of a two-stage training phase and an inference phase. In the training phase, we introduce the concept of domain prompt labels, which are adopted to incorporate the generative model with domain knowledge. During the inference phase, the generative model is employed to obtain domain-specific soft prompts for target domain data. SPG relies exclusively on a generative model to directly produce soft prompts. This preserves the diversity of generated soft prompts, aiming to learn domain information with the generative model, including the target domain. Extensive experiments on three DG tasks of five DG benchmark datasets confirm the effectiveness of our proposed SPG method. Compared with traditional DG methods and CLIP-based approaches, SPG achieves new state-of-the-art performance for domain generalization. We hope this research inspires future exploration into tapping the potential of generative models in prompt generation and learning.
本文从生成角度重新框定提示学习框架，首次将生成模型引入视觉语言模型（VLMs）的提示学习。我们提出一种简单而高效的领域泛化（DG） Soft Prompt Generation（SPG）方法。SPG 是一种新的提示微调范式，由两阶段训练和推断阶段组成。在训练阶段，我们引入领域提示标签的概念，将生成模型与领域知识结合起来。在推断阶段，使用生成模型为目标领域数据获得领域特定的软提示。SPG 完全依赖生成模型直接生成软提示，保留了生成软提示的多样性，旨在通过生成模型学习领域信息，包括目标领域。对五个 DG 基准数据集的三个 DG 任务进行的广泛实验证实了所提 SPG 方法的有效性。与传统 DG 方法和基于 CLIP 的方法相比，SPG 实现了领域泛化的新 State-of-the-art。我们希望这项研究能激发未来在提示生成与学习中充分挖掘生成模型潜力的探索。


## Acknowledgments
## 致谢


This work was supported by the National Natural Science Foundation of China under grant number U21A20485, 62088102.
本研究得到中国国家自然科学基金资助，资助号 U21A20485，62088102。


## References
## 参考文献


1. Bahng, H., Jahanian, A., Sankaranarayanan, S., Isola, P.: Exploring visual prompts for adapting large-scale models. arXiv preprint arXiv:2203.17274 (2022)
1. Bahng, H., Jahanian, A., Sankaranarayanan, S., Isola, P.: Exploring visual prompts for adapting large-scale models. arXiv preprint arXiv:2203.17274 (2022)


2. Bai, S., Zhang, M., Zhou, W., Huang, S., Luan, Z., Wang, D., Chen, B.: Prompt-based distribution alignment for unsupervised domain adaptation. In: Proceedings of the 38th AAAI Conference on Artificial Intelligence (AAAI 2024). AAAI Press (2024)
2. Bai, S., Zhang, M., Zhou, W., Huang, S., Luan, Z., Wang, D., Chen, B.: Prompt-based distribution alignment for unsupervised domain adaptation. In: Proceedings of the 38th AAAI Conference on Artificial Intelligence (AAAI 2024). AAAI Press (2024)


3. Beery, S., Van Horn, G., Perona, P.: Recognition in terra incognita. In: Proceedings of the European conference on computer vision (ECCV). pp. 456-473 (2018)
3. Beery, S., Van Horn, G., Perona, P.: Recognition in terra incognita. In: Proceedings of the European conference on computer vision (ECCV). pp. 456-473 (2018)


4. Cha, J., Chun, S., Lee, K., Cho, H.C., Park, S., Lee, Y., Park, S.: Swad: Domain generalization by seeking flat minima. Advances in Neural Information Processing Systems 34, 22405-22418 (2021)
4. Cha, J., Chun, S., Lee, K., Cho, H.C., Park, S., Lee, Y., Park, S.: Swad: Domain generalization by seeking flat minima. Advances in Neural Information Processing Systems 34, 22405-22418 (2021)


5. Cha, J., Lee, K., Park, S., Chun, S.: Domain generalization by mutual-information regularization with pre-trained models. In: European Conference on Computer Vision. pp. 440-457. Springer (2022)
5. Cha, J., Lee, K., Park, S., Chun, S.: Domain generalization by mutual-information regularization with pre-trained models. In: European Conference on Computer Vision. pp. 440-457. Springer (2022)


6. Cheng, D., Xu, Z., Jiang, X., Wang, N., Li, D., Gao, X.: Disentangled prompt representation for domain generalization. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 23595-23604 (2024)
6. Cheng, D., Xu, Z., Jiang, X., Wang, N., Li, D., Gao, X.: Disentangled prompt representation for domain generalization. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 23595-23604 (2024)


7. Cho, J., Nam, G., Kim, S., Yang, H., Kwak, S.: Promptstyler: Prompt-driven style generation for source-free domain generalization. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 15702-15712 (2023)
7. Cho, J., Nam, G., Kim, S., Yang, H., Kwak, S.: Promptstyler: Prompt-driven style generation for source-free domain generalization. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 15702-15712 (2023)


8. Derakhshani, M.M., Sanchez, E., Bulat, A., da Costa, V.G.T., Snoek, C.G., Tz-imiropoulos, G., Martinez, B.: Bayesian prompt learning for image-language model generalization. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 15237-15246 (2023)
8. Derakhshani, M.M., Sanchez, E., Bulat, A., da Costa, V.G.T., Snoek, C.G., Tz-imiropoulos, G., Martinez, B.: Bayesian prompt learning for image-language model generalization. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 15237-15246 (2023)


9. Donahue, J., Jia, Y., Vinyals, O., Hoffman, J., Zhang, N., Tzeng, E., Darrell, T.: Decaf: A deep convolutional activation feature for generic visual recognition. In: International conference on machine learning. pp. 647-655. PMLR (2014)
9. Donahue, J., Jia, Y., Vinyals, O., Hoffman, J., Zhang, N., Tzeng, E., Darrell, T.: Decaf: A deep convolutional activation feature for generic visual recognition. In: International conference on machine learning. pp. 647-655. PMLR (2014)


10. Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., et al.: An image is worth 16x16 words: Transformers for image recognition at scale. International Conference on Learning Representations (2021)
10. Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., et al.: 一张图片胜过16x16个词：大规模图像识别的 Transformers。 International Conference on Learning Representations (2021)


11. Fang, C., Xu, Y., Rockmore, D.N.: Unbiased metric learning: On the utilization of multiple datasets and web images for softening bias. In: Proceedings of the IEEE International Conference on Computer Vision. pp. 1657-1664 (2013)
11. Fang, C., Xu, Y., Rockmore, D.N.: 无偏度量学习：在多数据集与网络图像的利用中软化偏差。In: Proceedings of the IEEE International Conference on Computer Vision. pp. 1657-1664 (2013)


12. Ganin, Y., Ustinova, E., Ajakan, H., Germain, P., Larochelle, H., Laviolette, F., March, M., Lempitsky, V.: Domain-adversarial training of neural networks. Journal of machine learning research 17(59), 1-35 (2016)
12. Ganin, Y., Ustinova, E., Ajakan, H., Germain, P., Larochelle, H., Laviolette, F., March, M., Lempitsky, V.: 神经网络的领域对抗训练。Journal of machine learning research 17(59), 1-35 (2016)


13. Ge, C., Huang, R., Xie, M., Lai, Z., Song, S., Li, S., Huang, G.: Domain adaptation via prompt learning. IEEE Transactions on Neural Networks and Learning Systems (2023)
13. Ge, C., Huang, R., Xie, M., Lai, Z., Song, S., Li, S., Huang, G.: 通过提示学习进行领域自适应。IEEE Transactions on Neural Networks and Learning Systems (2023)


14. Gulrajani, I., Lopez-Paz, D.: In search of lost domain generalization. In: International Conference on Learning Representations (2020)
14. Gulrajani, I., Lopez-Paz, D.: 在寻觅失落的领域泛化。In: International Conference on Learning Representations (2020)


15. He, K., Zhang, X., Ren, S., Sun, J.: Deep residual learning for image recognition. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 770-778 (2016)
15. He, K., Zhang, X., Ren, S., Sun, J.: 深度残差学习用于图像识别。In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 770-778 (2016)


16. Jia, C., Yang, Y., Xia, Y., Chen, Y.T., Parekh, Z., Pham, H., Le, Q., Sung, Y.H., Li, Z., Duerig, T.: Scaling up visual and vision-language representation learning with noisy text supervision. In: International conference on machine learning. pp. 4904-4916. PMLR (2021)
16. Jia, C., Yang, Y., Xia, Y., Chen, Y.T., Parekh, Z., Pham, H., Le, Q., Sung, Y.H., Li, Z., Duerig, T.: 用带噪文本监督的大规模视觉与视觉-语言表示学习。In: International conference on machine learning. pp. 4904-4916. PMLR (2021)


17. Jia, M., Tang, L., Chen, B.C., Cardie, C., Belongie, S., Hariharan, B., Lim, S.N.: Visual prompt tuning. In: European Conference on Computer Vision. pp. 709-727. Springer (2022)
17. Jia, M., Tang, L., Chen, B.C., Cardie, C., Belongie, S., Hariharan, B., Lim, S.N.: Visual prompt tuning. In: European Conference on Computer Vision. pp. 709-727. Springer (2022)


18. Khattak, M.U., Rasheed, H., Maaz, M., Khan, S., Khan, F.S.: Maple: Multi-modal prompt learning. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 19113-19122 (2023)
18. Khattak, M.U., Rasheed, H., Maaz, M., Khan, S., Khan, F.S.: Maple: 多模态提示学习。In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 19113-19122 (2023)


19. Li, C., Liu, X., Wang, Y., Li, D., Lan, Y., Shen, C.: Dialogue for prompting: A policy-gradient-based discrete prompt generation for few-shot learning. In: Proceedings of the AAAI Conference on Artificial Intelligence. vol. 38, pp. 18481-18489 (2024)
19. Li, C., Liu, X., Wang, Y., Li, D., Lan, Y., Shen, C.: 对话式提示：一种基于策略梯度的离散提示生成用于小样本学习。In: Proceedings of the AAAI Conference on Artificial Intelligence. vol. 38, pp. 18481-18489 (2024)


20. Li, D., Yang, Y., Song, Y.Z., Hospedales, T.: Learning to generalize: Meta-learning for domain generalization. In: Proceedings of the AAAI conference on artificial intelligence. vol. 32 (2018)
20. Li, D., Yang, Y., Song, Y.Z., Hospedales, T.: 学习泛化：用于领域泛化的元学习。In: Proceedings of the AAAI conference on artificial intelligence. vol. 32 (2018)


21. Li, D., Yang, Y., Song, Y.Z., Hospedales, T.M.: Deeper, broader and artier domain generalization. In: Proceedings of the IEEE international conference on computer vision. pp. 5542-5550 (2017)
21. Li, D., Yang, Y., Song, Y.Z., Hospedales, T.M.: 更深更广更具艺术性的领域泛化。In: Proceedings of the IEEE国际会议 on computer vision. pp. 5542-5550 (2017)


22. Li, H., Pan, S.J., Wang, S., Kot, A.C.: Domain generalization with adversarial feature learning. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 5400-5409 (2018)
22. Li, H., Pan, S.J., Wang, S., Kot, A.C.: 通过对抗特征学习实现领域泛化。In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 5400-5409 (2018)


23. Li, H., Wang, Y., Wan, R., Wang, S., Li, T.Q., Kot, A.: Domain generalization for medical imaging classification with linear-dependency regularization. Advances in neural information processing systems 33, 3118-3129 (2020)
23. Li, H., Wang, Y., Wan, R., Wang, S., Li, T.Q., Kot, A.: 用线性相关性正则化的医学影像分类领域泛化。 Advances in neural information processing systems 33, 3118-3129 (2020)


24. Li, P., Li, D., Li, W., Gong, S., Fu, Y., Hospedales, T.M.: A simple feature augmentation for domain generalization. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 8886-8895 (2021)
24. Li, P., Li, D., Li, W., Gong, S., Fu, Y., Hospedales, T.M.: 一种简单的特征增强用于领域泛化。In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 8886-8895 (2021)


25. Li, Y., Tian, X., Gong, M., Liu, Y., Liu, T., Zhang, K., Tao, D.: Deep domain generalization via conditional invariant adversarial networks. In: Proceedings of the European conference on computer vision (ECCV). pp. 624-639 (2018)
25. Li, Y., Tian, X., Gong, M., Liu, Y., Liu, T., Zhang, K., Tao, D.: 通过条件不变量对抗网络实现深度领域泛化。In: Proceedings of the European conference on computer vision (ECCV). pp. 624-639 (2018)


26. Li, Z., Ren, K., Jiang, X., Shen, Y., Zhang, H., Li, D.: Simple: Specialized model-sample matching for domain generalization. In: The Eleventh International Conference on Learning Representations (2022)
26. Li, Z., Ren, K., Jiang, X., Shen, Y., Zhang, H., Li, D.: Simple: Specialized model-sample matching for domain generalization. In: The Eleventh International Conference on Learning Representations (2022)


27. Mirza, M., Osindero, S.: Conditional generative adversarial nets. arXiv preprint arXiv:1411.1784 (2014)
27. Mirza, M., Osindero, S.: Conditional generative adversarial nets. arXiv preprint arXiv:1411.1784 (2014)


28. Muandet, K., Balduzzi, D., Schölkopf, B.: Domain generalization via invariant feature representation. In: International conference on machine learning. pp. 10- 18. PMLR (2013)
28. Muandet, K., Balduzzi, D., Schölkopf, B.: Domain generalization via invariant feature representation. In: International conference on machine learning. pp. 10-18. PMLR (2013)


29. Niu, H., Li, H., Zhao, F., Li, B.: Domain-unified prompt representations for source-free domain generalization. arXiv preprint arXiv:2209.14926 (2022)
29. Niu, H., Li, H., Zhao, F., Li, B.: Domain-unified prompt representations for source-free domain generalization. arXiv preprint arXiv:2209.14926 (2022)


30. Peng, X., Bai, Q., Xia, X., Huang, Z., Saenko, K., Wang, B.: Moment matching for multi-source domain adaptation. In: Proceedings of the IEEE/CVF international conference on computer vision. pp. 1406-1415 (2019)
30. Peng, X., Bai, Q., Xia, X., Huang, Z., Saenko, K., Wang, B.: Moment matching for multi-source domain adaptation. In: Proceedings of the IEEE/CVF international conference on computer vision. pp. 1406-1415 (2019)


31. Qin, G., Eisner, J.: Learning How to Ask: Querying LMs with Mixtures of Soft Prompts. In: Proceedings of Annual Conference of the North American Chapter of the Association for Computational Linguistics (NAACL) (2021), https: //aclanthology.org/2021.naacl-main.410/
31. Qin, G., Eisner, J.: Learning How to Ask: Querying LMs with Mixtures of Soft Prompts. In: Proceedings of Annual Conference of the North American Chapter of the Association for Computational Linguistics (NAACL) (2021), https: //aclanthology.org/2021.naacl-main.410/


32. Radford, A., Kim, J.W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., et al.: Learning transferable visual models from natural language supervision. In: International conference on machine learning. pp. 8748-8763. PMLR (2021)
32. Radford, A., Kim, J.W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., et al.: Learning transferable visual models from natural language supervision. In: International conference on machine learning. pp. 8748-8763. PMLR (2021)


33. Venkateswara, H., Eusebio, J., Chakraborty, S., Panchanathan, S.: Deep hashing network for unsupervised domain adaptation. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 5018-5027 (2017)
33. Venkateswara, H., Eusebio, J., Chakraborty, S., Panchanathan, S.: Deep hashing network for unsupervised domain adaptation. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 5018-5027 (2017)


34. Volpi, R., Namkoong, H., Sener, O., Duchi, J.C., Murino, V., Savarese, S.: Generalizing to unseen domains via adversarial data augmentation. Advances in neural information processing systems 31 (2018)
34. Volpi, R., Namkoong, H., Sener, O., Duchi, J.C., Murino, V., Savarese, S.: Generalizing to unseen domains via adversarial data augmentation. Advances in neural information processing systems 31 (2018)


35. Wang, Q., Lin, Y., Chen, Y., Schmidt, L., Han, B., Zhang, T.: Do clips always generalize better than imagenet models? arXiv preprint arXiv:2403.11497 (2024)
35. Wang, Q., Lin, Y., Chen, Y., Schmidt, L., Han, B., Zhang, T.: Do clips always generalize better than imagenet models? arXiv preprint arXiv:2403.11497 (2024)


36. Zhang, X., Gu, S.S., Matsuo, Y., Iwasawa, Y.: Domain prompt learning for efficiently adapting clip to unseen domains. Transactions of the Japanese Society for Artificial Intelligence 38(6), B-MC2_1 (2023)
36. Zhang, X., Gu, S.S., Matsuo, Y., Iwasawa, Y.: Domain prompt learning for efficiently adapting clip to unseen domains. Transactions of the Japanese Society for Artificial Intelligence 38(6), B-MC2_1 (2023)


37. Zheng, Z., Yue, X., Wang, K., You, Y.: Prompt vision transformer for domain generalization. arXiv preprint arXiv:2208.08914 (2022)
37. Zheng, Z., Yue, X., Wang, K., You, Y.: Prompt vision transformer for domain generalization. arXiv preprint arXiv:2208.08914 (2022)


38. Zhou, K., Liu, Z., Qiao, Y., Xiang, T., Loy, C.C.: Domain generalization: A survey. IEEE Transactions on Pattern Analysis and Machine Intelligence (2022)
38. Zhou, K., Liu, Z., Qiao, Y., Xiang, T., Loy, C.C.: Domain generalization: A survey. IEEE Transactions on Pattern Analysis and Machine Intelligence (2022)


39. Zhou, K., Yang, J., Loy, C.C., Liu, Z.: Conditional prompt learning for vision-language models. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 16816-16825 (2022)
39. Zhou, K., Yang, J., Loy, C.C., Liu, Z.: Conditional prompt learning for vision-language models. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 16816-16825 (2022)


40. Zhou, K., Yang, J., Loy, C.C., Liu, Z.: Learning to prompt for vision-language models. International Journal of Computer Vision 130(9), 2337-2348 (2022)
40. Zhou, K., Yang, J., Loy, C.C., Liu, Z.: Learning to prompt for vision-language models. International Journal of Computer Vision 130(9), 2337-2348 (2022)


This appendix is organized as follows:
本附录组织如下：


- Section A provides the detailed dataset information.
- A 部分提供详细的数据集信息。


- Section B provides the algorithm details.
- B 部分提供算法细节。


- Section C provides the additional training implementation details.
- C 部分提供额外的训练实现细节。


- Section D gives additional experiment results, including additional comparisons of domain-wise results of multi-source domain generalization and single-source domain generalization.
- D 部分给出额外的实验结果，包括对多源域泛化与单源域泛化的领域级结果的额外对比。


## A Dataset Details
## A 数据集详情


PACS [21] is a commonly used small-scaled dataset in the field of domain adaptation and domain generalization. It consists of 4 domains, a total of 9991 images, namely Photo (1,670 images), Art Painting (2,048 images), Cartoon (2,344 images), and Sketch (3,929 images). Each domain contains 7 categories.
PACS [21] 是领域自适应与领域泛化领域常用的小规模数据集。它由 4 个域、共计 9991 张图像组成，分别是 Photo (1,670 张)、Art Painting (2,048 张)、Cartoon (2,344 张) 和 Sketch (3,929 张)。每个域包含 7 个类别。


VLCS [11] is also a small-scaled benchmark dataset, a total of 7,510 images, including 4 domains: Caltech (991 images), LabelMe (1,859 images), Pascal (2,363 images) and Sun (2,297 images). Each domain contains 5 categories.
VLCS [11] 也是一个小规模基准数据集，共 7,510 张图像，包含 4 个域：Caltech (991 张)、LabelMe (1,859 张)、Pascal (2,363 张) 和 Sun (2,297 张)。每个域包含 5 个类别。


Office-Home 33 is a medium-scaled benchmark for domain adaptation and domain generalization. It contains a total of around 15,500 images from 4 distinct domains: Art (2,427 images), Clip Art (4,365 images), Product (4,439 images), and Real World (4,357 images). Each domain contains objects from 65 categories commonly found in office and home environments.
Office-Home 33 是一个中等规模的域适应与域泛化基准。总共约 15,500 张来自 4 个不同域的图像：Art (2,427 张)、Clip Art (4,365 张)、Product (4,439 张) 与 Real World (4,357 张)。每个域包含在办公与家居环境中常见的 65 个类别的对象。


TerraIncognita [3] is a large-scaled benchmark for visual recognition. It contains 243,187 images from 140 camera locations. For DG, a subset is selected that includes 4 domains: Location38 (9,736 images), Location43 (3,970 images), Location46 (5,883 images) and Location100 (4,741 images). Each domain contains animals from 10 categories found in the wild.
TerraIncognita [3] 是一个大规模视觉识别基准。它包含 243,187 张图像，来自 140 个相机位置。对 DG，选择了包含 4 个域的子集：Location38 (9,736 张)、Location43 (3,970 张)、Location46 (5,883 张) 和 Location100 (4,741 张)。每个域包含野外环境中发现的 10 个类别的动物。


DomainNet [30] is a large-scaled benchmark for domain adaptation and domain generalization. It contains a total of around 586,575 images from 6 distinct domains: Clipart (48,129 images), Infograph (51,605 images), Painting (72,266 images), Quickdraw (172,500 images), Real (172,947 images), Sketch (69,128 images). Each domain includes 345 categories of objects. We sample around 20 thousand data from all domains as the subset.
DomainNet [30] 是一个大型的域适应与域泛化基准。总共约 586,575 张图像，来自 6 个域：Clipart (48,129 张)、Infograph (51,605 张)、Painting (72,266 张)、Quickdraw (172,500 张)、Real (172,947 张) 与 Sketch (69,128 张)。每个域包含 345 类对象。我们从所有域中抽取约 2 万数据作为子集。


## B Algorithm Details
## B 算法细节


The overall framework of the pseudo-code of SPG is described in Algorithm 1 and Algorithm 2 Algorithm 1 demonstrates the process of stage I: Domain Prompt Labels Learning, and Algorithm 2 shows the process of stage II: Generative Model Pre-training.
SPG 的伪代码总体框架在算法 1 和算法 2 中描述。算法 1 展示阶段 I：领域提示标签学习的过程，算法 2 展示阶段 II：生成模型预训练的过程。


Algorithm 1 Soft Prompt Generation - Domain Prompt Labels Learning
Algorithm 1 Soft Prompt Generation - Domain Prompt Labels Learning


---



Requirement: pre-defined ${N}_{c}$ class names in the target task
_REQUIRE: 目标任务中预定义 ${N}_{c}$ 类名


Input: images and labels of training samples $\left( {{\mathbf{x}}_{j}^{{d}_{i}},{y}_{j}^{{d}_{i}}}\right)$ ,number of training iterations
<Input:> 训练样本的图像与标签 $\left( {{\mathbf{x}}_{j}^{{d}_{i}},{y}_{j}^{{d}_{i}}}\right)$，训练迭代次数


$L$



Output: ${N}_{d}$ domain prompt labels
<Output:> ${N}_{d}$ 领域提示标签


&nbsp;&nbsp;&nbsp;&nbsp;#learn prompt label on each domain separately
&nbsp;&nbsp;&nbsp;&nbsp;#逐域学习提示标签


&nbsp;&nbsp;&nbsp;&nbsp;for $i = 1,2,\ldots ,{N}_{d}$ do
&nbsp;&nbsp;&nbsp;&nbsp;对于 $i = 1,2,\ldots ,{N}_{d}$ 做


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;#initialize $i$ -th domain prompt label with prompt prefix ${\mathbf{v}}^{p}$ and learnable
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;#		将 $i$ 第 n 域提示标签与前缀 ${\mathbf{v}}^{p}$ 一起初始化并可学习


&nbsp;&nbsp;&nbsp;&nbsp;vector ${\mathbf{v}}^{i}$ .
&nbsp;&nbsp;&nbsp;&nbsp;向量 ${\mathbf{v}}^{i}$ 。


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;${\mathbf{v}}^{{d}_{i}} \leftarrow$ initialize $\left( {{\mathbf{v}}^{p},{\mathbf{v}}^{i}}\right)$
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;${\mathbf{v}}^{{d}_{i}} \leftarrow$ 初始化 $\left( {{\mathbf{v}}^{p},{\mathbf{v}}^{i}}\right)$


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;#$L$ training iterations for learning each domain prompt label
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;#为学习每个域的提示标签进行 $L$ 训练迭代


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;for iteration $= 1,2,\ldots ,L$ do
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;for 迭代 $= 1,2,\ldots ,L$ 做


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;#update learnable vector ${v}^{{d}_{i}}$
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;#更新可学习向量 ${v}^{{d}_{i}}$


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;${\mathbf{v}}^{{d}_{i} * } = \arg \mathop{\min }\limits_{\mathbf{v}}{\mathbb{E}}_{{\mathbf{x}}_{j}^{{d}_{i}},{y}_{j}^{{d}_{i}}}\left\lbrack  {-\log p\left( {{y}_{j}^{{d}_{i}} \mid  {\mathbf{x}}_{j}^{{d}_{i}},{\mathbf{v}}^{{d}_{i}}}\right) }\right\rbrack$



&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Update ${\mathbf{v}}^{{d}_{i}}$ by gradient descent
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;通过梯度下降更新 ${\mathbf{v}}^{{d}_{i}}$


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;end for
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;end for


&nbsp;&nbsp;&nbsp;&nbsp;end for
&nbsp;&nbsp;&nbsp;&nbsp;end for


&nbsp;&nbsp;&nbsp;&nbsp;Store optimal domain prompt label ${\mathbf{v}}^{{d}_{i}}$ for each domain
&nbsp;&nbsp;&nbsp;&nbsp;为每个域存储最优域提示标签 ${\mathbf{v}}^{{d}_{i}}$


---



Algorithm 2 Soft Prompt Generation - Generative Model Pre-training
算法 2 软提示生成 - 生成模型预训练


---



Requirement: A CGAN with a generator $G$ and a discriminator $D$ ,real vector ${\mathbf{v}}_{\text{ real }}$
要求：一个带生成器 $G$ 和判别器 $D$ 的 CGAN，真实向量 ${\mathbf{v}}_{\text{ real }}$


and fake vector ${\mathbf{v}}_{\text{ fake }}$ ,and domain prompt labels ${\mathbf{v}}^{{d}_{i}}$
以及假向量 ${\mathbf{v}}_{\text{ fake }}$，以及域提示标签 ${\mathbf{v}}^{{d}_{i}}$


Input: image embeddings $f\left( \mathbf{x}\right)$ ,number of training iterations $L$
输入：图像嵌入 $f\left( \mathbf{x}\right)$，训练迭代次数 $L$


Output: optimal prompt for each image
输出：每张图片的最优提示


&nbsp;&nbsp;&nbsp;&nbsp;#$L$ training iterations
&nbsp;&nbsp;&nbsp;&nbsp;#$L$ 训练迭代


&nbsp;&nbsp;&nbsp;&nbsp;for iteration $= 1,2,\ldots ,L$ do
&nbsp;&nbsp;&nbsp;&nbsp;对于迭代 $= 1,2,\ldots ,L$ 做


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;#define input noise variable $\mathbf{z}$ and combines these noise variables with image
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;#define 输入噪声变量 $\mathbf{z}$ 并将这些噪声变量与图像


&nbsp;&nbsp;&nbsp;&nbsp;embeddings $f\left( \mathbf{x}\right)$ as input of generator $\mathrm{G}$ and output ${\mathbf{v}}_{g}$
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;embeddings $f\left( \mathbf{x}\right)$ 作为生成器 $\mathrm{G}$ 的输入并输出 ${\mathbf{v}}_{g}$


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;${\mathbf{v}}_{g} = G\left( \text{ input }\right)  \leftarrow$ input $= \operatorname{concat}\left( \left\lbrack  {\mathbf{z},f\left( \mathbf{x}\right) }\right\rbrack  \right)$
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;${\mathbf{v}}_{g} = G\left( \text{ input }\right)  \leftarrow$ 输入 $= \operatorname{concat}\left( \left\lbrack  {\mathbf{z},f\left( \mathbf{x}\right) }\right\rbrack  \right)$


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;#discriminator determines the authenticity of domain prompt labels ${\mathbf{v}}^{{d}_{i}}$ and
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;#判别器确定域提示标签 ${\mathbf{v}}^{{d}_{i}}$ 的真实性与


&nbsp;&nbsp;&nbsp;&nbsp;generated prompt ${\mathbf{v}}_{g}$
&nbsp;&nbsp;&nbsp;&nbsp;生成的提示 ${\mathbf{v}}_{g}$


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;${\mathbf{v}}_{d,\text{ real }} = D\left( {\mathbf{v}}^{{d}_{i}}\right) ,{\mathbf{v}}_{d,\text{ fake }} = D\left( {\mathbf{v}}_{g}\right)$



&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;#compute ${\mathcal{L}}_{\text{ real }}$ with discriminator_real output ${\mathbf{v}}_{d,\text{ real }}$ and pre-defined real
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;#使用判别器 real 输出 ${\mathbf{v}}_{d,\text{ real }}$ 和预定义真实值计算 ${\mathcal{L}}_{\text{ real }}$


&nbsp;&nbsp;&nbsp;&nbsp;vector ${\mathbf{v}}_{\text{ real }}$
&nbsp;&nbsp;&nbsp;&nbsp;向量 ${\mathbf{v}}_{\text{ real }}$


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;${\mathcal{L}}_{\text{ real }} \leftarrow$ mse_loss $\left( {{\mathbf{v}}_{d,\text{ real }},{\mathbf{v}}_{\text{ real }}}\right)$
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;${\mathcal{L}}_{\text{ real }} \leftarrow$ 均方误差损失 $\left( {{\mathbf{v}}_{d,\text{ real }},{\mathbf{v}}_{\text{ real }}}\right)$


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;#compute ${\mathcal{L}}_{\text{ fake }}$ with discriminator_fake output ${\mathbf{v}}_{d,\text{ fake }}$ and pre-defined fake
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;#使用判别器 fake 输出 ${\mathbf{v}}_{d,\text{ fake }}$ 和预定义假的计算 ${\mathcal{L}}_{\text{ fake }}$


&nbsp;&nbsp;&nbsp;&nbsp;vector ${\mathbf{v}}_{\text{ fake }}$
&nbsp;&nbsp;&nbsp;&nbsp;向量 ${\mathbf{v}}_{\text{ fake }}$


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;${\mathcal{L}}_{\text{ fake }} \leftarrow  \operatorname{mse\_ loss}\left( {{\mathbf{v}}_{d,\text{ fake }},{\mathbf{v}}_{\text{ fake }}}\right)$



&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;${\mathcal{L}}_{\text{ discriminator }} \leftarrow  {\mathcal{L}}_{\text{ real }} + {\mathcal{L}}_{\text{ fake }}$



&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Update parameters of $D$ using ${\mathcal{L}}_{\text{ discriminator }}$ by gradient descent
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;使用梯度下降更新 $D$ 的参数，基于 ${\mathcal{L}}_{\text{ discriminator }}$


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;#compute ${\mathcal{L}}_{\text{ generator }}$ with discriminator_fake output ${\mathbf{v}}_{d,\text{ fake }}$ and pre-defined
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;#使用判别器 fake 输出 ${\mathbf{v}}_{d,\text{ fake }}$ 和预定义的计算 ${\mathcal{L}}_{\text{ generator }}$


&nbsp;&nbsp;&nbsp;&nbsp;real vector ${\mathbf{v}}_{\text{ real }}$
&nbsp;&nbsp;&nbsp;&nbsp;真实向量 ${\mathbf{v}}_{\text{ real }}$


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;${\mathcal{L}}_{\text{ generator }} \leftarrow  \operatorname{mse\_ loss}\left( {{\mathbf{v}}_{d,\text{ fake }},{\mathbf{v}}_{\text{ real }}}\right)$



&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Update parameters of $G$ using ${\mathcal{L}}_{\text{ generator }}$ by gradient descent
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;更新参数 $G$，使用 ${\mathcal{L}}_{\text{ generator }}$ 通过梯度下降


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;end for
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;结束 for


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Generate the optimal prompt for each input image
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;为每个输入图像生成最优提示


---



## C Implementation Details
## C 实现细节


For our proposed SPG, in the first phase of the training stage, we employ the text prompt 40 as our prompt design, which is also the prototype for the domain prompt labels and generated prompts. We initialize the context with the phrase "a photo of a" and set the prompt's context length to 4 . In the second phase of the training stage, we train the CGAN model on different domains of various datasets, employing a tailored set of training parameters. Specifically, we set the batch size to around 32 and adjust the initial learning rate between 1e-4 and 2e-3 on different datasets, We employ a cosine learning rate scheduler and conduct training for 70 to 100 epochs, incorporating a linear learning rate warm-up phase at 1e-5 over the first 4 epochs. For optimization, we use the AdamW optimizer, configuring it with a weight decay of 1e-4 and beta values set to (0.9, 0.999).
对于我们提出的 SPG，在训练阶段的第一阶段，我们采用文本提示 40 作为提示设计，也是领域提示标签和生成提示的原型。我们用短语"a photo of a"初始化上下文，并将提示的上下文长度设置为 4。在训练阶段的第二阶段，我们在不同数据集的不同领域上训练 CGAN 模型，使用一组量身定制的训练参数。具体地，我们将批量大小设为约 32，并在不同数据集上将初始学习率在 1e-4 和 2e-3 之间调整。我们采用余弦学习率调度器，并训练 70 到 100 个时期，在前 4 个时期以 1e-5 进行线性学习率热身。对于优化，我们使用 AdamW 优化器，配置权重衰减为 1e-4，β 值设为 (0.9, 0.999)。


Meanwhile, given the observed instability in CGAN training, we implement the gradient clipping strategy to control the magnitude of gradients within the generator and discriminator networks. Specifically, for the discriminator, we establish norm upper limits for general weights and biases in the range of 5e-2 to 5e-1, while setting those for particular weights and biases at a ceiling of 5. For the generator, we set the norm upper limit for universal weights between 5e-3 and 5e-2, the norm upper limit for universal biases between 5e-8 and 5e-7, and the norm upper limit for special biases between 0.5 and 5 . This strategy aims to enhance training stability by preventing excessive gradient values.
同时，鉴于 CGAN 训练中的不稳定性，我们实现梯度裁剪策略，以控制生成器和判别器网络中梯度的大小。具体地，对于判别器，我们为通用权重和偏置设定范数上限，范围为 5e-2 到 5e-1，而特定权重和偏置的上限为 5。对于生成器，我们将通用权重的范数上限设为 5e-3 到 5e-2，通用偏置的范数上限设为 5e-8 到 5e-7，特殊偏置的范数上限设为 0.5 到 5。该策略旨在通过防止梯度值过大来提升训练稳定性。


## D Supplement Experiments
## D 补充实验


The supplement experiments mainly demonstrate the domain-wise results of multi-source DG and single-source DG.
补充实验主要展示多源 DG 和单源 DG 的领域级结果。


### D.1 Multi-source DG Comparisons
### D.1 多源 DG 比较


Table 7 11 demonstrate the per-domain multi-source DG top-1 classification accuracy on PACS, VLCS, OfficeHome, TerraIncognita, and DomainNet, respectively. We observe that although SPG may not achieve the best performance in every individual domain, it consistently reaches state-of-the-art levels across five datasets under two different backbones on average. For some tasks, such as the target domain is Sketch in PACS and Sun in VLCS, SPG outperforms the previous SOTA method with a large margin of 4.3% and 3.6%, respectively.
表 7 11 分别展示在 PACS、VLCS、OfficeHome、TerraIncognita 和 DomainNet 的每个领域的多源 DG 的 top-1 分类准确率。我们观察到，尽管 SPG 可能不是每个单独领域的最佳，但在五个数据集上在两种不同骨干网络下的平均水平均达到或接近最先进水平。对于某些任务，例如 PACS 的目标域为 Sketch 和 VLCS 的 Sun，SPG 分别以 4.3%、3.6% 的幅度领先于之前的 SOTA 方法。


### D.2 Single-source DG Comparisons
### D.2 单源 DG 比较


Table 12 16 demonstrate the per-domain single-source DG top-1 classification accuracy on PACS, VLCS, OfficeHome, TerraIncognita, and DomainNet, respectively. Relying on a single domain for generalization can significantly degrade performance. While our approach may not be optimal in certain cases, it still achieves state-of-the-art performance on average.
表 12 16 展示在 PACS、VLCS、OfficeHome、TerraIncognita 和 DomainNet 的每个领域的单源 DG 的 top-1 分类准确率。仅凭单一领域进行泛化会显著降低性能。尽管我们的方法在某些情况下可能并非最优，但平均而言仍然达到了最先进的性能。


Table 7: Comparisons with SOTA methods on PACS for multi-source DG in terms of mean leave-one-domain-out performance with ResNet50 and ViT-B/16 as the backbone (B.). Average accuracies are reported from three trials. Bold denotes the best scores.
Table 7: 与 SOTA 方法在 PACS 上的多源 DG 的比较，基于 ResNet50 和 ViT-B/16 作为骨干网络（B.）。平均准确率来自三次试验。粗体表示最好分数。


<table><tr><td>Method</td><td>B.</td><td>Art</td><td>Cartoon</td><td>Photo</td><td>Sketch</td><td>Avg</td></tr><tr><td>ZS-CLIP 32</td><td rowspan="7">RN50</td><td>90.9</td><td>93.3</td><td>99.2</td><td>79.5</td><td>90.7</td></tr><tr><td>Lin. Prob. 32</td><td>90.8</td><td>92.7</td><td>99.1</td><td>79.8</td><td>90.6</td></tr><tr><td>CoOp 40</td><td>92.0</td><td>93.8</td><td>98.6</td><td>80.7</td><td>91.3</td></tr><tr><td>CoCoOp 39</td><td>93.1</td><td>94.3</td><td>99.3</td><td>80.8</td><td>91.9</td></tr><tr><td>DPL 36</td><td>93.6</td><td>93.8</td><td>99.0</td><td>80.7</td><td>91.8</td></tr><tr><td>VP 1</td><td>90.6</td><td>92.7</td><td>99.3</td><td>78.0</td><td>90.2</td></tr><tr><td>SPG (ours)</td><td>92.8</td><td>93.8</td><td>99.5</td><td>85.1</td><td>92.8</td></tr><tr><td>ZS-CLIP 32</td><td rowspan="9">ViT-B/16</td><td>97.2</td><td>99.1</td><td>99.9</td><td>88.2</td><td>96.1</td></tr><tr><td>Lin. Prob. 32</td><td>96.2</td><td>94.7</td><td>98.7</td><td>90.1</td><td>94.9</td></tr><tr><td>CoOp 40</td><td>97.7</td><td>98.4</td><td>99.6</td><td>90.0</td><td>96.4</td></tr><tr><td>CoCoOp 39</td><td>97.7</td><td>99.0</td><td>99.8</td><td>90.4</td><td>96.7</td></tr><tr><td>DPL 36</td><td>97.8</td><td>98.5</td><td>99.9</td><td>89.5</td><td>96.4</td></tr><tr><td>VP [1]</td><td>96.9</td><td>98.9</td><td>99.9</td><td>87.3</td><td>95.8</td></tr><tr><td>VPT 17</td><td>97.9</td><td>98.9</td><td>99.9</td><td>91.0</td><td>96.9</td></tr><tr><td>MaPLe 18</td><td>97.9</td><td>98.7</td><td>99.7</td><td>89.8</td><td>96.5</td></tr><tr><td>SPG (ours)</td><td>97.7</td><td>99.0</td><td>99.9</td><td>91.3</td><td>97.0</td></tr></table>
<table><tbody><tr><td>方法</td><td>B.</td><td>艺术</td><td>漫画</td><td>照片</td><td>草图</td><td>平均</td></tr><tr><td>ZS-CLIP 32</td><td rowspan="7">RN50</td><td>90.9</td><td>93.3</td><td>99.2</td><td>79.5</td><td>90.7</td></tr><tr><td>线性概率 32</td><td>90.8</td><td>92.7</td><td>99.1</td><td>79.8</td><td>90.6</td></tr><tr><td>CoOp 40</td><td>92.0</td><td>93.8</td><td>98.6</td><td>80.7</td><td>91.3</td></tr><tr><td>CoCoOp 39</td><td>93.1</td><td>94.3</td><td>99.3</td><td>80.8</td><td>91.9</td></tr><tr><td>DPL 36</td><td>93.6</td><td>93.8</td><td>99.0</td><td>80.7</td><td>91.8</td></tr><tr><td>VP 1</td><td>90.6</td><td>92.7</td><td>99.3</td><td>78.0</td><td>90.2</td></tr><tr><td>SPG (我们)</td><td>92.8</td><td>93.8</td><td>99.5</td><td>85.1</td><td>92.8</td></tr><tr><td>ZS-CLIP 32</td><td rowspan="9">ViT-B/16</td><td>97.2</td><td>99.1</td><td>99.9</td><td>88.2</td><td>96.1</td></tr><tr><td>线性概率 32</td><td>96.2</td><td>94.7</td><td>98.7</td><td>90.1</td><td>94.9</td></tr><tr><td>CoOp 40</td><td>97.7</td><td>98.4</td><td>99.6</td><td>90.0</td><td>96.4</td></tr><tr><td>CoCoOp 39</td><td>97.7</td><td>99.0</td><td>99.8</td><td>90.4</td><td>96.7</td></tr><tr><td>DPL 36</td><td>97.8</td><td>98.5</td><td>99.9</td><td>89.5</td><td>96.4</td></tr><tr><td>VP [1]</td><td>96.9</td><td>98.9</td><td>99.9</td><td>87.3</td><td>95.8</td></tr><tr><td>VPT 17</td><td>97.9</td><td>98.9</td><td>99.9</td><td>91.0</td><td>96.9</td></tr><tr><td>MaPLe 18</td><td>97.9</td><td>98.7</td><td>99.7</td><td>89.8</td><td>96.5</td></tr><tr><td>SPG (我们)</td><td>97.7</td><td>99.0</td><td>99.9</td><td>91.3</td><td>97.0</td></tr></tbody></table>


Table 8: Comparisons with SOTA methods on VLCS for multi-source DG in terms of mean leave-one-domain-out performance with ResNet50 and ViT-B/16 as the backbone (B.). Average accuracies are reported from three trials. Bold denotes the best scores.
表 8：在 VLCS 上针对多源域泛化的与 SOTA 方法的比较，基于 ResNet50 与 ViT-B/16 为骨干网络时的平均“留一域”的性能（B.）。平均准确率来自三次试验。粗体表示最佳分数。


<table><tr><td>Method</td><td>B.</td><td>Caltech</td><td>LableMe</td><td>Pascal</td><td>Sun</td><td>Avg</td></tr><tr><td>ZS-CLIP 32</td><td rowspan="7">RN50</td><td>99.4</td><td>64.9</td><td>84.1</td><td>71.6</td><td>80.0</td></tr><tr><td>Lin. Prob. 32</td><td>99.3</td><td>61.1</td><td>81.8</td><td>76.9</td><td>79.8</td></tr><tr><td>CoOp 40</td><td>99.7</td><td>64.0</td><td>84.7</td><td>77.3</td><td>81.4</td></tr><tr><td>CoCoOp 39</td><td>99.7</td><td>63.7</td><td>84.8</td><td>78.8</td><td>81.8</td></tr><tr><td>DPL 36</td><td>99.8</td><td>62.5</td><td>84.5</td><td>76.3</td><td>80.8</td></tr><tr><td>VP 1</td><td>99.6</td><td>66.3</td><td>84.6</td><td>71.5</td><td>80.5</td></tr><tr><td>SPG (ours)</td><td>99.5</td><td>68.7</td><td>85.4</td><td>82.4</td><td>84.0</td></tr><tr><td>zero-shot CLIP</td><td></td><td>99.9</td><td>68.6</td><td>85.9</td><td>74.8</td><td>82.3</td></tr><tr><td>Lin. Prob. 32</td><td></td><td>95.9</td><td>63.7</td><td>76.3</td><td>74.2</td><td>77.5</td></tr><tr><td>CoOp [40]</td><td></td><td>99.6</td><td>61.4</td><td>84.6</td><td>77.5</td><td>80.8</td></tr><tr><td>CoCoOp [39]</td><td></td><td>99.9</td><td>59.7</td><td>85.9</td><td>75.5</td><td>80.3</td></tr><tr><td>DPL 36</td><td>ViT-B/16</td><td>99.8</td><td>61.5</td><td>84.6</td><td>77.8</td><td>80.9</td></tr><tr><td>VP [1]</td><td></td><td>100.0</td><td>68.5</td><td>86.2</td><td>73.9</td><td>82.2</td></tr><tr><td>VPT [17]</td><td></td><td>99.9</td><td>64.8</td><td>85.2</td><td>78.2</td><td>82.0</td></tr><tr><td>MaPLe 18</td><td></td><td>98.3</td><td>64.8</td><td>85.1</td><td>80.6</td><td>82.2</td></tr><tr><td>SPG (ours)</td><td></td><td>99.7</td><td>64.7</td><td>84.4</td><td>80.7</td><td>82.4</td></tr></table>
<table><tbody><tr><td>方法</td><td>B.</td><td>加州理工学院</td><td>LableMe</td><td>帕斯卡</td><td>阳光</td><td>平均</td></tr><tr><td>ZS-CLIP 32</td><td rowspan="7">RN50</td><td>99.4</td><td>64.9</td><td>84.1</td><td>71.6</td><td>80.0</td></tr><tr><td>线性概率 32</td><td>99.3</td><td>61.1</td><td>81.8</td><td>76.9</td><td>79.8</td></tr><tr><td>CoOp 40</td><td>99.7</td><td>64.0</td><td>84.7</td><td>77.3</td><td>81.4</td></tr><tr><td>CoCoOp 39</td><td>99.7</td><td>63.7</td><td>84.8</td><td>78.8</td><td>81.8</td></tr><tr><td>DPL 36</td><td>99.8</td><td>62.5</td><td>84.5</td><td>76.3</td><td>80.8</td></tr><tr><td>VP 1</td><td>99.6</td><td>66.3</td><td>84.6</td><td>71.5</td><td>80.5</td></tr><tr><td>SPG（我们的）</td><td>99.5</td><td>68.7</td><td>85.4</td><td>82.4</td><td>84.0</td></tr><tr><td>零-shot CLIP</td><td></td><td>99.9</td><td>68.6</td><td>85.9</td><td>74.8</td><td>82.3</td></tr><tr><td>线性概率 32</td><td></td><td>95.9</td><td>63.7</td><td>76.3</td><td>74.2</td><td>77.5</td></tr><tr><td>CoOp [40]</td><td></td><td>99.6</td><td>61.4</td><td>84.6</td><td>77.5</td><td>80.8</td></tr><tr><td>CoCoOp [39]</td><td></td><td>99.9</td><td>59.7</td><td>85.9</td><td>75.5</td><td>80.3</td></tr><tr><td>DPL 36</td><td>ViT-B/16</td><td>99.8</td><td>61.5</td><td>84.6</td><td>77.8</td><td>80.9</td></tr><tr><td>VP [1]</td><td></td><td>100.0</td><td>68.5</td><td>86.2</td><td>73.9</td><td>82.2</td></tr><tr><td>VPT [17]</td><td></td><td>99.9</td><td>64.8</td><td>85.2</td><td>78.2</td><td>82.0</td></tr><tr><td>MaPLe 18</td><td></td><td>98.3</td><td>64.8</td><td>85.1</td><td>80.6</td><td>82.2</td></tr><tr><td>SPG（我们的）</td><td></td><td>99.7</td><td>64.7</td><td>84.4</td><td>80.7</td><td>82.4</td></tr></tbody></table>


Table 9: Comparisons with SOTA methods on OfficeHome for multi-source DG in terms of mean leave-one-domain-out performance with ResNet50 and ViT-B/16 as the backbone (B.). Average accuracies are reported from three trials. Bold denotes the best scores.
表9：在 OfficeHome 上使用 ResNet50 和 ViT-B/16 作为骨干网的多源 DG 的平均逐一放弃域性能的与 SOTA 方法的比较（B.）。平均准确率来自三次试验。粗体表示最高分。


<table><tr><td>Method</td><td>B.</td><td>Art</td><td>Clipart</td><td>Product</td><td>Real</td><td>Avg</td></tr><tr><td>ZS-CLIP 32</td><td rowspan="7">RN50</td><td>69.0</td><td>53.5</td><td>80.1</td><td>80.5</td><td>70.8</td></tr><tr><td>Lin. Prob. 32</td><td>62.0</td><td>49.0</td><td>73.6</td><td>77.4</td><td>65.5</td></tr><tr><td>CoOp [40]</td><td>71.3</td><td>56.1</td><td>83.2</td><td>83.2</td><td>73.5</td></tr><tr><td>CoCoOp 39</td><td>70.3</td><td>56.7</td><td>83.4</td><td>83.3</td><td>73.4</td></tr><tr><td>DPL 36</td><td>71.5</td><td>56.2</td><td>83.5</td><td>83.1</td><td>73.6</td></tr><tr><td>VP 1</td><td>67.7</td><td>52.5</td><td>80.0</td><td>80.4</td><td>70.2</td></tr><tr><td>SPG (ours)</td><td>71.3</td><td>55.6</td><td>84.8</td><td>83.4</td><td>73.8</td></tr><tr><td>ZS-CLIP 32</td><td></td><td>80.1</td><td>70.0</td><td>88.2</td><td>89.0</td><td>81.8</td></tr><tr><td>Lin. Prob. 32</td><td></td><td>73.5</td><td>69.9</td><td>87.4</td><td>86.4</td><td>79.3</td></tr><tr><td>CoOp 40</td><td></td><td>81.2</td><td>72.0</td><td>89.7</td><td>89.2</td><td>83.0</td></tr><tr><td>CoCoOp 39</td><td></td><td>81.8</td><td>71.7</td><td>90.3</td><td>89.7</td><td>83.4</td></tr><tr><td>DPL 36</td><td>ViT-B/16</td><td>81.0</td><td>71.2</td><td>90.0</td><td>89.6</td><td>83.0</td></tr><tr><td>VP [1]</td><td></td><td>79.8</td><td>69.1</td><td>87.4</td><td>88.6</td><td>81.2</td></tr><tr><td>VPT [7]</td><td></td><td>80.9</td><td>72.5</td><td>89.0</td><td>90.4</td><td>83.2</td></tr><tr><td>MaPLe 18</td><td></td><td>81.6</td><td>72.6</td><td>90.2</td><td>89.0</td><td>83.4</td></tr><tr><td>SPG (ours)</td><td></td><td>81.6</td><td>72.7</td><td>90.2</td><td>89.9</td><td>83.6</td></tr></table>
<table><tbody><tr><td>方法</td><td>B.</td><td>艺术</td><td>剪贴画</td><td>产品</td><td>真实</td><td>平均</td></tr><tr><td>ZS-CLIP 32</td><td rowspan="7">RN50</td><td>69.0</td><td>53.5</td><td>80.1</td><td>80.5</td><td>70.8</td></tr><tr><td>线性概率 32</td><td>62.0</td><td>49.0</td><td>73.6</td><td>77.4</td><td>65.5</td></tr><tr><td>CoOp [40]</td><td>71.3</td><td>56.1</td><td>83.2</td><td>83.2</td><td>73.5</td></tr><tr><td>CoCoOp 39</td><td>70.3</td><td>56.7</td><td>83.4</td><td>83.3</td><td>73.4</td></tr><tr><td>DPL 36</td><td>71.5</td><td>56.2</td><td>83.5</td><td>83.1</td><td>73.6</td></tr><tr><td>VP 1</td><td>67.7</td><td>52.5</td><td>80.0</td><td>80.4</td><td>70.2</td></tr><tr><td>SPG（我们的）</td><td>71.3</td><td>55.6</td><td>84.8</td><td>83.4</td><td>73.8</td></tr><tr><td>ZS-CLIP 32</td><td></td><td>80.1</td><td>70.0</td><td>88.2</td><td>89.0</td><td>81.8</td></tr><tr><td>线性概率 32</td><td></td><td>73.5</td><td>69.9</td><td>87.4</td><td>86.4</td><td>79.3</td></tr><tr><td>CoOp 40</td><td></td><td>81.2</td><td>72.0</td><td>89.7</td><td>89.2</td><td>83.0</td></tr><tr><td>CoCoOp 39</td><td></td><td>81.8</td><td>71.7</td><td>90.3</td><td>89.7</td><td>83.4</td></tr><tr><td>DPL 36</td><td>ViT-B/16</td><td>81.0</td><td>71.2</td><td>90.0</td><td>89.6</td><td>83.0</td></tr><tr><td>VP [1]</td><td></td><td>79.8</td><td>69.1</td><td>87.4</td><td>88.6</td><td>81.2</td></tr><tr><td>VPT [7]</td><td></td><td>80.9</td><td>72.5</td><td>89.0</td><td>90.4</td><td>83.2</td></tr><tr><td>MaPLe 18</td><td></td><td>81.6</td><td>72.6</td><td>90.2</td><td>89.0</td><td>83.4</td></tr><tr><td>SPG（我们的）</td><td></td><td>81.6</td><td>72.7</td><td>90.2</td><td>89.9</td><td>83.6</td></tr></tbody></table>


Table 10: Comparisons with SOTA methods on TerraIncognita for multi-source DG in terms of mean leave-one-domain-out performance with ResNet50 and ViT-B/16 as the backbone (B.). Average accuracies are reported from three trials. Bold denotes the best scores.
表10：在 TerraIncognita 上与 SOTA 方法的多源 DG 对比，基于 ResNet50 与 ViT-B/16 作为骨干网络（B.），以逐域留一法的平均性能表示。平均准确率来自三次试验。粗体表示最佳分数。


<table><tr><td>Method</td><td>B.</td><td>Location38</td><td>Location43</td><td>Location46</td><td>Location100</td><td>Avg</td></tr><tr><td>ZS-CLIP 32</td><td rowspan="7">RN50</td><td>28.4</td><td>32.8</td><td>24.0</td><td>10.1</td><td>23.8</td></tr><tr><td>Lin. Prob. 32</td><td>33.0</td><td>42.7</td><td>31.9</td><td>24.4</td><td>33.0</td></tr><tr><td>CoOp [40]</td><td>25.6</td><td>43.5</td><td>34.5</td><td>29.2</td><td>33.2</td></tr><tr><td>CoCoOp 39</td><td>35.9</td><td>42.1</td><td>32.5</td><td>25.8</td><td>34.1</td></tr><tr><td>DPL 36</td><td>36.0</td><td>41.1</td><td>32.9</td><td>27.6</td><td>34.4</td></tr><tr><td>VP 1</td><td>28.8</td><td>34.0</td><td>26.8</td><td>12.6</td><td>25.6</td></tr><tr><td>SPG (ours)</td><td>42.1</td><td>38.9</td><td>32.1</td><td>36.8</td><td>37.5</td></tr><tr><td>ZS-CLIP 32</td><td></td><td>20.5</td><td>32.8</td><td>29.6</td><td>52.4</td><td>33.8</td></tr><tr><td>Lin. Prob. 32</td><td></td><td>48.0</td><td>50.5</td><td>43.8</td><td>44.0</td><td>46.6</td></tr><tr><td>CoOp [40]</td><td></td><td>53.3</td><td>47.4</td><td>41.1</td><td>45.5</td><td>46.8</td></tr><tr><td>CoCoOp [39]</td><td></td><td>51.6</td><td>46.9</td><td>39.3</td><td>43.2</td><td>45.3</td></tr><tr><td>DPL 36</td><td></td><td>54.3</td><td>49.0</td><td>41.6</td><td>41.6</td><td>46.6</td></tr><tr><td>VP [1]</td><td>ViT-B/16</td><td>20.2</td><td>34.3</td><td>32.8</td><td>52.3</td><td>34.9</td></tr><tr><td>VPT 17</td><td></td><td>46.8</td><td>52.8</td><td>41.8</td><td>45.5</td><td>46.7</td></tr><tr><td>MaPLe 18</td><td></td><td>52.4</td><td>53.0</td><td>43.1</td><td>52.4</td><td>50.2</td></tr><tr><td>SPG (ours)</td><td></td><td>51.0</td><td>49.2</td><td>50.7</td><td>49.8</td><td>50.2</td></tr></table>
<table><tbody><tr><td>方法</td><td>B.</td><td>位置38</td><td>位置43</td><td>位置46</td><td>位置100</td><td>平均</td></tr><tr><td>ZS-CLIP 32</td><td rowspan="7">RN50</td><td>28.4</td><td>32.8</td><td>24.0</td><td>10.1</td><td>23.8</td></tr><tr><td>线性概率 32</td><td>33.0</td><td>42.7</td><td>31.9</td><td>24.4</td><td>33.0</td></tr><tr><td>CoOp [40]</td><td>25.6</td><td>43.5</td><td>34.5</td><td>29.2</td><td>33.2</td></tr><tr><td>CoCoOp 39</td><td>35.9</td><td>42.1</td><td>32.5</td><td>25.8</td><td>34.1</td></tr><tr><td>DPL 36</td><td>36.0</td><td>41.1</td><td>32.9</td><td>27.6</td><td>34.4</td></tr><tr><td>VP 1</td><td>28.8</td><td>34.0</td><td>26.8</td><td>12.6</td><td>25.6</td></tr><tr><td>SPG（我们）</td><td>42.1</td><td>38.9</td><td>32.1</td><td>36.8</td><td>37.5</td></tr><tr><td>ZS-CLIP 32</td><td></td><td>20.5</td><td>32.8</td><td>29.6</td><td>52.4</td><td>33.8</td></tr><tr><td>线性概率 32</td><td></td><td>48.0</td><td>50.5</td><td>43.8</td><td>44.0</td><td>46.6</td></tr><tr><td>CoOp [40]</td><td></td><td>53.3</td><td>47.4</td><td>41.1</td><td>45.5</td><td>46.8</td></tr><tr><td>CoCoOp [39]</td><td></td><td>51.6</td><td>46.9</td><td>39.3</td><td>43.2</td><td>45.3</td></tr><tr><td>DPL 36</td><td></td><td>54.3</td><td>49.0</td><td>41.6</td><td>41.6</td><td>46.6</td></tr><tr><td>VP [1]</td><td>ViT-B/16</td><td>20.2</td><td>34.3</td><td>32.8</td><td>52.3</td><td>34.9</td></tr><tr><td>VPT 17</td><td></td><td>46.8</td><td>52.8</td><td>41.8</td><td>45.5</td><td>46.7</td></tr><tr><td>MaPLe 18</td><td></td><td>52.4</td><td>53.0</td><td>43.1</td><td>52.4</td><td>50.2</td></tr><tr><td>SPG（我们）</td><td></td><td>51.0</td><td>49.2</td><td>50.7</td><td>49.8</td><td>50.2</td></tr></tbody></table>


Table 11: Comparisons with SOTA methods on DomainNet for multi-source DG in terms of mean leave-one-domain-out performance with ResNet50 and ViT-B/16 as the backbone (B.). Average accuracies are reported from three trials. Bold denotes the best scores.
表格 11：在 DomainNet 上对比 SOTA 方法的多源 DG，以 ResNet50 和 ViT-B/16 作为骨干网络（B.），以逐一剔除域的平均性能衡量。平均准确率来自三次试验。粗体表示最佳分数。


<table><tr><td>Method</td><td>B.</td><td>Clipart</td><td>Infograph</td><td>Painting</td><td>Quickdraw</td><td>Real</td><td>Sketch</td><td>Avg</td></tr><tr><td>ZS-CLIP 32</td><td rowspan="7">RN50</td><td>52.7</td><td>40.5</td><td>53.2</td><td>5.7</td><td>77.1</td><td>49.3</td><td>46.4</td></tr><tr><td>Lin. Prob. 32</td><td>34.6</td><td>24.7</td><td>35.3</td><td>4.1</td><td>28.2</td><td>35.9</td><td>27.1</td></tr><tr><td>CoOp 40</td><td>57.0</td><td>43.9</td><td>58.1</td><td>7.8</td><td>78.8</td><td>52.6</td><td>49.7</td></tr><tr><td>CoCoOp [39]</td><td>57.0</td><td>44.0</td><td>58.3</td><td>7.8</td><td>78.9</td><td>52.0</td><td>49.7</td></tr><tr><td>DPL 36</td><td>56.7</td><td>43.9</td><td>57.9</td><td>7.9</td><td>78.2</td><td>53.0</td><td>49.6</td></tr><tr><td>VP 1</td><td>52.4</td><td>40.3</td><td>52.7</td><td>5.3</td><td>76.8</td><td>47.1</td><td>45.8</td></tr><tr><td>SPG (ours)</td><td>57.3</td><td>41.7</td><td>58.3</td><td>7.9</td><td>80.0</td><td>55.5</td><td>50.1</td></tr><tr><td>ZS-CLIP 32</td><td rowspan="9">ViT-B/16</td><td>70.2</td><td>46.3</td><td>65.0</td><td>13.0</td><td>83.0</td><td>62.0</td><td>56.6</td></tr><tr><td>Lin. Prob. 32</td><td>62.9</td><td>35.4</td><td>56.8</td><td>11.3</td><td>65.8</td><td>56.7</td><td>48.2</td></tr><tr><td>CoOp [40]</td><td>72.7</td><td>50.2</td><td>68.5</td><td>15.6</td><td>84.2</td><td>65.9</td><td>59.5</td></tr><tr><td>CoCoOp 39</td><td>72.1</td><td>50.4</td><td>67.9</td><td>15.8</td><td>84.4</td><td>65.5</td><td>59.4</td></tr><tr><td>DPL 36</td><td>72.5</td><td>50.4</td><td>68.3</td><td>15.8</td><td>83.9</td><td>66.0</td><td>59.5</td></tr><tr><td>VP 1</td><td>70.1</td><td>45.5</td><td>64.6</td><td>14.1</td><td>82.7</td><td>62.0</td><td>56.5</td></tr><tr><td>VPT 17</td><td>71.0</td><td>48.5</td><td>66.2</td><td>16.3</td><td>83.6</td><td>65.2</td><td>58.5</td></tr><tr><td>MaPLe 18</td><td>73.1</td><td>49.9</td><td>67.8</td><td>16.6</td><td>83.5</td><td>65.9</td><td>59.5</td></tr><tr><td>SPG (ours)</td><td>68.7</td><td>50.2</td><td>73.2</td><td>16.6</td><td>83.3</td><td>68.5</td><td>60.1</td></tr></table>
<table><tbody><tr><td>方法</td><td>B.</td><td>剪贴画</td><td>信息图</td><td>绘画</td><td>快速绘制</td><td>真实</td><td>素描</td><td>平均</td></tr><tr><td>ZS-CLIP 32</td><td rowspan="7">RN50</td><td>52.7</td><td>40.5</td><td>53.2</td><td>5.7</td><td>77.1</td><td>49.3</td><td>46.4</td></tr><tr><td>线性概率 32</td><td>34.6</td><td>24.7</td><td>35.3</td><td>4.1</td><td>28.2</td><td>35.9</td><td>27.1</td></tr><tr><td>CoOp 40</td><td>57.0</td><td>43.9</td><td>58.1</td><td>7.8</td><td>78.8</td><td>52.6</td><td>49.7</td></tr><tr><td>CoCoOp [39]</td><td>57.0</td><td>44.0</td><td>58.3</td><td>7.8</td><td>78.9</td><td>52.0</td><td>49.7</td></tr><tr><td>DPL 36</td><td>56.7</td><td>43.9</td><td>57.9</td><td>7.9</td><td>78.2</td><td>53.0</td><td>49.6</td></tr><tr><td>VP 1</td><td>52.4</td><td>40.3</td><td>52.7</td><td>5.3</td><td>76.8</td><td>47.1</td><td>45.8</td></tr><tr><td>SPG（我们）</td><td>57.3</td><td>41.7</td><td>58.3</td><td>7.9</td><td>80.0</td><td>55.5</td><td>50.1</td></tr><tr><td>ZS-CLIP 32</td><td rowspan="9">ViT-B/16</td><td>70.2</td><td>46.3</td><td>65.0</td><td>13.0</td><td>83.0</td><td>62.0</td><td>56.6</td></tr><tr><td>线性概率 32</td><td>62.9</td><td>35.4</td><td>56.8</td><td>11.3</td><td>65.8</td><td>56.7</td><td>48.2</td></tr><tr><td>CoOp [40]</td><td>72.7</td><td>50.2</td><td>68.5</td><td>15.6</td><td>84.2</td><td>65.9</td><td>59.5</td></tr><tr><td>CoCoOp 39</td><td>72.1</td><td>50.4</td><td>67.9</td><td>15.8</td><td>84.4</td><td>65.5</td><td>59.4</td></tr><tr><td>DPL 36</td><td>72.5</td><td>50.4</td><td>68.3</td><td>15.8</td><td>83.9</td><td>66.0</td><td>59.5</td></tr><tr><td>VP 1</td><td>70.1</td><td>45.5</td><td>64.6</td><td>14.1</td><td>82.7</td><td>62.0</td><td>56.5</td></tr><tr><td>VPT 17</td><td>71.0</td><td>48.5</td><td>66.2</td><td>16.3</td><td>83.6</td><td>65.2</td><td>58.5</td></tr><tr><td>MaPLe 18</td><td>73.1</td><td>49.9</td><td>67.8</td><td>16.6</td><td>83.5</td><td>65.9</td><td>59.5</td></tr><tr><td>SPG（我们）</td><td>68.7</td><td>50.2</td><td>73.2</td><td>16.6</td><td>83.3</td><td>68.5</td><td>60.1</td></tr></tbody></table>


Table 12: Comparisons with CLIP-base fine-tuning methods on PACS for single-source DG in terms of leave-all-but-one-domain-out performance with ResNet50 as the backbone. Bold denotes the best scores.
表 12：在 PACS 上对单源 DG 的 leave-all-but-one-domain-out 演练下，使用 ResNet50 作为骨干的 CLIP-base 微调方法比较。粗体表示最佳分数。


<table><tr><td>Method</td><td>Art</td><td>Cartoon</td><td>Photo</td><td>Sketch</td><td>Avg</td></tr><tr><td>Lin. Prob. 32</td><td>79.2</td><td>82.2</td><td>76.7</td><td>71.2</td><td>77.3</td></tr><tr><td>CoOp [40]</td><td>91.4</td><td>84.5</td><td>82.5</td><td>85.4</td><td>86.0</td></tr><tr><td>CoCoOp [39]</td><td>91.1</td><td>85.7</td><td>88.2</td><td>87.5</td><td>88.1</td></tr><tr><td>DPL 36</td><td>90.4</td><td>83.7</td><td>84.5</td><td>88.5</td><td>86.8</td></tr><tr><td>SPG (ours)</td><td>90.5</td><td>87.4</td><td>88.4</td><td>88.7</td><td>88.8</td></tr></table>
<table><tbody><tr><td>方法</td><td>艺术</td><td>漫画</td><td>照片</td><td>素描</td><td>平均</td></tr><tr><td>线性概率 32</td><td>79.2</td><td>82.2</td><td>76.7</td><td>71.2</td><td>77.3</td></tr><tr><td>联合 [40]</td><td>91.4</td><td>84.5</td><td>82.5</td><td>85.4</td><td>86.0</td></tr><tr><td>联动 [39]</td><td>91.1</td><td>85.7</td><td>88.2</td><td>87.5</td><td>88.1</td></tr><tr><td>DPL 36</td><td>90.4</td><td>83.7</td><td>84.5</td><td>88.5</td><td>86.8</td></tr><tr><td>SPG（我们的方法）</td><td>90.5</td><td>87.4</td><td>88.4</td><td>88.7</td><td>88.8</td></tr></tbody></table>


Table 13: Comparisons with CLIP-base fine-tuning methods on VLCS for single-source DG in terms of leave-all-but-one-domain-out performance with ResNet50 as the backbone. Bold denotes the best scores.
表 13：在以 ResNet50 为 backbone 的前提下，对 VLCS 的单源 DG 进行基于 CLIP-base 微调方法的比较，按 leave-all-but-one-domain-out 性能排序。粗体表示最佳分数。


<table><tr><td>Method</td><td>Caltech</td><td>LableMe</td><td>Pascal</td><td>Sun</td><td>Avg</td></tr><tr><td>Lin. Prob. 32</td><td>58.9</td><td>62.6</td><td>76.7</td><td>63.8</td><td>65.5</td></tr><tr><td>CoOp [40]</td><td>74.6</td><td>76.9</td><td>78.7</td><td>71.0</td><td>75.3</td></tr><tr><td>CoCoOp 39</td><td>70.0</td><td>58.7</td><td>80.3</td><td>63.8</td><td>68.2</td></tr><tr><td>DPL 36</td><td>64.8</td><td>77.0</td><td>80.7</td><td>78.3</td><td>75.2</td></tr><tr><td>SPG (ours)</td><td>70.2</td><td>79.3</td><td>76.3</td><td>80.2</td><td>76.5</td></tr></table>
<table><tbody><tr><td>方法</td><td>Caltech</td><td>LableMe</td><td>Pascal</td><td>日</td><td>平均</td></tr><tr><td>线性概率 32</td><td>58.9</td><td>62.6</td><td>76.7</td><td>63.8</td><td>65.5</td></tr><tr><td>CoOp [40]</td><td>74.6</td><td>76.9</td><td>78.7</td><td>71.0</td><td>75.3</td></tr><tr><td>CoCoOp 39</td><td>70.0</td><td>58.7</td><td>80.3</td><td>63.8</td><td>68.2</td></tr><tr><td>DPL 36</td><td>64.8</td><td>77.0</td><td>80.7</td><td>78.3</td><td>75.2</td></tr><tr><td>SPG（我们的方法）</td><td>70.2</td><td>79.3</td><td>76.3</td><td>80.2</td><td>76.5</td></tr></tbody></table>


Table 14: Comparisons with CLIP-base fine-tuning methods on OfficeHome for single-source DG in terms of leave-all-but-one-domain-out performance with ResNet50 as the backbone. Bold denotes the best scores.
表 14：在 OfficeHome 上以 ResNet50 为骨干单一源域 DG 的留出所有除一域以外的性能比较，基于 CLIP-base 微调方法。粗体表示最佳分数。


<table><tr><td>Method</td><td>Art</td><td>Clipart</td><td>Product</td><td>Real</td><td>Avg</td></tr><tr><td>Lin. Prob. 32</td><td>36.8</td><td>42.1</td><td>50.8</td><td>55.8</td><td>46.4</td></tr><tr><td>CoOp [40]</td><td>71.3</td><td>75.6</td><td>65.8</td><td>70.2</td><td>70.7</td></tr><tr><td>CoCoOp 39</td><td>72.0</td><td>75.7</td><td>65.1</td><td>69.7</td><td>70.6</td></tr><tr><td>DPL 36</td><td>72.0</td><td>75.3</td><td>65.7</td><td>70.1</td><td>70.8</td></tr><tr><td>SPG (ours)</td><td>72.3</td><td>74.8</td><td>66.1</td><td>70.2</td><td>70.9</td></tr></table>
<table><tbody><tr><td>方法</td><td>艺术</td><td>剪贴画</td><td>产品</td><td>真实</td><td>平均</td></tr><tr><td>线性概率 32</td><td>36.8</td><td>42.1</td><td>50.8</td><td>55.8</td><td>46.4</td></tr><tr><td>协作 [40]</td><td>71.3</td><td>75.6</td><td>65.8</td><td>70.2</td><td>70.7</td></tr><tr><td>协同协作 39</td><td>72.0</td><td>75.7</td><td>65.1</td><td>69.7</td><td>70.6</td></tr><tr><td>DPL 36</td><td>72.0</td><td>75.3</td><td>65.7</td><td>70.1</td><td>70.8</td></tr><tr><td>SPG（我们）</td><td>72.3</td><td>74.8</td><td>66.1</td><td>70.2</td><td>70.9</td></tr></tbody></table>


Table 15: Comparisons with CLIP-base fine-tuning methods on TerraIncognita for single-source DG in terms of leave-all-but-one-domain-out performance with ResNet50 as the backbone. Bold denotes the best scores.
表 15：在 TerraIncognita 上基于 CLIP-base 微调方法的单源 DG 的离开全部/除一个域外的性能对比，Backbone 为 ResNet50。粗体表示最佳分数。


<table><tr><td>Method</td><td></td><td></td><td></td><td>Location38Location43Location46Location100</td><td>Avg</td></tr><tr><td>Lin. Prob. 32</td><td>29.6</td><td>16.0</td><td>18.3</td><td>29.1</td><td>23.3</td></tr><tr><td>CoOp 40</td><td>23.7</td><td>39.2</td><td>40.8</td><td>19.2</td><td>30.7</td></tr><tr><td>CoCoOp 39</td><td>22.8</td><td>27.5</td><td>34.0</td><td>18.1</td><td>25.6</td></tr><tr><td>DPL 36</td><td>23.2</td><td>32.9</td><td>28.5</td><td>29.1</td><td>28.4</td></tr><tr><td>SPG (ours)</td><td>21.5</td><td>30.3</td><td>40.8</td><td>36.5</td><td>32.3</td></tr></table>
<table><tbody><tr><td>方法</td><td></td><td></td><td></td><td>Location38Location43Location46Location100</td><td>平均</td></tr><tr><td>线性概率 32</td><td>29.6</td><td>16.0</td><td>18.3</td><td>29.1</td><td>23.3</td></tr><tr><td>CoOp 40</td><td>23.7</td><td>39.2</td><td>40.8</td><td>19.2</td><td>30.7</td></tr><tr><td>CoCoOp 39</td><td>22.8</td><td>27.5</td><td>34.0</td><td>18.1</td><td>25.6</td></tr><tr><td>DPL 36</td><td>23.2</td><td>32.9</td><td>28.5</td><td>29.1</td><td>28.4</td></tr><tr><td>SPG (ours)</td><td>21.5</td><td>30.3</td><td>40.8</td><td>36.5</td><td>32.3</td></tr></tbody></table>


Table 16: Comparisons with CLIP-base fine-tuning methods on DomainNet for single-source DG in terms of leave-all-but-one-domain-out performance with ResNet50 as the backbone. Bold denotes the best scores.
表 16：在 DomainNet 上以 ResNet50 为骨干的单源 DG，使用留全保留除一个域以外的性能进行比较，基于 CLIP-base 微调方法。加粗表示最佳分数。


<table><tr><td>Method</td><td>Clipart</td><td>Infograph</td><td>Painting</td><td>Quickdraw</td><td>Real</td><td>Sketch</td><td>Avg</td></tr><tr><td>Lin. Prob. 32</td><td>4.2</td><td>3.3</td><td>6.6</td><td>2.7</td><td>16.4</td><td>4.4</td><td>6.3</td></tr><tr><td>CoOp [40]</td><td>47.2</td><td>46.6</td><td>45.3</td><td>48.0</td><td>35.0</td><td>47.1</td><td>44.9</td></tr><tr><td>CoCoOp [39]</td><td>46.8</td><td>48.0</td><td>46.3</td><td>50.4</td><td>34.4</td><td>47.1</td><td>45.5</td></tr><tr><td>DPL 36</td><td>46.6</td><td>47.5</td><td>45.2</td><td>40.6</td><td>35.6</td><td>47.3</td><td>43.8</td></tr><tr><td>SPG (ours)</td><td>44.0</td><td>45.4</td><td>45.2</td><td>55.6</td><td>36.6</td><td>46.8</td><td>45.6</td></tr></table>
<table><tbody><tr><td>方法</td><td>剪贴画</td><td>信息图</td><td>绘画</td><td>快速绘制</td><td>真实</td><td>草图</td><td>平均</td></tr><tr><td>线性概率 32</td><td>4.2</td><td>3.3</td><td>6.6</td><td>2.7</td><td>16.4</td><td>4.4</td><td>6.3</td></tr><tr><td>CoOp [40]</td><td>47.2</td><td>46.6</td><td>45.3</td><td>48.0</td><td>35.0</td><td>47.1</td><td>44.9</td></tr><tr><td>CoCoOp [39]</td><td>46.8</td><td>48.0</td><td>46.3</td><td>50.4</td><td>34.4</td><td>47.1</td><td>45.5</td></tr><tr><td>DPL 36</td><td>46.6</td><td>47.5</td><td>45.2</td><td>40.6</td><td>35.6</td><td>47.3</td><td>43.8</td></tr><tr><td>SPG(ours)</td><td>44.0</td><td>45.4</td><td>45.2</td><td>55.6</td><td>36.6</td><td>46.8</td><td>45.6</td></tr></tbody></table>


### D.3 Ablation Study
### D.3 消融研究


Component ablation. The domain prompt label and generative model are indispensable components of our SPG method and cannot be directly removed, but can be replaced. For domain prompt label, We replace the text prompt used in our work with VP 1. For backbone, we replace the CGAN 27 with an MLP. Additional domain label and backbone ablations are as follows.
组件消融。域提示标签和生成模型是我们 SPG 方法的不可或缺组成部分，不能直接移除，但可以替换。对于域提示标签，我们将本文中使用的文本提示替换为 VP 1。对于主干网络，我们将 CGAN 27 替换为 MLP。其他域标签与主干网络的消融如下所示。


Table 17: Component ablation for multi-source DG in terms of mean leave-one-domain-out performance with ResNet50 as the backbone. Bold denotes the best scores.
表 17：以 ResNet50 为主干的多源 DG 的平均留一域外性能的组件消融。粗体表示最佳分数。


<table><tr><td>Examples</td><td>PACS</td><td>VLCS</td><td>O.H.</td><td>TerraInc.</td><td>Do.Net</td><td>Avg</td></tr><tr><td>w/ VP</td><td>89.4</td><td>79.8</td><td>67.8</td><td>19.1</td><td>44.7</td><td>60.2</td></tr><tr><td>w/ MLP</td><td>90.4</td><td>79.6</td><td>72.3</td><td>31.7</td><td>47.8</td><td>64.4</td></tr><tr><td>SPG (Ours)</td><td>92.8</td><td>84.0</td><td>73.8</td><td>37.5</td><td>50.1</td><td>67.6</td></tr></table>
<table><tbody><tr><td>示例</td><td>PACS</td><td>VLCS</td><td>O.H.</td><td>TerraInc.</td><td>Do.Net</td><td>平均</td></tr><tr><td>含 VP</td><td>89.4</td><td>79.8</td><td>67.8</td><td>19.1</td><td>44.7</td><td>60.2</td></tr><tr><td>含 MLP</td><td>90.4</td><td>79.6</td><td>72.3</td><td>31.7</td><td>47.8</td><td>64.4</td></tr><tr><td>SPG（我们的）</td><td>92.8</td><td>84.0</td><td>73.8</td><td>37.5</td><td>50.1</td><td>67.6</td></tr></tbody></table>