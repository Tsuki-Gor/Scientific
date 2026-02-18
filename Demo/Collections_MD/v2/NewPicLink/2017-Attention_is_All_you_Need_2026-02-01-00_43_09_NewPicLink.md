# Attention Is All You Need
# 注意力即你所需要的一切


Ashish Vaswani*
Ashish Vaswani*


Google Brain
Google Brain


avaswani@google.com



Noam Shazeer*
Noam Shazeer*


Google Brain
Google Brain


noam@google.com



Niki Parmar*
Niki Parmar*


Google Research
Google Research


nikip@google.com



Jakob Uszkoreit*
Jakob Uszkoreit*


Google Research
Google Research


usz@google.com



Llion Jones*
Llion Jones*


Google Research
Google Research


llion@google.com



Aidan N. Gomez* ${}^{ \dagger  }$
Aidan N. Gomez* ${}^{ \dagger  }$


University of Toronto
多伦多大学


aidan@cs.toronto.edu



Łukasz Kaiser*
Łukasz Kaiser*


Google Brain
Google Brain


lukaszkaiser@google.com



Illia Polosukhin* ‡
Illia Polosukhin* ‡


illia.polosukhin@gmail.com



## Abstract
## 摘要


The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train. Our model achieves 28.4 BLEU on the WMT 2014 English-to-German translation task, improving over the existing best results, including ensembles, by over 2 BLEU. On the WMT 2014 English-to-French translation task, our model establishes a new single-model state-of-the-art BLEU score of 41.0 after training for 3.5 days on eight GPUs, a small fraction of the training costs of the best models from the literature.
主流的序列转导模型基于包含编码器和解码器的复杂循环或卷积神经网络。表现最佳的模型还通过注意力机制连接编码器和解码器。我们提出了一种新的简单网络架构 Transformer，它完全基于注意力机制，彻底抛弃了循环和卷积。在两个机器翻译任务上的实验表明，这些模型在质量上更优，同时更易于并行化，且训练所需时间显著减少。我们的模型在 WMT 2014 英德翻译任务上达到 28.4 BLEU，比包括集成模型在内的现有最佳结果提高了 2 个 BLEU 以上。在 WMT 2014 英法翻译任务中，我们的模型在 8 个 GPU 上训练 3.5 天后，创下了 41.0 的单模型新纪录，而训练成本仅为文献中最佳模型的一小部分。


## 1 Introduction
## 1 引言


Recurrent neural networks, long short-term memory [12] and gated recurrent [7] neural networks in particular, have been firmly established as state of the art approaches in sequence modeling and transduction problems such as language modeling and machine translation [29, 2, 5]. Numerous efforts have since continued to push the boundaries of recurrent language models and encoder-decoder architectures [31, 21, 13].
循环神经网络，特别是长短期记忆 [12] 和门控循环 [7] 神经网络，已在序列建模和转导问题（如语言建模和机器翻译 [29, 2, 5]）中确立了最先进方法的地位。此后，许多努力继续推进循环语言模型和编码器-解码器架构的边界 [31, 21, 13]。


---



*Equal contribution. Listing order is random. Jakob proposed replacing RNNs with self-attention and started the effort to evaluate this idea. Ashish, with Illia, designed and implemented the first Transformer models and has been crucially involved in every aspect of this work. Noam proposed scaled dot-product attention, multi-head attention and the parameter-free position representation and became the other person involved in nearly every detail. Niki designed, implemented, tuned and evaluated countless model variants in our original codebase and tensor2tensor. Llion also experimented with novel model variants, was responsible for our initial codebase, and efficient inference and visualizations. Lukasz and Aidan spent countless long days designing various parts of and implementing tensor2tensor, replacing our earlier codebase, greatly improving results and massively accelerating our research.
*同等贡献。排名不分先后。Jakob 提出用自注意力取代 RNN，并开始评估这一构想。Ashish 与 Illia 设计并实现了第一个 Transformer 模型，并深度参与了这项工作的方方面面。Noam 提出了缩放点积注意力、多头注意力和无参数位置表示，并成为参与几乎所有细节的另一个人。Niki 在我们原始代码库和 tensor2tensor 中设计、实现、调优并评估了无数模型变体。Llion 也尝试了新型模型变体，负责我们最初的代码库，以及高效的推理和可视化。Lukasz 和 Aidan 花费了大量时间设计 tensor2tensor 的各个部分并进行实现，取代了我们早期的代码库，极大地改善了结果并大幅加速了我们的研究。


${}^{ \dagger  }$ Work performed while at Google Brain.
${}^{ \dagger  }$ 在 Google Brain 工作时完成。


*Work performed while at Google Research.
*在 Google Research 工作时完成。


---



Recurrent models typically factor computation along the symbol positions of the input and output sequences. Aligning the positions to steps in computation time, they generate a sequence of hidden states ${h}_{t}$ ,as a function of the previous hidden state ${h}_{t - 1}$ and the input for position $t$ . This inherently sequential nature precludes parallelization within training examples, which becomes critical at longer sequence lengths, as memory constraints limit batching across examples. Recent work has achieved significant improvements in computational efficiency through factorization tricks [18] and conditional computation [26], while also improving model performance in case of the latter. The fundamental constraint of sequential computation, however, remains.
循环模型通常沿输入和输出序列的符号位置进行分解计算。通过将位置与计算时间步对齐，它们生成一系列隐藏状态 ${h}_{t}$，作为前一隐藏状态 ${h}_{t - 1}$ 和位置 $t$ 输入的函数。这种固有的顺序特性阻碍了训练样本内部的并行化，这在序列长度较长时变得至关重要，因为内存限制了跨样本的批处理。最近的研究通过分解技巧 [18] 和条件计算 [26] 在计算效率上取得了显著提升，后者同时也提高了模型性能。然而，顺序计算的基本限制仍然存在。


Attention mechanisms have become an integral part of compelling sequence modeling and transduction models in various tasks, allowing modeling of dependencies without regard to their distance in the input or output sequences [2, 16]. In all but a few cases [22], however, such attention mechanisms are used in conjunction with a recurrent network.
注意力机制已成为各种任务中极具吸引力的序列建模和转导模型不可或缺的一部分，它允许对依赖关系进行建模，而不考虑它们在输入或输出序列中的距离 [2, 16]。然而，除少数情况 [22] 外，此类注意力机制均与循环网络结合使用。


In this work we propose the Transformer, a model architecture eschewing recurrence and instead relying entirely on an attention mechanism to draw global dependencies between input and output. The Transformer allows for significantly more parallelization and can reach a new state of the art in translation quality after being trained for as little as twelve hours on eight P100 GPUs.
在这项工作中，我们提出了 Transformer，这是一种避开循环，转而完全依赖注意力机制来绘制输入和输出之间全局依赖关系的模型架构。Transformer 允许显著更多的并行化，并且在 8 个 P100 GPU 上仅训练 12 小时后即可达到翻译质量的新高度。


## 2 Background
## 2 背景


The goal of reducing sequential computation also forms the foundation of the Extended Neural GPU [20], ByteNet [15] and ConvS2S [8], all of which use convolutional neural networks as basic building block, computing hidden representations in parallel for all input and output positions. In these models, the number of operations required to relate signals from two arbitrary input or output positions grows in the distance between positions, linearly for ConvS2S and logarithmically for ByteNet. This makes it more difficult to learn dependencies between distant positions [11]. In the Transformer this is reduced to a constant number of operations, albeit at the cost of reduced effective resolution due to averaging attention-weighted positions, an effect we counteract with Multi-Head Attention as described in section 3.2
减少顺序计算的目标也构成了 Extended Neural GPU [20]、ByteNet [15] 和 ConvS2S [8] 的基础，这些模型都使用卷积神经网络作为基本构建模块，并行计算所有输入和输出位置的隐藏表示。在这些模型中，关联两个任意输入或输出位置信号所需的计算量随位置间距离而增长，ConvS2S 呈线性增长，ByteNet 呈对数增长。这使得学习长距离位置之间的依赖关系变得更加困难 [11]。在 Transformer 中，这被减少到恒定次数的操作，尽管代价是由于平均注意力加权位置而导致有效分辨率降低，我们通过 3.2 节所述的多头注意力来抵消这一影响。


Self-attention, sometimes called intra-attention is an attention mechanism relating different positions of a single sequence in order to compute a representation of the sequence. Self-attention has been used successfully in a variety of tasks including reading comprehension, abstractive summarization, textual entailment and learning task-independent sentence representations [4, 22, 23, 19].
自注意力（有时称为内部注意力）是一种关联单个序列不同位置以计算该序列表示的注意力机制。自注意力已成功应用于各种任务，包括阅读理解、摘要提取、文本蕴含和学习任务无关的句子表示 [4, 22, 23, 19]。


End-to-end memory networks are based on a recurrent attention mechanism instead of sequence-aligned recurrence and have been shown to perform well on simple-language question answering and language modeling tasks [28].
端到端记忆网络基于循环注意力机制而非序列对齐的循环，并已证明在简单语言问答和语言建模任务中表现良好 [28]。


To the best of our knowledge, however, the Transformer is the first transduction model relying entirely on self-attention to compute representations of its input and output without using sequence-aligned RNNs or convolution. In the following sections, we will describe the Transformer, motivate self-attention and discuss its advantages over models such as [14, 15] and [8].
然而，据我们所知，Transformer 是首个完全依赖自注意力机制来计算其输入输出表示的转换模型，而无需使用序列对齐的 RNN 或卷积。在接下来的章节中，我们将介绍 Transformer，阐述自注意力的动机，并讨论其相较于 [14, 15] 和 [8] 等模型的优势。


## 3 Model Architecture
## 3 模型架构


Most competitive neural sequence transduction models have an encoder-decoder structure [5, 2, 29]. Here,the encoder maps an input sequence of symbol representations $\left( {{x}_{1},\ldots ,{x}_{n}}\right)$ to a sequence of continuous representations $\mathbf{z} = \left( {{z}_{1},\ldots ,{z}_{n}}\right)$ . Given $\mathbf{z}$ ,the decoder then generates an output sequence $\left( {{y}_{1},\ldots ,{y}_{m}}\right)$ of symbols one element at a time. At each step the model is auto-regressive [9], consuming the previously generated symbols as additional input when generating the next.
大多数有竞争力的神经序列转导模型都具有编码器-解码器结构 [5, 2, 29]。在这里，编码器将符号表示的输入序列 $\left( {{x}_{1},\ldots ,{x}_{n}}\right)$ 映射到连续表示序列 $\mathbf{z} = \left( {{z}_{1},\ldots ,{z}_{n}}\right)$。给定 $\mathbf{z}$，解码器随后每次生成一个符号，即输出序列 $\left( {{y}_{1},\ldots ,{y}_{m}}\right)$。在每一步中，模型都是自回归的 [9]，在生成下一个符号时将先前生成的符号作为额外输入。


The Transformer follows this overall architecture using stacked self-attention and point-wise, fully connected layers for both the encoder and decoder, shown in the left and right halves of Figure 1 respectively.
Transformer 遵循这种整体架构，在编码器和解码器中均使用了堆叠的自注意力层和逐点全连接层，分别如图 1 的左半部分和右半部分所示。


### 3.1 Encoder and Decoder Stacks
### 3.1 编码器与解码器堆栈


Encoder: The encoder is composed of a stack of $N = 6$ identical layers. Each layer has two sub-layers. The first is a multi-head self-attention mechanism, and the second is a simple, position-wise fully connected feed-forward network. We employ a residual connection [10] around each of the two sub-layers, followed by layer normalization [1]. That is, the output of each sub-layer is LayerNorm $\left( {x + \text{ Sublayer }\left( x\right) }\right)$ ,where Sublayer $\left( x\right)$ is the function implemented by the sub-layer itself. To facilitate these residual connections, all sub-layers in the model, as well as the embedding layers,produce outputs of dimension ${d}_{\text{ model }} = {512}$ .
编码器：编码器由 $N = 6$ 层完全相同的层堆叠而成。每层包含两个子层。第一层是多头自注意力机制，第二层是一个简单的逐位置全连接前馈网络。我们在每个子层周围采用残差连接 [10]，随后进行层归一化 [1]. 即每个子层的输出为 LayerNorm $\left( {x + \text{ Sublayer }\left( x\right) }\right)$，其中 Sublayer $\left( x\right)$ 是子层本身实现的函数。为了便于这些残差连接，模型中所有子层以及嵌入层的输出维度均为 ${d}_{\text{ model }} = {512}$。


<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/02/2026_02_18__00_16_19_9698a1.jpg"/>



Figure 1: The Transformer - model architecture.
图 1：Transformer - 模型架构。


Decoder: The decoder is also composed of a stack of $N = 6$ identical layers. In addition to the two sub-layers in each encoder layer, the decoder inserts a third sub-layer, which performs multi-head attention over the output of the encoder stack. Similar to the encoder, we employ residual connections around each of the sub-layers, followed by layer normalization. We also modify the self-attention sub-layer in the decoder stack to prevent positions from attending to subsequent positions. This masking, combined with fact that the output embeddings are offset by one position, ensures that the predictions for position $i$ can depend only on the known outputs at positions less than $i$ .
解码器：解码器也由 $N = 6$ 个相同层堆叠而成。除了每个编码器层中的两个子层外，解码器还插入了第三个子层，用于对编码器堆栈的输出执行多头注意力机制。与编码器类似，我们在每个子层周围采用残差连接，随后进行层归一化。我们还修改了解码器堆栈中的自注意力子层，以防止当前位置关注到后续位置。这种掩码机制结合输出嵌入偏移一个位置的事实，确保对位置 $i$ 的预测仅依赖于小于 $i$ 位置的已知输出。


### 3.2 Attention
### 3.2 Attention


An attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors. The output is computed as a weighted sum of the values, where the weight assigned to each value is computed by a compatibility function of the query with the corresponding key.
注意力函数可被描述为将查询和一组键值对映射到输出的过程，其中查询、键、值及输出均为向量。输出是值的加权和，而分配给每个值的权重由查询与对应键的兼容性函数计算得出。


#### 3.2.1 Scaled Dot-Product Attention
#### 3.2.1 缩放点积注意力 (Scaled Dot-Product Attention)


We call our particular attention "Scaled Dot-Product Attention" (Figure 2). The input consists of queries and keys of dimension ${d}_{k}$ ,and values of dimension ${d}_{v}$ . We compute the dot products of the query with all keys,divide each by $\sqrt{{d}_{k}}$ ,and apply a softmax function to obtain the weights on the values.
我们将这种特殊注意力机制称为“缩放点积注意力”（图2）。输入由维度为 ${d}_{k}$ 的查询和键，以及维度为 ${d}_{v}$ 的值组成。我们计算查询与所有键的点积，将每个结果除以 $\sqrt{{d}_{k}}$ ，并应用 softmax 函数以获得值的权重。


<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/02/2026_02_18__00_16_19_536bae.jpg"/>



Figure 2: (left) Scaled Dot-Product Attention. (right) Multi-Head Attention consists of several attention layers running in parallel.
图 2：(左) 缩放点积注意力机制。(右) 多头注意力机制由多个并行运行的注意力层组成。


In practice, we compute the attention function on a set of queries simultaneously, packed together into a matrix $Q$ . The keys and values are also packed together into matrices $K$ and $V$ . We compute the matrix of outputs as:
在实践中，我们同时对一组查询计算注意力函数，并将其打包成矩阵 $Q$。键和值也分别打包成矩阵 $K$ 和 $V$。我们按如下公式计算输出矩阵：


$$
\operatorname{Attention}\left( {Q,K,V}\right)  = \operatorname{softmax}\left( \frac{Q{K}^{T}}{\sqrt{{d}_{k}}}\right) V \tag{1}
$$



The two most commonly used attention functions are additive attention [2], and dot-product (multiplicative) attention. Dot-product attention is identical to our algorithm, except for the scaling factor of $\frac{1}{\sqrt{{d}_{k}}}$ . Additive attention computes the compatibility function using a feed-forward network with a single hidden layer. While the two are similar in theoretical complexity, dot-product attention is much faster and more space-efficient in practice, since it can be implemented using highly optimized matrix multiplication code.
两种最常用的注意力函数是加法注意力[2]和点积（乘法）注意力。除了缩放因子$\frac{1}{\sqrt{{d}_{k}}}$之外，点积注意力与我们的算法完全相同。加法注意力使用具有单个隐藏层的前馈网络来计算兼容性函数。虽然两者在理论复杂度上相似，但在实践中，点积注意力由于可以使用高度优化的矩阵乘法代码实现，因此速度更快且更节省空间。


While for small values of ${d}_{k}$ the two mechanisms perform similarly,additive attention outperforms dot product attention without scaling for larger values of ${d}_{k}$ [3]. We suspect that for large values of ${d}_{k}$ ,the dot products grow large in magnitude,pushing the softmax function into regions where it has extremely small gradients ${}^{4}$ To counteract this effect,we scale the dot products by $\frac{1}{\sqrt{{d}_{k}}}$ .
虽然当 ${d}_{k}$ 较小时两种机制表现相似，但在 ${d}_{k}$ 较大时，加性注意力优于不进行缩放的点积注意力 [3]。我们怀疑对于较大的 ${d}_{k}$ 值，点积在数量级上增长很大，将 softmax 函数推向梯度极小的区域 ${}^{4}$。为了抵消这种影响，我们将点积缩放 $\frac{1}{\sqrt{{d}_{k}}}$。


#### 3.2.2 Multi-Head Attention
#### 3.2.2 多头注意力


Instead of performing a single attention function with ${d}_{\text{ model }}$ -dimensional keys,values and queries, we found it beneficial to linearly project the queries,keys and values $h$ times with different,learned linear projections to ${d}_{k},{d}_{k}$ and ${d}_{v}$ dimensions,respectively. On each of these projected versions of queries,keys and values we then perform the attention function in parallel,yielding ${d}_{v}$ -dimensional output values. These are concatenated and once again projected, resulting in the final values, as depicted in Figure 2
我们发现，与其使用 ${d}_{\text{ model }}$ 维的键、值和查询执行单个注意力函数，不如将查询、键和值分别通过不同的、学习到的线性投影进行 $h$ 次线性投影，分别投影到 ${d}_{k},{d}_{k}$ 和 ${d}_{v}$ 维度。在这些投影后的查询、键和值版本上，我们并行执行注意力函数，产生 ${d}_{v}$ 维的输出值。如图 2 所示，这些值被拼接并再次投影，得到最终数值。


Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions. With a single attention head, averaging inhibits this.
多头注意力允许模型同时关注来自不同位置、不同表示子空间的信息。对于单个注意力头，平均化会抑制这一点。


$$
\text{ MultiHead }\left( {Q,K,V}\right)  = \text{ Concat }\left( {{\text{ head }}_{1},\ldots ,{\text{ head }}_{\mathrm{h}}}\right) {W}^{O}
$$



$$
\text{ where }{\operatorname{head}}_{\mathrm{i}} = \operatorname{Attention}\left( {Q{W}_{i}^{Q},K{W}_{i}^{K},V{W}_{i}^{V}}\right)
$$



---



${}^{4}$ To illustrate why the dot products get large,assume that the components of $q$ and $k$ are independent random variables with mean 0 and variance 1 . Then their dot product, $q \cdot  k = \mathop{\sum }\limits_{{i = 1}}^{{d}_{k}}{q}_{i}{k}_{i}$ ,has mean 0 and variance ${d}_{k}$ .
${}^{4}$ 为了说明点积为何变大，假设 $q$ 和 $k$ 的分量是均值为 0、方差为 1 的独立随机变量。那么它们的点积 $q \cdot  k = \mathop{\sum }\limits_{{i = 1}}^{{d}_{k}}{q}_{i}{k}_{i}$ 均值为 0，方差为 ${d}_{k}$。


---



Where the projections are parameter matrices ${W}_{i}^{Q} \in  {\mathbb{R}}^{{d}_{\text{ model }} \times  {d}_{k}},{W}_{i}^{K} \in  {\mathbb{R}}^{{d}_{\text{ model }} \times  {d}_{k}},{W}_{i}^{V} \in  {\mathbb{R}}^{{d}_{\text{ model }} \times  {d}_{v}}$ and ${W}^{O} \in  {\mathbb{R}}^{h{d}_{v} \times  {d}_{\text{ model }}}$ .
其中投影是参数矩阵 ${W}_{i}^{Q} \in  {\mathbb{R}}^{{d}_{\text{ model }} \times  {d}_{k}},{W}_{i}^{K} \in  {\mathbb{R}}^{{d}_{\text{ model }} \times  {d}_{k}},{W}_{i}^{V} \in  {\mathbb{R}}^{{d}_{\text{ model }} \times  {d}_{v}}$ 和 ${W}^{O} \in  {\mathbb{R}}^{h{d}_{v} \times  {d}_{\text{ model }}}$。


In this work we employ $h = 8$ parallel attention layers,or heads. For each of these we use ${d}_{k} = {d}_{v} = {d}_{\text{ model }}/h = {64}$ . Due to the reduced dimension of each head,the total computational cost is similar to that of single-head attention with full dimensionality.
在这项工作中，我们采用 $h = 8$ 个并行注意力层（即头）。对于每一个，我们使用 ${d}_{k} = {d}_{v} = {d}_{\text{ model }}/h = {64}$。由于每个头的维度降低，总计算成本与全维度的单头注意力相似。


#### 3.2.3 Applications of Attention in our Model
#### 3.2.3 注意力在模型中的应用


The Transformer uses multi-head attention in three different ways:
Transformer 以三种不同的方式使用多头注意力：


- In "encoder-decoder attention" layers, the queries come from the previous decoder layer, and the memory keys and values come from the output of the encoder. This allows every position in the decoder to attend over all positions in the input sequence. This mimics the typical encoder-decoder attention mechanisms in sequence-to-sequence models such as [31, 2, 8].
- 在“编码器-解码器注意力”层中，查询来自前一个解码器层，而记忆键和值来自编码器的输出。这使得解码器中的每个位置都能关注输入序列中的所有位置。这模拟了序列到序列模型（如 [31, 2, 8]）中典型的编码器-解码器注意力机制。


- The encoder contains self-attention layers. In a self-attention layer all of the keys, values and queries come from the same place, in this case, the output of the previous layer in the encoder. Each position in the encoder can attend to all positions in the previous layer of the encoder.
- 编码器包含自注意力层。在自注意力层中，所有的键、值和查询都来自同一个地方，即编码器前一层的输出。编码器中的每个位置都可以关注编码器前一层的所有位置。


- Similarly, self-attention layers in the decoder allow each position in the decoder to attend to all positions in the decoder up to and including that position. We need to prevent leftward information flow in the decoder to preserve the auto-regressive property. We implement this inside of scaled dot-product attention by masking out (setting to - ∞) all values in the input of the softmax which correspond to illegal connections. See Figure 2
- 类似地，解码器中的自注意力层允许解码器中的每个位置关注到该位置（含）之前的解码器所有位置。我们需要防止解码器中的向左信息流，以保持自回归特性。我们在缩放点积注意力内部通过掩码处理（设为 -∞）来实现这一点，即遮蔽 softmax 输入中所有对应非法连接的值。参见图 2


### 3.3 Position-wise Feed-Forward Networks
### 3.3 逐位置前馈网络


In addition to attention sub-layers, each of the layers in our encoder and decoder contains a fully connected feed-forward network, which is applied to each position separately and identically. This consists of two linear transformations with a ReLU activation in between.
除了注意力子层外，编码器和解码器中的每一层都包含一个全连接的前馈网络，该网络分别且同样地应用于每个位置。它由两个线性变换组成，中间有一个 ReLU 激活。


$$
\operatorname{FFN}\left( x\right)  = \max \left( {0,x{W}_{1} + {b}_{1}}\right) {W}_{2} + {b}_{2} \tag{2}
$$



While the linear transformations are the same across different positions, they use different parameters from layer to layer. Another way of describing this is as two convolutions with kernel size 1. The dimensionality of input and output is ${d}_{\text{ model }} = {512}$ ,and the inner-layer has dimensionality ${d}_{ff} = {2048}$ .
虽然线性变换在不同位置上是相同的，但在层与层之间使用不同的参数。另一种描述方式是两个卷积核大小为 1 的卷积。输入和输出的维度为 ${d}_{\text{ model }} = {512}$，内层维度为 ${d}_{ff} = {2048}$。


### 3.4 Embeddings and Softmax
### 3.4 嵌入和 Softmax


Similarly to other sequence transduction models, we use learned embeddings to convert the input tokens and output tokens to vectors of dimension ${d}_{\text{ model }}$ . We also use the usual learned linear transformation and softmax function to convert the decoder output to predicted next-token probabilities. In our model, we share the same weight matrix between the two embedding layers and the pre-softmax linear transformation,similar to [24]. In the embedding layers,we multiply those weights by $\sqrt{{d}_{\text{ model }}}$ .
与其他序列转导模型类似，我们使用学习到的嵌入将输入标记和输出标记转换为维度为 ${d}_{\text{ model }}$ 的向量。我们还使用通常的学习线性变换和 softmax 函数将解码器输出转换为预测的下一个标记概率。在我们的模型中，我们在两个嵌入层和 softmax 前线性变换之间共享相同的权重矩阵，类似于 [24]。在嵌入层中，我们将这些权重乘以 $\sqrt{{d}_{\text{ model }}}$。


### 3.5 Positional Encoding
### 3.5 位置编码


Since our model contains no recurrence and no convolution, in order for the model to make use of the order of the sequence, we must inject some information about the relative or absolute position of the tokens in the sequence. To this end, we add "positional encodings" to the input embeddings at the bottoms of the encoder and decoder stacks. The positional encodings have the same dimension ${d}_{\text{ model }}$ as the embeddings, so that the two can be summed. There are many choices of positional encodings, learned and fixed [8].
由于我们的模型不包含循环和卷积，为了使模型能够利用序列的顺序，我们必须注入一些关于序列中标记相对或绝对位置的信息。为此，我们在编码器和解码器堆栈底部的输入嵌入中添加“位置编码”。位置编码具有与嵌入相同的维度 ${d}_{\text{ model }}$，因此两者可以相加。位置编码有许多选择，包括学习型和固定型 [8]。


Table 1: Maximum path lengths, per-layer complexity and minimum number of sequential operations for different layer types. $n$ is the sequence length, $d$ is the representation dimension, $k$ is the kernel size of convolutions and $r$ the size of the neighborhood in restricted self-attention.
表 1：不同层类型的最大路径长度、每层复杂度和最小顺序操作数。$n$ 是序列长度，$d$ 是表示维度，$k$ 是卷积核大小，$r$ 是受限自注意力的邻域大小。


<table><tr><td>Layer Type</td><td>Complexity per Layer</td><td>Sequential Operations</td><td>Maximum Path Length</td></tr><tr><td>Self-Attention</td><td>$O\left( {{n}^{2} \cdot  d}\right)$</td><td>$O\left( 1\right)$</td><td>$O\left( 1\right)$</td></tr><tr><td>Recurrent</td><td>$O\left( {n \cdot  {d}^{2}}\right)$</td><td>$O\left( n\right)$</td><td>$O\left( n\right)$</td></tr><tr><td>Convolutional</td><td>$O\left( {k \cdot  n \cdot  {d}^{2}}\right)$</td><td>$O\left( 1\right)$</td><td>$O\left( {{\log }_{k}\left( n\right) }\right)$</td></tr><tr><td>Self-Attention (restricted)</td><td>$O\left( {r \cdot  n \cdot  d}\right)$</td><td>$O\left( 1\right)$</td><td>$O\left( {n/r}\right)$</td></tr></table>
<table><tbody><tr><td>层类型</td><td>每层复杂度</td><td>串行操作次数</td><td>最大路径长度</td></tr><tr><td>自注意力</td><td>$O\left( {{n}^{2} \cdot  d}\right)$</td><td>$O\left( 1\right)$</td><td>$O\left( 1\right)$</td></tr><tr><td>递归</td><td>$O\left( {n \cdot  {d}^{2}}\right)$</td><td>$O\left( n\right)$</td><td>$O\left( n\right)$</td></tr><tr><td>卷积</td><td>$O\left( {k \cdot  n \cdot  {d}^{2}}\right)$</td><td>$O\left( 1\right)$</td><td>$O\left( {{\log }_{k}\left( n\right) }\right)$</td></tr><tr><td>自注意力（受限）</td><td>$O\left( {r \cdot  n \cdot  d}\right)$</td><td>$O\left( 1\right)$</td><td>$O\left( {n/r}\right)$</td></tr></tbody></table>


In this work, we use sine and cosine functions of different frequencies:
在工作中，我们使用了不同频率的正弦和余弦函数：


$$
P{E}_{\left( pos,2i\right) } = \sin \left( {{pos}/{10000}^{{2i}/{d}_{\text{ model }}}}\right)
$$



$$
P{E}_{\left( pos,2i + 1\right) } = \cos \left( {{pos}/{10000}^{{2i}/{d}_{\text{ model }}}}\right)
$$



where pos is the position and $i$ is the dimension. That is,each dimension of the positional encoding corresponds to a sinusoid. The wavelengths form a geometric progression from ${2\pi }$ to ${10000} \cdot  {2\pi }$ . We chose this function because we hypothesized it would allow the model to easily learn to attend by relative positions,since for any fixed offset $k,P{E}_{{pos} + k}$ can be represented as a linear function of $P{E}_{pos}$ .
其中 pos 是位置，$i$ 是维度。也就是说，位置编码的每个维度都对应一个正弦曲线。波长形成从 ${2\pi }$ 到 ${10000} \cdot  {2\pi }$ 的等比数列。我们选择这个函数是因为我们假设它能让模型很容易地学习通过相对位置进行注意力机制的关注，因为对于任何固定的偏移量 $k,P{E}_{{pos} + k}$，都可以表示为 $P{E}_{pos}$ 的线性函数。


We also experimented with using learned positional embeddings [8] instead, and found that the two versions produced nearly identical results (see Table 3 row (E)). We chose the sinusoidal version because it may allow the model to extrapolate to sequence lengths longer than the ones encountered during training.
我们也尝试过使用学习到的位置嵌入 [8] 来代替，并发现这两个版本产生的结果几乎完全相同（见表 3 第 (E) 行）。我们选择正弦版本是因为它可能允许模型外推到比训练期间遇到的序列长度更长的序列。


## 4 Why Self-Attention
## 4 为什么选择自注意力


In this section we compare various aspects of self-attention layers to the recurrent and convolutional layers commonly used for mapping one variable-length sequence of symbol representations $\left( {{x}_{1},\ldots ,{x}_{n}}\right)$ to another sequence of equal length $\left( {{z}_{1},\ldots ,{z}_{n}}\right)$ ,with ${x}_{i},{z}_{i} \in  {\mathbb{R}}^{d}$ ,such as a hidden layer in a typical sequence transduction encoder or decoder. Motivating our use of self-attention we consider three desiderata.
在本节中，我们将自注意力层的各个方面与常用于将一个可变长度的符号表示序列 $\left( {{x}_{1},\ldots ,{x}_{n}}\right)$ 映射到另一个等长序列 $\left( {{z}_{1},\ldots ,{z}_{n}}\right)$（其中 ${x}_{i},{z}_{i} \in  {\mathbb{R}}^{d}$）的循环层和卷积层进行比较，例如典型的序列转导编码器或解码器中的隐藏层。为了说明我们使用自注意力的动机，我们考虑了三个要求。


One is the total computational complexity per layer. Another is the amount of computation that can be parallelized, as measured by the minimum number of sequential operations required.
一个是每层的总计算复杂度。另一个是可以并行化的计算量，由所需的最小顺序操作步数来衡量。


The third is the path length between long-range dependencies in the network. Learning long-range dependencies is a key challenge in many sequence transduction tasks. One key factor affecting the ability to learn such dependencies is the length of the paths forward and backward signals have to traverse in the network. The shorter these paths between any combination of positions in the input and output sequences, the easier it is to learn long-range dependencies [11]. Hence we also compare the maximum path length between any two input and output positions in networks composed of the different layer types.
第三是网络中长距离依赖关系之间的路径长度。学习长距离依赖关系是许多序列转导任务中的关键挑战。影响学习此类依赖关系能力的一个关键因素是前向和后向信号必须在网络中经过的路径长度。输入和输出序列中任何位置组合之间的这些路径越短，就越容易学习长距离依赖关系 [11]。因此，我们还比较了由不同层类型组成的网络中任何两个输入和输出位置之间的最大路径长度。


As noted in Table 1, a self-attention layer connects all positions with a constant number of sequentially executed operations,whereas a recurrent layer requires $O\left( n\right)$ sequential operations. In terms of computational complexity, self-attention layers are faster than recurrent layers when the sequence length $n$ is smaller than the representation dimensionality $d$ ,which is most often the case with sentence representations used by state-of-the-art models in machine translations, such as word-piece [31] and byte-pair [25] representations. To improve computational performance for tasks involving very long sequences,self-attention could be restricted to considering only a neighborhood of size $r$ in the input sequence centered around the respective output position. This would increase the maximum path length to $O\left( {n/r}\right)$ . We plan to investigate this approach further in future work.
如表 1 所示，自注意力层用常数步顺序执行的操作连接所有位置，而循环层则需要 $O\left( n\right)$ 步顺序操作。在计算复杂度方面，当序列长度 $n$ 小于表示维度 $d$ 时，自注意力层比循环层更快，这在机器翻译中最先进模型所使用的句子表示（如 word-piece [31] 和 byte-pair [25] 表示）中是最常见的情况。为了提高处理极长序列任务的计算性能，自注意力可以限制为仅考虑输入序列中以相应输出位置为中心、大小为 $r$ 的邻域。这会将最大路径长度增加到 $O\left( {n/r}\right)$。我们计划在未来的工作中进一步研究这种方法。


A single convolutional layer with kernel width $k < n$ does not connect all pairs of input and output positions. Doing so requires a stack of $O\left( {n/k}\right)$ convolutional layers in the case of contiguous kernels, or $O\left( {{\log }_{k}\left( n\right) }\right)$ in the case of dilated convolutions [15],increasing the length of the longest paths between any two positions in the network. Convolutional layers are generally more expensive than recurrent layers,by a factor of $k$ . Separable convolutions [6],however,decrease the complexity considerably,to $O\left( {k \cdot  n \cdot  d + n \cdot  {d}^{2}}\right)$ . Even with $k = n$ ,however,the complexity of a separable convolution is equal to the combination of a self-attention layer and a point-wise feed-forward layer, the approach we take in our model.
卷积核宽度为 $k < n$ 的单个卷积层并不能连接所有输入和输出位置对。在连续卷积核的情况下，这需要堆叠 $O\left( {n/k}\right)$ 个卷积层，而在空心卷积 [15] 的情况下需要堆叠 $O\left( {{\log }_{k}\left( n\right) }\right)$ 个，这增加了网络中任何两个位置之间最长路径的长度。卷积层通常比循环层更昂贵，倍数为 $k$。然而，可分离卷积 [6] 显著降低了复杂度，达到 $O\left( {k \cdot  n \cdot  d + n \cdot  {d}^{2}}\right)$。然而，即使 $k = n$，可分离卷积的复杂度也等于自注意力层和逐点前馈层的组合，即我们在模型中采用的方法。


As side benefit, self-attention could yield more interpretable models. We inspect attention distributions from our models and present and discuss examples in the appendix. Not only do individual attention heads clearly learn to perform different tasks, many appear to exhibit behavior related to the syntactic and semantic structure of the sentences.
作为额外的好处，自注意力可以产生更具可解释性的模型。我们检查了模型的注意力分布，并在附录中展示和讨论了示例。不仅单个注意力头明显学会了执行不同的任务，而且许多注意力头似乎表现出与句子的语法和语义结构相关的行为。


## 5 Training
## 5 训练


This section describes the training regime for our models.
本节描述了我们模型的训练方案。


### 5.1 Training Data and Batching
### 5.1 训练数据和分批


We trained on the standard WMT 2014 English-German dataset consisting of about 4.5 million sentence pairs. Sentences were encoded using byte-pair encoding [3], which has a shared source-target vocabulary of about 37000 tokens. For English-French, we used the significantly larger WMT 2014 English-French dataset consisting of ${36}\mathrm{M}$ sentences and split tokens into a 32000 word-piece vocabulary [31]. Sentence pairs were batched together by approximate sequence length. Each training batch contained a set of sentence pairs containing approximately 25000 source tokens and 25000 target tokens.
我们在标准的 WMT 2014 英德数据集上进行训练，该数据集包含约 450 万个句子对。句子采用字节对编码 [3] 进行编码，其共享的源语-目标语词表包含约 37000 个标记。对于英法翻译，我们使用了规模大得多的 WMT 2014 英法数据集，包含 ${36}\mathrm{M}$ 个句子，并将标记拆分为 32000 个词片词表 [31]。句子对按近似序列长度进行批处理。每个训练批次包含一组句子对，约含 25000 个源语标记和 25000 个目标语标记。


### 5.2 Hardware and Schedule
### 5.2 硬件与进度


We trained our models on one machine with 8 NVIDIA P100 GPUs. For our base models using the hyperparameters described throughout the paper, each training step took about 0.4 seconds. We trained the base models for a total of 100,000 steps or 12 hours. For our big models,(described on the bottom line of table 3), step time was 1.0 seconds. The big models were trained for 300,000 steps (3.5 days).
我们在配备 8 个 NVIDIA P100 GPU 的单机上训练模型。对于使用文中描述的超参数的基础模型，每个训练步骤耗时约 0.4 秒。基础模型总共训练了 100,000 步或 12 小时。对于大模型（见表 3 底行描述），单步用时为 1.0 秒。大模型训练了 300,000 步（3.5 天）。


### 5.3 Optimizer
### 5.3 优化器


We used the Adam optimizer [17] with ${\beta }_{1} = {0.9},{\beta }_{2} = {0.98}$ and $\epsilon  = {10}^{-9}$ . We varied the learning rate over the course of training, according to the formula:
我们使用了 Adam 优化器 [17]，参数为 ${\beta }_{1} = {0.9},{\beta }_{2} = {0.98}$ 和 $\epsilon  = {10}^{-9}$。在训练过程中，我们根据以下公式改变学习率：


$$
\text{ lrate } = {d}_{\text{ model }}^{-{0.5}} \cdot  \min \left( \right. \text{ step }\left. {{ \text{ \_ num } }^{-{0.5}},\text{ step\_num } \cdot  \text{ warmup\_steps }{}^{-{1.5}}}\right) \tag{3}
$$



This corresponds to increasing the learning rate linearly for the first warmup_steps training steps, and decreasing it thereafter proportionally to the inverse square root of the step number. We used warmup_steps = 4000.
这对应于在最初的 warmup_steps 训练步数内线性增加学习率，此后学习率与步数的平方根倒数成比例地减小。我们使用 warmup_steps = 4000。


### 5.4 Regularization
### 5.4 正则化


We employ three types of regularization during training:
我们在训练期间采用了三种类型的正则化：


Residual Dropout We apply dropout [27] to the output of each sub-layer, before it is added to the sub-layer input and normalized. In addition, we apply dropout to the sums of the embeddings and the positional encodings in both the encoder and decoder stacks. For the base model, we use a rate of ${P}_{drop} = {0.1}$ .
残差 Dropout 我们在每个子层的输出添加到子层输入并归一化之前应用 Dropout [27]。此外，我们在编码器和解码器栈中对嵌入与位置编码之和应用 Dropout。对于基础模型，我们使用的比率为 ${P}_{drop} = {0.1}$。


Table 2: The Transformer achieves better BLEU scores than previous state-of-the-art models on the English-to-German and English-to-French newstest2014 tests at a fraction of the training cost.
表 2：Transformer 在英德和英法 newstest2014 测试集上取得了比之前最先进模型更好的 BLEU 评分，且训练成本仅为后者的一小部分。


<table><tr><td rowspan="2">Model</td><td colspan="2">BLEU</td><td colspan="2">Training Cost (FLOPs)</td></tr><tr><td>EN-DE</td><td>EN-FR</td><td>EN-DE</td><td>EN-FR</td></tr><tr><td>ByteNet [15]</td><td>23.75</td><td></td><td></td><td></td></tr><tr><td>Deep-Att + PosUnk 32</td><td></td><td>39.2</td><td></td><td>${1.0} \cdot  {10}^{20}$</td></tr><tr><td>GNMT + RL [31]</td><td>24.6</td><td>39.92</td><td>${2.3} \cdot  {10}^{19}$</td><td>${1.4} \cdot  {10}^{20}$</td></tr><tr><td>ConvS2S [8]</td><td>25.16</td><td>40.46</td><td>${9.6} \cdot  {10}^{18}$</td><td>${1.5} \cdot  {10}^{20}$</td></tr><tr><td>MoE [26]</td><td>26.03</td><td>40.56</td><td>${2.0} \cdot  {10}^{19}$</td><td>${1.2} \cdot  {10}^{20}$</td></tr><tr><td>Deep-Att + PosUnk Ensemble [32]</td><td></td><td>40.4</td><td></td><td>${8.0} \cdot  {10}^{20}$</td></tr><tr><td>GNMT + RL Ensemble [31]</td><td>26.30</td><td>41.16</td><td>1.8 $\cdot  {10}^{20}$</td><td>${1.1} \cdot  {10}^{21}$</td></tr><tr><td>ConvS2S Ensemble [8]</td><td>26.36</td><td>41.29</td><td>7.7 $\cdot  {10}^{19}$</td><td>${1.2} \cdot  {10}^{21}$</td></tr><tr><td>Transformer (base model)</td><td>27.3</td><td>38.1</td><td colspan="2">${3.3} \cdot  {10}^{18}$</td></tr><tr><td>Transformer (big)</td><td>28.4</td><td>41.0</td><td colspan="2">${2.3} \cdot  {10}^{19}$</td></tr></table>
<table><tbody><tr><td rowspan="2">模型</td><td colspan="2">BLEU</td><td colspan="2">训练成本 (FLOPs)</td></tr><tr><td>英-德</td><td>英-法</td><td>英-德</td><td>英-法</td></tr><tr><td>ByteNet [15]</td><td>23.75</td><td></td><td></td><td></td></tr><tr><td>Deep-Att + PosUnk 32</td><td></td><td>39.2</td><td></td><td>${1.0} \cdot  {10}^{20}$</td></tr><tr><td>GNMT + RL [31]</td><td>24.6</td><td>39.92</td><td>${2.3} \cdot  {10}^{19}$</td><td>${1.4} \cdot  {10}^{20}$</td></tr><tr><td>ConvS2S [8]</td><td>25.16</td><td>40.46</td><td>${9.6} \cdot  {10}^{18}$</td><td>${1.5} \cdot  {10}^{20}$</td></tr><tr><td>MoE [26]</td><td>26.03</td><td>40.56</td><td>${2.0} \cdot  {10}^{19}$</td><td>${1.2} \cdot  {10}^{20}$</td></tr><tr><td>Deep-Att + PosUnk 集成 [32]</td><td></td><td>40.4</td><td></td><td>${8.0} \cdot  {10}^{20}$</td></tr><tr><td>GNMT + RL 集成 [31]</td><td>26.30</td><td>41.16</td><td>1.8 $\cdot  {10}^{20}$</td><td>${1.1} \cdot  {10}^{21}$</td></tr><tr><td>ConvS2S 集成 [8]</td><td>26.36</td><td>41.29</td><td>7.7 $\cdot  {10}^{19}$</td><td>${1.2} \cdot  {10}^{21}$</td></tr><tr><td>Transformer (基础模型)</td><td>27.3</td><td>38.1</td><td colspan="2">${3.3} \cdot  {10}^{18}$</td></tr><tr><td>Transformer (大模型)</td><td>28.4</td><td>41.0</td><td colspan="2">${2.3} \cdot  {10}^{19}$</td></tr></tbody></table>


Label Smoothing During training,we employed label smoothing of value ${\epsilon }_{ls} = {0.1}$ [30]. This hurts perplexity, as the model learns to be more unsure, but improves accuracy and BLEU score.
标签平滑 训练期间，我们采用了值 ${\epsilon }_{ls} = {0.1}$ [30] 的标签平滑。由于模型学会了更加不确定，这会损害困惑度，但能提高准确率和 BLEU 分数。


## 6 Results
## 6 结果


### 6.1 Machine Translation
### 6.1 机器翻译


On the WMT 2014 English-to-German translation task, the big transformer model (Transformer (big) in Table 2) outperforms the best previously reported models (including ensembles) by more than 2.0 BLEU, establishing a new state-of-the-art BLEU score of 28.4. The configuration of this model is listed in the bottom line of Table 3 Training took 3.5 days on 8 P100 GPUs. Even our base model surpasses all previously published models and ensembles, at a fraction of the training cost of any of the competitive models.
在 WMT 2014 英德翻译任务中，大型 Transformer 模型（表 2 中的 Transformer (big)）比之前报告的最佳模型（包括集成模型）高出超过 2.0 BLEU，创下了 28.4 的新标杆 BLEU 分数。该模型的配置列在表 3 的最后一行。在 8 个 P100 GPU 上训练耗时 3.5 天。即使是我们的基础模型也超过了之前所有发布的模型和集成模型，而训练成本仅为任何竞争模型的一小部分。


On the WMT 2014 English-to-French translation task, our big model achieves a BLEU score of 41.0, outperforming all of the previously published single models, at less than 1/4 the training cost of the previous state-of-the-art model. The Transformer (big) model trained for English-to-French used dropout rate ${P}_{\text{ drop }} = {0.1}$ ,instead of 0.3 .
在 WMT 2014 英法翻译任务中，我们的大型模型达到了 41.0 的 BLEU 分数，优于之前所有发布的单模型，且训练成本不足先前最先进模型的 1/4。用于英法翻译的 Transformer (big) 模型使用的丢弃率为 ${P}_{\text{ drop }} = {0.1}$，而非 0.3。


For the base models, we used a single model obtained by averaging the last 5 checkpoints, which were written at 10-minute intervals. For the big models, we averaged the last 20 checkpoints. We used beam search with a beam size of 4 and length penalty $\alpha  = {0.6}$ [31]. These hyperparameters were chosen after experimentation on the development set. We set the maximum output length during inference to input length + 50, but terminate early when possible [31].
对于基础模型，我们使用通过对最后 5 个检查点（每 10 分钟写入一次）取平均值获得的单个模型。对于大型模型，我们对最后 20 个检查点取平均值。我们使用了束搜索，束大小为 4，长度惩罚为 $\alpha  = {0.6}$ [31]。这些超参数是在开发集上进行实验后选定的。我们在推理过程中将最大输出长度设置为输入长度 + 50，但在可能的情况下尽早终止 [31]。


Table 2 summarizes our results and compares our translation quality and training costs to other model architectures from the literature. We estimate the number of floating point operations used to train a model by multiplying the training time, the number of GPUs used, and an estimate of the sustained single-precision floating-point capacity of each GPU ${}^{5}$
表 2 总结了我们的结果，并将我们的翻译质量和训练成本与文献中的其他模型架构进行了比较。我们通过将训练时间、使用的 GPU 数量以及每个 GPU 的持续单精度浮点能力的估计值 ${}^{5}$ 相乘，来估算训练模型所使用的浮点运算次数。


### 6.2 Model Variations
### 6.2 模型变体


To evaluate the importance of different components of the Transformer, we varied our base model in different ways, measuring the change in performance on English-to-German translation on the development set, newstest2013. We used beam search as described in the previous section, but no checkpoint averaging. We present these results in Table 3
为了评估 Transformer 不同组件的重要性，我们以不同方式改变了基础模型，测量了开发集 newstest2013 上英德翻译性能的变化。我们使用了前一节所述的束搜索，但没有进行检查点平均。我们在表 3 中展示了这些结果。


In Table 3 rows (A), we vary the number of attention heads and the attention key and value dimensions, keeping the amount of computation constant, as described in Section 3.2.2 While single-head attention is 0.9 BLEU worse than the best setting, quality also drops off with too many heads.
在表 3 的 (A) 行中，我们改变了注意力头数以及注意力键和值的维度，并保持计算量不变，如第 3.2.2 节所述。虽然单头注意力比最佳设置差 0.9 BLEU，但头数过多质量也会下降。


---



${}^{5}$ We used values of 2.8,3.7,6.0 and 9.5 TFLOPS for K80,K40,M40 and P100,respectively.
${}^{5}$ 我们为 K80、K40、M40 和 P100 分别使用了 2.8、3.7、6.0 和 9.5 TFLOPS 的值。


---



Table 3: Variations on the Transformer architecture. Unlisted values are identical to those of the base model. All metrics are on the English-to-German translation development set, newstest2013. Listed perplexities are per-wordpiece, according to our byte-pair encoding, and should not be compared to per-word perplexities.
表 3：Transformer 架构的变体。未列出的值与基础模型相同。所有指标均基于英德翻译开发集 newstest2013。列出的困惑度是根据我们的字节对编码计算的每个分词的困惑度，不应与每个单词的困惑度进行比较。


<table><tr><td></td><td>$N$</td><td>${d}_{\text{ model }}$</td><td>${d}_{\mathrm{{ff}}}$</td><td>$h$</td><td>${d}_{k}$</td><td>${d}_{v}$</td><td>${P}_{drop}$</td><td>${\epsilon }_{ls}$</td><td>train steps</td><td>PPL (dev)</td><td>BLEU (dev)</td><td>params $\times  {10}^{6}$</td></tr><tr><td>base</td><td>6</td><td>512</td><td>2048</td><td>8</td><td>64</td><td>64</td><td>0.1</td><td>0.1</td><td>100K</td><td>4.92</td><td>25.8</td><td>65</td></tr><tr><td rowspan="4">(A)</td><td></td><td></td><td></td><td>1</td><td>512</td><td>512</td><td></td><td></td><td></td><td>5.29</td><td>24.9</td><td></td></tr><tr><td></td><td></td><td></td><td>4</td><td>128</td><td>128</td><td></td><td></td><td></td><td>5.00</td><td>25.5</td><td></td></tr><tr><td></td><td></td><td></td><td>16</td><td>32</td><td>32</td><td></td><td></td><td></td><td>4.91</td><td>25.8</td><td></td></tr><tr><td></td><td></td><td></td><td>32</td><td>16</td><td>16</td><td></td><td></td><td></td><td>5.01</td><td>25.4</td><td></td></tr><tr><td rowspan="2">(B)</td><td></td><td></td><td></td><td></td><td>16</td><td></td><td></td><td></td><td></td><td>5.16</td><td>25.1</td><td>58</td></tr><tr><td></td><td></td><td></td><td></td><td>32</td><td></td><td></td><td></td><td></td><td>5.01</td><td>25.4</td><td>60</td></tr><tr><td rowspan="7">(C)</td><td>2</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>6.11</td><td>23.7</td><td>36</td></tr><tr><td>4</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>5.19</td><td>25.3</td><td>50</td></tr><tr><td>8</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>4.88</td><td>25.5</td><td>80</td></tr><tr><td></td><td>256</td><td></td><td></td><td>32</td><td>32</td><td></td><td></td><td></td><td>5.75</td><td>24.5</td><td>28</td></tr><tr><td></td><td>1024</td><td></td><td></td><td>128</td><td>128</td><td></td><td></td><td></td><td>4.66</td><td>26.0</td><td>168</td></tr><tr><td></td><td></td><td>1024</td><td></td><td></td><td></td><td></td><td></td><td></td><td>5.12</td><td>25.4</td><td>53</td></tr><tr><td></td><td></td><td>4096</td><td></td><td></td><td></td><td></td><td></td><td></td><td>4.75</td><td>26.2</td><td>90</td></tr><tr><td rowspan="4">(D)</td><td></td><td></td><td></td><td></td><td></td><td></td><td>0.0</td><td></td><td></td><td>5.77</td><td>24.6</td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td>0.2</td><td></td><td></td><td>4.95</td><td>25.5</td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0.0</td><td></td><td>4.67</td><td>25.3</td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0.2</td><td></td><td>5.47</td><td>25.7</td><td></td></tr><tr><td>(E)</td><td colspan="9">positional embedding instead of sinusoids</td><td>4.92</td><td>25.7</td><td></td></tr><tr><td>big</td><td>6</td><td>1024</td><td>4096</td><td>16</td><td></td><td></td><td>0.3</td><td></td><td>300K</td><td>4.33</td><td>26.4</td><td>213</td></tr></table>
<table><tbody><tr><td></td><td>$N$</td><td>${d}_{\text{ model }}$</td><td>${d}_{\mathrm{{ff}}}$</td><td>$h$</td><td>${d}_{k}$</td><td>${d}_{v}$</td><td>${P}_{drop}$</td><td>${\epsilon }_{ls}$</td><td>训练步数</td><td>PPL (验证集)</td><td>BLEU (验证集)</td><td>参数量 $\times  {10}^{6}$</td></tr><tr><td>基础</td><td>6</td><td>512</td><td>2048</td><td>8</td><td>64</td><td>64</td><td>0.1</td><td>0.1</td><td>100K</td><td>4.92</td><td>25.8</td><td>65</td></tr><tr><td rowspan="4">(A)</td><td></td><td></td><td></td><td>1</td><td>512</td><td>512</td><td></td><td></td><td></td><td>5.29</td><td>24.9</td><td></td></tr><tr><td></td><td></td><td></td><td>4</td><td>128</td><td>128</td><td></td><td></td><td></td><td>5.00</td><td>25.5</td><td></td></tr><tr><td></td><td></td><td></td><td>16</td><td>32</td><td>32</td><td></td><td></td><td></td><td>4.91</td><td>25.8</td><td></td></tr><tr><td></td><td></td><td></td><td>32</td><td>16</td><td>16</td><td></td><td></td><td></td><td>5.01</td><td>25.4</td><td></td></tr><tr><td rowspan="2">(B)</td><td></td><td></td><td></td><td></td><td>16</td><td></td><td></td><td></td><td></td><td>5.16</td><td>25.1</td><td>58</td></tr><tr><td></td><td></td><td></td><td></td><td>32</td><td></td><td></td><td></td><td></td><td>5.01</td><td>25.4</td><td>60</td></tr><tr><td rowspan="7">(C)</td><td>2</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>6.11</td><td>23.7</td><td>36</td></tr><tr><td>4</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>5.19</td><td>25.3</td><td>50</td></tr><tr><td>8</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>4.88</td><td>25.5</td><td>80</td></tr><tr><td></td><td>256</td><td></td><td></td><td>32</td><td>32</td><td></td><td></td><td></td><td>5.75</td><td>24.5</td><td>28</td></tr><tr><td></td><td>1024</td><td></td><td></td><td>128</td><td>128</td><td></td><td></td><td></td><td>4.66</td><td>26.0</td><td>168</td></tr><tr><td></td><td></td><td>1024</td><td></td><td></td><td></td><td></td><td></td><td></td><td>5.12</td><td>25.4</td><td>53</td></tr><tr><td></td><td></td><td>4096</td><td></td><td></td><td></td><td></td><td></td><td></td><td>4.75</td><td>26.2</td><td>90</td></tr><tr><td rowspan="4">(D)</td><td></td><td></td><td></td><td></td><td></td><td></td><td>0.0</td><td></td><td></td><td>5.77</td><td>24.6</td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td>0.2</td><td></td><td></td><td>4.95</td><td>25.5</td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0.0</td><td></td><td>4.67</td><td>25.3</td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0.2</td><td></td><td>5.47</td><td>25.7</td><td></td></tr><tr><td>(E)</td><td colspan="9">位置嵌入代替正弦函数</td><td>4.92</td><td>25.7</td><td></td></tr><tr><td>大型</td><td>6</td><td>1024</td><td>4096</td><td>16</td><td></td><td></td><td>0.3</td><td></td><td>300K</td><td>4.33</td><td>26.4</td><td>213</td></tr></tbody></table>


In Table 3 rows (B),we observe that reducing the attention key size ${d}_{k}$ hurts model quality. This suggests that determining compatibility is not easy and that a more sophisticated compatibility function than dot product may be beneficial. We further observe in rows (C) and (D) that, as expected, bigger models are better, and dropout is very helpful in avoiding over-fitting. In row (E) we replace our sinusoidal positional encoding with learned positional embeddings [8], and observe nearly identical results to the base model.
在表3第(B)行中，我们观察到减小注意力键的大小${d}_{k}$会损害模型质量。这表明确定相关性并非易事，且采用比点积更复杂的兼容性函数可能更为有利。我们在(C)行和(D)行中进一步观察到，正如预期的那样，模型越大效果越好，而dropout在避免过拟合方面非常有帮助。在第(E)行中，我们用学习到的位置嵌入[8]替换了正弦位置编码，并观察到与基础模型几乎相同的结果。


## 7 Conclusion
## 7 结论


In this work, we presented the Transformer, the first sequence transduction model based entirely on attention, replacing the recurrent layers most commonly used in encoder-decoder architectures with multi-headed self-attention.
在这项工作中，我们提出了Transformer，这是首个完全基于注意力的序列转导模型，用多头自注意力取代了编码器-解码器架构中最常用的循环层。


For translation tasks, the Transformer can be trained significantly faster than architectures based on recurrent or convolutional layers. On both WMT 2014 English-to-German and WMT 2014 English-to-French translation tasks, we achieve a new state of the art. In the former task our best model outperforms even all previously reported ensembles.
对于翻译任务，Transformer的训练速度显著快于基于循环或卷积层的架构。在WMT 2014英德和WMT 2014英法翻译任务中，我们都取得了新的最先进结果。在前者任务中，我们的最佳模型甚至超越了之前报道的所有集成模型。


We are excited about the future of attention-based models and plan to apply them to other tasks. We plan to extend the Transformer to problems involving input and output modalities other than text and to investigate local, restricted attention mechanisms to efficiently handle large inputs and outputs such as images, audio and video. Making generation less sequential is another research goals of ours.
我们对基于注意力的模型的未来感到兴奋，并计划将其应用于其他任务。我们计划将Transformer扩展到涉及文本以外输入和输出模态的问题，并研究局部、受限的注意力机制，以有效处理图像、音频和视频等大型输入和输出。使生成过程更具非序列性是我们的另一个研究目标。


The code we used to train and evaluate our models is available at https://github.com/ tensorflow/tensor2tensor
我们用于训练和评估模型的代码可在 https://github.com/tensorflow/tensor2tensor 获取


Acknowledgements We are grateful to Nal Kalchbrenner and Stephan Gouws for their fruitful comments, corrections and inspiration.
致谢 我们感谢Nal Kalchbrenner和Stephan Gouws提出的富有成效的意见、修正和灵感。


## References
## 参考文献


[1] Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E Hinton. Layer normalization. arXiv preprint arXiv:1607.06450, 2016.
[1] Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E Hinton. Layer normalization. arXiv preprint arXiv:1607.06450, 2016.


[2] Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio. Neural machine translation by jointly learning to align and translate. CoRR, abs/1409.0473, 2014.
[2] Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio. Neural machine translation by jointly learning to align and translate. CoRR, abs/1409.0473, 2014.


[3] Denny Britz, Anna Goldie, Minh-Thang Luong, and Quoc V. Le. Massive exploration of neural machine translation architectures. CoRR, abs/1703.03906, 2017.
[3] Denny Britz, Anna Goldie, Minh-Thang Luong, and Quoc V. Le. Massive exploration of neural machine translation architectures. CoRR, abs/1703.03906, 2017.


[4] Jianpeng Cheng, Li Dong, and Mirella Lapata. Long short-term memory-networks for machine reading. arXiv preprint arXiv:1601.06733, 2016.
[4] Jianpeng Cheng, Li Dong, and Mirella Lapata. Long short-term memory-networks for machine reading. arXiv preprint arXiv:1601.06733, 2016.


[5] Kyunghyun Cho, Bart van Merrienboer, Caglar Gulcehre, Fethi Bougares, Holger Schwenk, and Yoshua Bengio. Learning phrase representations using rnn encoder-decoder for statistical machine translation. CoRR, abs/1406.1078, 2014.
[5] Kyunghyun Cho, Bart van Merrienboer, Caglar Gulcehre, Fethi Bougares, Holger Schwenk, and Yoshua Bengio. Learning phrase representations using rnn encoder-decoder for statistical machine translation. CoRR, abs/1406.1078, 2014.


[6] Francois Chollet. Xception: Deep learning with depthwise separable convolutions. arXiv preprint arXiv:1610.02357, 2016.
[6] Francois Chollet. Xception: Deep learning with depthwise separable convolutions. arXiv preprint arXiv:1610.02357, 2016.


[7] Junyoung Chung, Caglar Gülgehre, Kyunghyun Cho, and Yoshua Bengio. Empirical evaluation of gated recurrent neural networks on sequence modeling. CoRR, abs/1412.3555, 2014.
[7] Junyoung Chung, Caglar Gülgehre, Kyunghyun Cho, and Yoshua Bengio. Empirical evaluation of gated recurrent neural networks on sequence modeling. CoRR, abs/1412.3555, 2014.


[8] Jonas Gehring, Michael Auli, David Grangier, Denis Yarats, and Yann N. Dauphin. Convolutional sequence to sequence learning. arXiv preprint arXiv:1705.03122v2, 2017.
[8] Jonas Gehring, Michael Auli, David Grangier, Denis Yarats, and Yann N. Dauphin. Convolutional sequence to sequence learning. arXiv preprint arXiv:1705.03122v2, 2017.


[9] Alex Graves. Generating sequences with recurrent neural networks. arXiv preprint arXiv:1308.0850, 2013.
[9] Alex Graves. 使用循环神经网络生成序列. arXiv 预印本 arXiv:1308.0850, 2013.


[10] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 770-778, 2016.
[10] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. 用于图像识别的深度残差学习. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 770-778, 2016.


[11] Sepp Hochreiter, Yoshua Bengio, Paolo Frasconi, and Jürgen Schmidhuber. Gradient flow in recurrent nets: the difficulty of learning long-term dependencies, 2001.
[11] Sepp Hochreiter, Yoshua Bengio, Paolo Frasconi, and Jürgen Schmidhuber. 循环网络中的梯度流：学习长期依赖的困难, 2001.


[12] Sepp Hochreiter and Jürgen Schmidhuber. Long short-term memory. Neural computation, 9(8):1735-1780, 1997.
[12] Sepp Hochreiter and Jürgen Schmidhuber. 长短期记忆. Neural computation, 9(8):1735-1780, 1997.


[13] Rafal Jozefowicz, Oriol Vinyals, Mike Schuster, Noam Shazeer, and Yonghui Wu. Exploring the limits of language modeling. arXiv preprint arXiv:1602.02410, 2016.
[13] Rafal Jozefowicz, Oriol Vinyals, Mike Schuster, Noam Shazeer, and Yonghui Wu. 探索语言建模的极限. arXiv 预印本 arXiv:1602.02410, 2016.


[14] Łukasz Kaiser and Ilya Sutskever. Neural GPUs learn algorithms. In International Conference on Learning Representations (ICLR), 2016.
[14] Łukasz Kaiser and Ilya Sutskever. 神经 GPU 学习算法. In International Conference on Learning Representations (ICLR), 2016.


[15] Nal Kalchbrenner, Lasse Espeholt, Karen Simonyan, Aaron van den Oord, Alex Graves, and Ko-ray Kavukcuoglu. Neural machine translation in linear time. arXiv preprint arXiv:1610.10099v2, 2017.
[15] Nal Kalchbrenner, Lasse Espeholt, Karen Simonyan, Aaron van den Oord, Alex Graves, and Ko-ray Kavukcuoglu. 线性时间内的神经机器翻译. arXiv 预印本 arXiv:1610.10099v2, 2017.


[16] Yoon Kim, Carl Denton, Luong Hoang, and Alexander M. Rush. Structured attention networks. In International Conference on Learning Representations, 2017.
[16] Yoon Kim, Carl Denton, Luong Hoang, and Alexander M. Rush. 结构化注意力网络. In International Conference on Learning Representations, 2017.


[17] Diederik Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In ICLR, 2015.
[17] Diederik Kingma and Jimmy Ba. Adam：一种随机优化方法. In ICLR, 2015.


[18] Oleksii Kuchaiev and Boris Ginsburg. Factorization tricks for LSTM networks. arXiv preprint arXiv:1703.10722, 2017.
[18] Oleksii Kuchaiev and Boris Ginsburg. LSTM 网络的因子分解技巧. arXiv 预印本 arXiv:1703.10722, 2017.


[19] Zhouhan Lin, Minwei Feng, Cicero Nogueira dos Santos, Mo Yu, Bing Xiang, Bowen Zhou, and Yoshua Bengio. A structured self-attentive sentence embedding. arXiv preprint arXiv:1703.03130, 2017.
[19] Zhouhan Lin, Minwei Feng, Cicero Nogueira dos Santos, Mo Yu, Bing Xiang, Bowen Zhou, and Yoshua Bengio. 一种结构化自注意力句子嵌入. arXiv 预印本 arXiv:1703.03130, 2017.


[20] Samy Bengio Łukasz Kaiser. Can active memory replace attention? In Advances in Neural Information Processing Systems, (NIPS), 2016.
[20] Samy Bengio Łukasz Kaiser. 主动记忆能否取代注意力？ In Advances in Neural Information Processing Systems, (NIPS), 2016.


[21] Minh-Thang Luong, Hieu Pham, and Christopher D Manning. Effective approaches to attention-based neural machine translation. arXiv preprint arXiv: 1508.04025, 2015.
[21] Minh-Thang Luong, Hieu Pham, and Christopher D Manning. 基于注意力的神经机器翻译的有效方法. arXiv 预印本 arXiv: 1508.04025, 2015.


[22] Ankur Parikh, Oscar Täckström, Dipanjan Das, and Jakob Uszkoreit. A decomposable attention model. In Empirical Methods in Natural Language Processing, 2016.
[22] Ankur Parikh, Oscar Täckström, Dipanjan Das, and Jakob Uszkoreit. 一种可分解的注意力模型. In Empirical Methods in Natural Language Processing, 2016.


[23] Romain Paulus, Caiming Xiong, and Richard Socher. A deep reinforced model for abstractive summarization. arXiv preprint arXiv:1705.04304, 2017.
[23] Romain Paulus, Caiming Xiong, and Richard Socher. 一种用于抽象性摘要的深度强化模型. arXiv 预印本 arXiv:1705.04304, 2017.


[24] Ofir Press and Lior Wolf. Using the output embedding to improve language models. arXiv preprint arXiv:1608.05859, 2016.
[24] Ofir Press and Lior Wolf. 利用输出嵌入改进语言模型. arXiv 预印本 arXiv:1608.05859, 2016.


[25] Rico Sennrich, Barry Haddow, and Alexandra Birch. Neural machine translation of rare words with subword units. arXiv preprint arXiv:1508.07909, 2015.
[25] Rico Sennrich, Barry Haddow, and Alexandra Birch. 通过分词单元实现罕见词的神级网络机器翻译. arXiv preprint arXiv:1508.07909, 2015.


[26] Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton, and Jeff Dean. Outrageously large neural networks: The sparsely-gated mixture-of-experts layer. arXiv preprint arXiv:1701.06538, 2017.
[26] Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton, and Jeff Dean. 超大型神经网络：稀疏门控混合专家层. arXiv preprint arXiv:1701.06538, 2017.


[27] Nitish Srivastava, Geoffrey E Hinton, Alex Krizhevsky, Ilya Sutskever, and Ruslan Salakhutdi-nov. Dropout: a simple way to prevent neural networks from overfitting. Journal of Machine Learning Research, 15(1):1929-1958, 2014.
[27] Nitish Srivastava, Geoffrey E Hinton, Alex Krizhevsky, Ilya Sutskever, and Ruslan Salakhutdi-nov. Dropout：一种防止神经网络过拟合的简单方法. Journal of Machine Learning Research, 15(1):1929-1958, 2014.


[28] Sainbayar Sukhbaatar, arthur szlam, Jason Weston, and Rob Fergus. End-to-end memory networks. In C. Cortes, N. D. Lawrence, D. D. Lee, M. Sugiyama, and R. Garnett, editors, Advances in Neural Information Processing Systems 28, pages 2440-2448. Curran Associates, Inc., 2015.
[28] Sainbayar Sukhbaatar, arthur szlam, Jason Weston, and Rob Fergus. 端到端记忆网络. In C. Cortes, N. D. Lawrence, D. D. Lee, M. Sugiyama, and R. Garnett, editors, Advances in Neural Information Processing Systems 28, pages 2440-2448. Curran Associates, Inc., 2015.


[29] Ilya Sutskever, Oriol Vinyals, and Quoc VV Le. Sequence to sequence learning with neural networks. In Advances in Neural Information Processing Systems, pages 3104-3112, 2014.
[29] Ilya Sutskever, Oriol Vinyals, and Quoc VV Le. 基于神经网络的序列到序列学习. In Advances in Neural Information Processing Systems, pages 3104-3112, 2014.


[30] Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jonathon Shlens, and Zbigniew Wojna. Rethinking the inception architecture for computer vision. CoRR, abs/1512.00567, 2015.
[30] Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jonathon Shlens, and Zbigniew Wojna. 重新思考计算机视觉的 Inception 架构. CoRR, abs/1512.00567, 2015.


[31] Yonghui Wu, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Wolfgang Macherey, Maxim Krikun, Yuan Cao, Qin Gao, Klaus Macherey, et al. Google's neural machine translation system: Bridging the gap between human and machine translation. arXiv preprint arXiv:1609.08144, 2016.
[31] Yonghui Wu, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Wolfgang Macherey, Maxim Krikun, Yuan Cao, Qin Gao, Klaus Macherey, et al. Google 神经网络机器翻译系统：弥合人类与机器翻译之间的差距. arXiv preprint arXiv:1609.08144, 2016.


[32] Jie Zhou, Ying Cao, Xuguang Wang, Peng Li, and Wei Xu. Deep recurrent models with fast-forward connections for neural machine translation. CoRR, abs/1606.04199, 2016.
[32] Jie Zhou, Ying Cao, Xuguang Wang, Peng Li, and Wei Xu. 具有快速前馈连接的深度循环神经网络机器翻译模型. CoRR, abs/1606.04199, 2016.