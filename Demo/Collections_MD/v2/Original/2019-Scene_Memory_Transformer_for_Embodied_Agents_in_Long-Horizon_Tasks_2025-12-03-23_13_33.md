# Scene Memory Transformer for Embodied Agents in Long-Horizon Tasks
# 用于长时任务中具身智能体的场景记忆Transformer


Kuan Fang1 Alexander Toshev2 Li Fei-Fei ${}^{1}$ Silvio Savarese1
方宽1 亚历山大·托舍夫2 李菲菲 ${}^{1}$ 西尔维奥·萨瓦雷塞1


${}^{1}$ Stanford University ${}^{2}$ Google Brain
${}^{1}$ 斯坦福大学 ${}^{2}$ 谷歌大脑


## Abstract
## 摘要


Many robotic applications require the agent to perform long-horizon tasks in partially observable environments. In such applications, decision making at any step can depend on observations received far in the past. Hence, being able to properly memorize and utilize the long-term history is crucial. In this work, we propose a novel memory-based policy, named Scene Memory Transformer (SMT). The proposed policy embeds and adds each observation to a memory and uses the attention mechanism to exploit spatio-temporal dependencies. This model is generic and can be efficiently trained with reinforcement learning over long episodes. On a range of visual navigation tasks, SMT demonstrates superior performance to existing reactive and memory-based policies by a margin.
许多机器人应用要求智能体在部分可观测的环境中执行长时任务。在这类应用中，任何一步的决策都可能依赖于很久之前接收到的观测信息。因此，能够正确地记忆和利用长期历史信息至关重要。在这项工作中，我们提出了一种基于记忆的新型策略，名为场景记忆Transformer（SMT）。该策略将每次观测信息嵌入并添加到记忆中，并利用注意力机制挖掘时空依赖关系。该模型具有通用性，可通过强化学习在长时情节上进行高效训练。在一系列视觉导航任务中，SMT的性能显著优于现有的反应式和基于记忆的策略。


## 1. Introduction
## 1. 引言


Autonomous agents, controlled by neural network policies and trained with reinforcement learning algorithms, have been used in a wide range of robot navigation applications [1, 2, 3, 22, 27, 31, 46, 49, 50]. In many of these applications, the agent needs to perform tasks over long time horizons in unseen environments. Consider a robot patrolling or searching for an object in a large unexplored building. Typically, completing such tasks requires the robot to utilize the received observation at each step and to grow its knowledge of the environment, e.g. building structures, object arrangements, explored area , etc. Therefore, it is crucial for the agent to maintain a detailed memory of past observations and actions over the task execution.
由神经网络策略控制并通过强化学习算法训练的自主智能体已广泛应用于机器人导航领域 [1, 2, 3, 22, 27, 31, 46, 49, 50]。在许多此类应用中，智能体需要在未知环境中长时间执行任务。例如，让机器人在一座未探索过的大型建筑中巡逻或寻找某个物体。通常，完成这类任务需要机器人利用每一步接收到的观测信息，并不断积累对环境的认知，如建筑结构、物体布局、已探索区域等。因此，智能体在执行任务过程中保持对过去观测和动作的详细记忆至关重要。


The most common way of endowing an agent's policy with memory is to use recurrent neural networks (RNNs), with LSTM [5] as a popular choice. An RNN stores the information in a fixed-size state vector by combining the input observation with the state vector at each time step. The policy outputs actions for the agent to take given the updated state vector. Unfortunately, however, RNNs often fail to capture long-term dependencies [33].
为智能体策略赋予记忆能力最常见的方法是使用循环神经网络（RNN），其中长短期记忆网络（LSTM） [5] 是常用选择。RNN通过在每个时间步将输入观测信息与状态向量相结合，将信息存储在固定大小的状态向量中。策略根据更新后的状态向量为智能体输出要执行的动作。然而，遗憾的是，RNN往往无法捕捉长期依赖关系 [33]。


To enhance agent's ability to plan and reason, neural network policies with external memories have been proposed [31, 48]. Such memory-based policies have been primarily studied in the context of robot navigation in partially observable environments, where the neural network learns to encode the received observations and write them into a map-like memory [15, 16, 18, 32, 43]. Despite their superior performance compared to reactive and RNN policies, existing memory-based policies suffer from limited flexibility and scalability. Specifically, strong domain-specific inductive biases go into the design of such memories, e.g. 2D layout of the environment, predefined size of this layout, geometry-based memory updates, etc. Meanwhile, RNNs are usually critical components for these memory-based policies for exploiting spatio-temporal dependencies. Thus they still suffer from the drawbacks of RNN models.
为增强智能体的规划和推理能力，人们提出了带有外部记忆的神经网络策略 [31, 48]。这类基于记忆的策略主要在部分可观测环境下的机器人导航场景中进行研究，其中神经网络学习对接收的观测信息进行编码，并将其写入类似地图的记忆中 [15, 16, 18, 32, 43]。尽管与反应式和RNN策略相比，它们表现更优，但现有的基于记忆的策略存在灵活性和可扩展性有限的问题。具体而言，这类记忆的设计中存在很强的特定领域归纳偏置，例如环境的二维布局、该布局的预定义大小、基于几何的记忆更新等。同时，RNN通常是这些基于记忆的策略挖掘时空依赖关系的关键组件。因此，它们仍然存在RNN模型的缺点。


<img src="https://cdn.noedgeai.com/bo_d4nfr977aajc73frs8r0_0.jpg?x=916&y=602&w=681&h=215&r=0"/>



Figure 1. The Scene Memory Transformer (SMT) policy embeds and adds each observation to a memory. Given the current observation, the attention mechanism is applied over the memory to produce an action. SMT is demonstrated successfully in several visual navigation tasks, all of which has long time horizons.
图1. 场景记忆Transformer（SMT）策略将每次观测信息嵌入并添加到记忆中。给定当前观测信息，对记忆应用注意力机制以产生动作。SMT已在多个长时视觉导航任务中成功应用。


In this work, we present Scene Memory Transformer (SMT), a memory-based policy using attention mechanisms, for understanding partially observable environments in long-horizon robot tasks. This policy is inspired by the Transformer model [42], which has been successfully applied to multiple natural language processing problems recently. As shown in Fig. 1, SMT consists of two modules: a scene memory which embeds and stores all encountered observations and a policy network which uses attention mechanism over the scene memory to produce an action.
在这项工作中，我们提出了场景记忆Transformer（SMT），这是一种使用注意力机制的基于记忆的策略，用于理解长时机器人任务中的部分可观测环境。该策略的灵感来源于Transformer模型 [42]，该模型近期已成功应用于多个自然语言处理问题。如图1所示，SMT由两个模块组成：一个用于嵌入和存储所有遇到的观测信息的场景记忆模块，以及一个对场景记忆应用注意力机制以产生动作的策略网络模块。


The proposed SMT policy is different from existing methods in terms of how to utilize observations received in the previous steps. Instead of combining past observations into a single state vector, as commonly done by RNN policies, SMT separately embeds the observations for each time step in the scene memory. In contrast to most existing memory models, the scene memory is simply a set of all embedded observations and any decisions of aggregating the stored information are deferred to a later point. We argue that this is a crucial property in long-horizon tasks where computation of action at a specific time step could depend on any provided information in the past, which might not be properly captured in a state vector or map-like memory. The policy network in SMT adopts attention mechanisms instead of RNNs to aggregate the visual and geometric information from the scene memory. This network efficiently learns to utilize the stored information and scales well with the time horizon. As a result, SMT effectively exploits long-term spatio-temporal dependencies without committing to a environment structure in the model design.
所提出的SMT策略在如何利用先前步骤接收到的观测信息方面与现有方法不同。与RNN策略通常将过去的观测信息合并为单个状态向量不同，SMT在场景记忆中分别嵌入每个时间步的观测信息。与大多数现有的记忆模型相比，场景记忆只是所有嵌入观测信息的集合，任何对存储信息进行聚合的决策都推迟到后续阶段。我们认为，在长时任务中这是一个关键特性，因为特定时间步的动作计算可能依赖于过去提供的任何信息，而这些信息可能无法在状态向量或类似地图的记忆中得到妥善捕捉。SMT中的策略网络采用注意力机制而非RNN来聚合场景记忆中的视觉和几何信息。该网络能够高效地学习利用存储的信息，并且在时间跨度上具有良好的扩展性。因此，SMT在模型设计中无需预设环境结构，即可有效挖掘长期时空依赖关系。


Although the scene memory grows linearly with the length of the episode, it stores only an embedding vector at each steps. Therefore, we can easily store hundreds of observations without any burden in the device memory. This overhead is justified as it gives us higher performance compared to established policies with more compact memories.
虽然场景记忆随情节长度线性增长，但在每一步只存储一个嵌入向量。因此，我们可以在设备内存中轻松存储数百个观测而无负担。这样的开销是合理的，因为它相比内存更紧凑的既有策略带来了更高的性能。


Further, as the computational complexity of the original model grows quadratically with the size of the scene memory, we introduce a memory factorization procedure as part of SMT. This reduces the computational complexity to linear. The procedure is applied when the number of the stored observations is high. In this way, we can leverage a large memory capacity without the taxing computational overhead of the original model.
此外，由于原始模型的计算复杂度随场景记忆大小呈二次增长，我们在SMT中引入了记忆分解程序。该方法将计算复杂度降为线性。当存储的观测数量高时应用此程序。这样，我们可以利用大容量记忆而无需承受原始模型的高昂计算开销。


The advantages of the proposed SMT are empirically verified on three long-horizon visual navigation tasks: roaming, coverage and search. We train the SMT policy using deep Q-learning [29] and thus demonstrate for the first time how attention mechanisms introduced in [42] can boost the task performance in a reinforcement learning setup. In these tasks, SMT considerably and consistently outperforms existing reactive and memory-based policies. Videos can found at https://sites.google.com/ view/scene-memory-transformer
所提出的SMT的优点在三个长时视覺导航任务上得到了实证验证：漫游、覆盖和搜寻。我们使用深度Q学习[29]训练SMT策略，从而首次展示了[42]中引入的注意力机制如何在强化学习设置下提升任务性能。在这些任务中，SMT显著且稳定地优于现有的反应式和基于记忆的策略。视频可见于 https://sites.google.com/ view/scene-memory-transformer


## 2. Related Work
## 2. Related Work


Memory-based policy using RNN. Policies using RNNs have been extensively studied in reinforcement learning settings for robot navigation and other tasks. The most common architectural choice is LSTM [19]. For example, Mirowski et al. [27] train an A3C [28] agent to navigate in synthesized mazes with an LSTM policy. Wu et al. [45] use a gated-LSTM policy with multi-modal inputs trained for room navigation. Moursavian et al. [30] use an LSTM policy for target driven navigation. The drawbacks of RNNs are mainly two-fold. First, merging all past observations into a single state vector of fixed size can easily lose useful information. Second, RNNs have optimization difficulties over long sequences [33, 41] in backpropagation through time (BPTT). In contrast, our model stores each observations separately in the memory and only aggregate the information when computing an action. And it extracts spatio-temporal dependencies using attention mechanisms, thereby it is not handicapped by the challenges of BPTT.
使用RNN的基于记忆的策略。使用RNN的策略在机器人导航等强化学习设置中被广泛研究。最常见的架构选择是LSTM[19]。例如，Mirowski 等人[27]训练了一个带LSTM策略的A3C[28]代理在合成迷宫中导航。Wu 等人[45]用带门控的LSTM策略和多模态输入训练用于房间导航。Moursavian 等人[30]为目标驱动导航使用LSTM策略。RNN的缺点主要有两点。首先，将所有过往观测合并为固定尺寸的单一状态向量容易丢失有用信息。其次，RNN在长序列的反向传播通过时间（BPTT）中存在优化困难[33, 41]。相比之下，我们的模型将每个观测单独存储在记忆中，并仅在计算动作时聚合信息。它使用注意力机制提取时空依赖，因此不受BPTT挑战的制约。


External memory. Memory models have been extensively studied in natural language processing for variety of tasks such as translation [42], question answering [38], summarization [26]. Such models are fairly generic, mostly based on attention functions and designed to deal with input data in the format of long sequences or large sets .
外部记忆。记忆模型在自然语言处理的多种任务中被广泛研究，如翻译[42]、问答[38]、摘要[26]。此类模型相当通用，多基于注意力函数，旨在处理长序列或大集合格式的输入数据。


When it comes to autonomous agents, most of the approaches structure the memory as a 2D grid. They are applied to visual navigation $\left\lbrack  {{15},{16},{32},{47}}\right\rbrack$ ,interactive question answering [13], and localization [7, 18, 20]. These methods exhibit certain rigidity. For instance, the 2D layout is of fixed size and same amount of memory capacity is allocated to each part of the environment. Henriques et al. [18] designs a differentiable mapping module with 2.5D representation of the spatial structure. Such a structured memory necessiates write operations, which compress all observations as the agent executes a task and potentially can lose information which could be useful later in the task execution. On the contrary, our SMT keeps all embedded observations and allows for the policy to attend to them as needed at any step. Further, the memory operations in the above papers are based on current estimate of robot localization, where the memory is being modified and how it is accessed. In contrast, we keep all pose information in its original form, thus allow for potentially more flexible use.
在自主代理方面，大多数方法将记忆结构化为二维网格。它们被应用于视觉导航$\left\lbrack  {{15},{16},{32},{47}}\right\rbrack$、交互式问答[13]和定位[7, 18, 20]。这些方法表现出一定的僵化性。例如，二维布局尺寸固定，环境每个部分分配相同的记忆容量。Henriques 等人[18]设计了具有二维半表示（2.5D）空间结构的可微分映射模块。这类结构化记忆需要写操作，在代理执行任务时压缩所有观测，可能会丢失后来任务执行中有用的信息。相反，我们的SMT保留所有嵌入观测，并允许策略在任一步按需关注它们。此外，上述论文中的记忆操作基于当前的机器人定位估计，从而修改记忆及其访问方式。相反，我们保留所有位姿信息的原始形式，从而允许更灵活的使用可能性。


A more generic view on memory for autonomous agents has been less popular. Savinov et al. [35] construct a topol-gical map of the environment, and uses it for planning. Oh at al. [31] use a single-layer attention decoder for control problems. However, the method relies on an LSTM as a memory controller, which comes with the challenges of backpropgation through time. Khan et al. [23] apply the very general Differentiable Neural Computer [14] to control problems. While this approach is hard to optimize and is applied on very simple navigation tasks.
关于自主代理的更通用记忆视角不太流行。Savinov 等人[35]构建了环境的拓扑地图，并用于规划。Oh 等人[31]在控制问题上使用了单层注意力解码器。然而，该方法依赖LSTM作为记忆控制器，因而面临通过时间反向传播的挑战。Khan 等人[23]将非常通用的可微分神经计算机（DNC）[14]应用于控制问题，但该方法难以优化且仅用于非常简单的导航任务。


Visual Navigation. We apply SMT on a set of visual navigation tasks, which have a long history in computer vision and robotics [6, 10, 39]. Our approach falls into visual navigation, where the agent does not have any scene-specific information about the environment $\lbrack 8,9,{36},{40}$ , 44]. As in recent works on end-to-end training policies for navigation tasks $\left\lbrack  {2,{22},{27},{46},{49},{50}}\right\rbrack$ ,our model does not need a map of the environment provided beforehand. While [22, 27] evaluates their models in 3D mazes, our model can handle more structured environments as realistic cluttered indoor scenes composed of multiple rooms. In contrast to $\left\lbrack  {{27},{49},{50}}\right\rbrack$ which train the policy for one or several known scenes, our trained model can generalize to unseen houses.
视觉导航。我们将场景记忆变换器（SMT）应用于一组视觉导航任务，这些任务在计算机视觉和机器人领域有着悠久的历史 [6, 10, 39]。我们的方法属于视觉导航范畴，即智能体没有关于环境的任何特定场景信息 $\lbrack 8,9,{36},{40}$ , 44]。与近期关于导航任务端到端训练策略的工作 $\left\lbrack  {2,{22},{27},{46},{49},{50}}\right\rbrack$ 一样，我们的模型不需要事先提供环境地图。虽然文献 [22, 27] 在 3D 迷宫中评估其模型，但我们的模型可以处理更具结构化的环境，例如由多个房间组成的真实杂乱室内场景。与 $\left\lbrack  {{27},{49},{50}}\right\rbrack$ 为一个或几个已知场景训练策略不同，我们训练的模型可以泛化到未见过的房屋。


<img src="https://cdn.noedgeai.com/bo_d4nfr977aajc73frs8r0_2.jpg?x=162&y=223&w=1424&h=554&r=0"/>



Figure 2. The Scene Memory Transformer (SMT) policy. At each time step $t$ ,the observation ${\mathbf{o}}_{t}$ is embedded and added to the scene memory. SMT has access to the full memory and produces an action according to the current observation.
图 2. 场景记忆变换器（SMT）策略。在每个时间步 $t$，观测值 ${\mathbf{o}}_{t}$ 被嵌入并添加到场景记忆中。SMT 可以访问完整的记忆，并根据当前观测值生成一个动作。


## 3. Method
## 3. 方法


In this section, we first describe the problem setup. Then we introduce the Scene Memory Transformer (SMT) and its variations as shown in Fig. 2.
在本节中，我们首先描述问题设置。然后介绍场景记忆变换器（SMT）及其变体，如图 2 所示。


### 3.1. Problem Setup
### 3.1. 问题设置


We are interested in a variety of tasks which require an embodied agent to navigate in unseen environments to achieve the task goal. These tasks can be formulated as the Partially Observable Markov Decision Process (POMDP) [21] $\left( {\mathcal{S},\mathcal{A},\mathcal{O},R\left( {s,a}\right) ,T\left( {{s}^{\prime } \mid  s,a}\right) ,P\left( {o \mid  s}\right) }\right)$ where $\mathcal{S},\mathcal{A},\mathcal{O}$ are state,action and observation spaces, $R\left( {s,a}\right)$ is the reward function, $T\left( {{s}^{\prime } \mid  s,a}\right)$ and $P\left( {o \mid  s}\right)$ are transition and observation probabilities.
我们对各种需要具身智能体在未知环境中导航以实现任务目标的任务感兴趣。这些任务可以被表述为部分可观测马尔可夫决策过程（POMDP） [21] $\left( {\mathcal{S},\mathcal{A},\mathcal{O},R\left( {s,a}\right) ,T\left( {{s}^{\prime } \mid  s,a}\right) ,P\left( {o \mid  s}\right) }\right)$，其中 $\mathcal{S},\mathcal{A},\mathcal{O}$ 是状态、动作和观测空间，$R\left( {s,a}\right)$ 是奖励函数，$T\left( {{s}^{\prime } \mid  s,a}\right)$ 和 $P\left( {o \mid  s}\right)$ 是转移和观测概率。


The observation is a tuple $o = \left( {I,p,{a}_{\text{ prev }}}\right)  \in  \mathcal{O}$ composed of multiple modalities. I represents the visual data consisting of an RGB image, a depth image and a semantic segmentation mask obtained from a camera sensor mounted on the robot. $p$ is the agent pose w.r.t. the starting pose of the episode,estimated or given by the environment. ${a}_{\text{ prev }}$ is the action taken at the previous time step.
观测值是一个由多种模态组成的元组 $o = \left( {I,p,{a}_{\text{ prev }}}\right)  \in  \mathcal{O}$。I 表示视觉数据，包括从安装在机器人上的相机传感器获取的 RGB 图像、深度图像和语义分割掩码。$p$ 是智能体相对于该回合起始姿态的姿态，由环境估计或给定。${a}_{\text{ prev }}$ 是上一个时间步采取的动作。


In our setup, we adopt a discrete action space defined as $\mathcal{A} =$ \{go_forward,turn_left,turn_right\},a common choice for navigation problems operating on a flat surface. Note that these actions are executed under noisy dynamics modeled by $P\left( {{s}^{\prime } \mid  s,a}\right)$ ,so the state space is continuous.
在我们的设置中，我们采用一个离散动作空间，定义为 $\mathcal{A} =$ \{前进，左转，右转\}，这是在平面上操作的导航问题的常见选择。请注意，这些动作是在由 $P\left( {{s}^{\prime } \mid  s,a}\right)$ 建模的噪声动力学下执行的，因此状态空间是连续的。


While we share the same $\mathcal{O}$ and $\mathcal{A}$ across tasks and environments, each task is defined by a different reward function $R\left( {s,a}\right)$ as described in Sec. 4.1. The policy for each task is trained to maximize the expected return, defined as the cumulative reward ${\mathbb{E}}_{\tau }\left\lbrack  {\mathop{\sum }\limits_{t}R\left( {{s}_{t},{a}_{t}}\right) }\right\rbrack$ over trajectories $\tau  = {\left( {s}_{t},{a}_{t}\right) }_{t = 1}^{H}$ of time horizon $H$ unrolled by the policy.
虽然我们在不同任务和环境中共享相同的 $\mathcal{O}$ 和 $\mathcal{A}$，但每个任务由不同的奖励函数 $R\left( {s,a}\right)$ 定义，如第 4.1 节所述。每个任务的策略被训练以最大化期望回报，期望回报定义为策略展开的时间范围为 $H$ 的轨迹 $\tau  = {\left( {s}_{t},{a}_{t}\right) }_{t = 1}^{H}$ 上的累积奖励 ${\mathbb{E}}_{\tau }\left\lbrack  {\mathop{\sum }\limits_{t}R\left( {{s}_{t},{a}_{t}}\right) }\right\rbrack$。


### 3.2. Scene Memory Transformer
### 3.2. 场景记忆变换器


The SMT policy, as outlined in Fig. 2, consists of two modules. The first module is the scene memory $M$ which stores all past observations in an embedded form. This memory is updated at each time step. The second module, denoted by $\pi \left( {a \mid  o,M}\right)$ ,is an attention-based policy network that uses the updated scene memory to compute an distribution over actions.
如图 2 所示，SMT 策略由两个模块组成。第一个模块是场景记忆 $M$，它以嵌入形式存储所有过去的观测值。这个记忆在每个时间步更新。第二个模块，用 $\pi \left( {a \mid  o,M}\right)$ 表示，是一个基于注意力的策略网络，它使用更新后的场景记忆来计算动作分布。


In a nutshell, the model and its interaction with the environment at time $t$ can be summarized as:
简而言之，模型及其在时间 $t$ 与环境的交互可归纳为：


$$
{o}_{t} \sim  P\left( {{o}_{t} \mid  {s}_{t}}\right)
$$



$$
{M}_{t} = \operatorname{Update}\left( {{M}_{t - 1},{o}_{t}}\right)
$$



$$
{a}_{t} \sim  \pi \left( {{a}_{t} \mid  {o}_{t},{M}_{t}}\right)
$$



$$
{s}_{t + 1} \sim  T\left( {{s}_{t + 1} \mid  {s}_{t},{a}_{t}}\right)
$$



In the following we define the above modules.
下文中我们对上述模块进行定义。


#### 3.2.1 Scene Memory
#### 3.2.1 场景记忆


The scene memory $M$ is intended to store all past observations in an embedded form. It is our intent not to endow it with any geometric structure, but to keep it as generic as possible. Moreover, we would like to avoid any loss of information when writing to $M$ and provide the policy with all available information from the history. So we separately keep observations of each step in the memory instead of merging them into a single state vector as in an RNN.
场景记忆 $M$ 用于以嵌入形式存储所有过去的观测。我们意在不赋予其任何几何结构，而尽可能保持通用性。此外，我们希望在写入 $M$ 时避免任何信息丢失，并向策略提供历史中所有可用信息。因此我们在记忆中单独保留每一步的观测，而不是像RNN那样将它们合并为单一状态向量。


The scene memory can be defined recursively as follows. Initially it is set to the empty set $\varnothing$ . At the current step,given an observation $o = \left( {I,p,{a}_{\text{ prev }}}\right)$ ,as defined in Sec. 3.1, we first embed all observation modalities, concatenate them, and apply a fully-connected layer FC:
场景记忆可递归定义如下。初始化时设为空集 $\varnothing$ 。在当前步骤，给定在第3.1节定义的观测 $o = \left( {I,p,{a}_{\text{ prev }}}\right)$ ，我们首先对所有观测模态进行嵌入，串联它们，并应用一个全连接层 FC：


$$
\psi \left( o\right)  = \mathrm{{FC}}\left( \left\{  {{\phi }_{I}\left( I\right) ,{\phi }_{p}\left( p\right) ,{\phi }_{a}\left( {a}_{\text{ prev }}\right) }\right\}  \right) \tag{1}
$$



<img src="https://cdn.noedgeai.com/bo_d4nfr977aajc73frs8r0_3.jpg?x=155&y=201&w=673&h=384&r=0"/>



Figure 3. Encoder without memory factorization, encoder with memory factorization, and decoder as in Sec. 3.2.2.
图3。无记忆分解的编码器、具有记忆分解的编码器，以及如第3.2.2节所述的解码器。


where ${\phi }_{I},{\phi }_{p},{\phi }_{a}$ are embedding networks for each modality as defined in Sec. 3.4. To obtain the memory for the next step,we update it by adding $\psi \left( o\right)$ to the set:
其中 ${\phi }_{I},{\phi }_{p},{\phi }_{a}$ 是第3.4节中为每种模态定义的嵌入网络。为得到下一步的记忆，我们通过将 $\psi \left( o\right)$ 添加到集合来更新它：


$$
\text{ Update }\left( {M,o}\right)  = M \cup  \{ \psi \left( o\right) \} \tag{2}
$$



The above memory grows linearly with the episode length. As each received observation is embedded into low-dimensional vectors in our design, one can easily store hundreds of time steps on the hardware devices. While RNNs are restricted to a fixed-size state vector, which usually can only capture short-term dependencies.
上述记忆随情节长度线性增长。由于我们设计中每个接收的观测都被嵌入为低维向量，因此可在硬件上轻松存储数百个时间步。而RNN受限于固定大小的状态向量，通常只能捕捉短期依赖。


#### 3.2.2 Attention-based Policy Network
#### 3.2.2 基于注意力的策略网络


The policy network $\pi \left( {a \mid  o,M}\right)$ uses the current observation and the scene memory to compute a distribution over the action space. As shown in Fig. 2, we first encode the memory by transforming each memory element in the context of all other elements. This step has the potential to capture the spatio-temporal dependencies in the environment. Then, we decode an action according to the current observation, using the encoded memory as the context.
策略网络 $\pi \left( {a \mid  o,M}\right)$ 使用当前观测和场景记忆来计算动作空间上的分布。如图2所示，我们首先通过在所有其他元素的语境中变换每个记忆元素来编码记忆。这一步有潜力捕捉环境中的时空依赖。然后，我们根据当前观测解码一个动作，使用编码后的记忆作为语境。


Attention Mechanism. Both encoding and decoding are defined using attention mechanisms, as detailed by [42]. In its general form,the attention function Att applies ${n}_{1}$ attention queries $U \in  {\mathbb{R}}^{{n}_{1} \times  {d}_{k}}$ over ${n}_{2}$ values $V \in  {\mathbb{R}}^{{n}_{2} \times  {d}_{v}}$ with associated keys $K \in  {\mathbb{R}}^{{n}_{2} \times  {d}_{k}}$ ,where ${d}_{k}$ and ${d}_{v}$ are dimensions of keys and values. The output of Att has ${n}_{1}$ elements of dimension ${d}_{v}$ ,defined as a weighted sum of the values, where the weights are based on dot-product similarity between the queries and the keys:
注意力机制。编码与解码均使用注意力机制定义，详见[42]。在一般形式中，注意力函数 Att 对 ${n}_{1}$ 个注意力查询 $U \in  {\mathbb{R}}^{{n}_{1} \times  {d}_{k}}$ 在 ${n}_{2}$ 个值 $V \in  {\mathbb{R}}^{{n}_{2} \times  {d}_{v}}$ 上施加注意力，值具有对应的键 $K \in  {\mathbb{R}}^{{n}_{2} \times  {d}_{k}}$，其中 ${d}_{k}$ 和 ${d}_{v}$ 是键和值的维度。Att 的输出有 ${n}_{1}$ 个维度为 ${d}_{v}$ 的元素，定义为值的加权和，权重基于查询与键之间的点积相似度：


$$
\operatorname{Att}\left( {U,K,V}\right)  = \operatorname{softmax}\left( {U{K}^{T}}\right) V \tag{3}
$$



An attention block AttBlock is built upon the above function and takes two inputs $X \in  {\mathbb{R}}^{{n}_{1} \times  {d}_{x}}$ and $Y \in  {\mathbb{R}}^{{n}_{2} \times  {d}_{y}}$ of dimension ${d}_{x}$ and ${d}_{y}$ respectively. It projects $X$ to the queries and $Y$ to the key-value pairs. It consists of two residual layers. The first is applied to the above Att and the second is applied to a fully-connected layer:
注意力模块 AttBlock 建立在上述函数之上，接受两个输入 $X \in  {\mathbb{R}}^{{n}_{1} \times  {d}_{x}}$ 和 $Y \in  {\mathbb{R}}^{{n}_{2} \times  {d}_{y}}$，其维度分别为 ${d}_{x}$ 和 ${d}_{y}$。它将 $X$ 投影为查询，将 $Y$ 投影为键-值对。它由两个残差层组成。第一个应用于上述 Att，第二个应用于一个全连接层：


$$
\operatorname{AttBlock}\left( {X,Y}\right)  = \operatorname{LN}\left( {\mathrm{{FC}}\left( H\right)  + H}\right) \tag{4}
$$



$$
\text{ where }H = \operatorname{LN}\left( {\operatorname{Att}\left( {X{W}^{U},Y{W}^{K},Y{W}^{V}}\right)  + X}\right)
$$



where ${W}^{U} \in  {\mathbb{R}}^{{d}_{x} \times  {d}_{k}},{W}^{K} \in  {\mathbb{R}}^{{d}_{y} \times  {d}_{k}}$ and ${W}^{V} \in  {\mathbb{R}}^{{d}_{y} \times  {d}_{v}}$ are projection matrices and LN stands for layer normalization [4]. We choose ${d}_{v} = {d}_{x}$ for the residual layer.
其中 ${W}^{U} \in  {\mathbb{R}}^{{d}_{x} \times  {d}_{k}},{W}^{K} \in  {\mathbb{R}}^{{d}_{y} \times  {d}_{k}}$ 和 ${W}^{V} \in  {\mathbb{R}}^{{d}_{y} \times  {d}_{v}}$ 是投影矩阵，LN 表示层归一化 [4]。我们为残差层选择 ${d}_{v} = {d}_{x}$。


Encoder. As in [42], our SMT model uses self-attention to encode the memory $M$ . More specifically,we use $M$ as both inputs of the attention block. As shown in Fig. 3, this transforms each embedded observation by using its relations to other past observations:
编码器。与文献[42]类似，我们的 SMT 模型使用自注意力对记忆 $M$ 进行编码。更具体地，我们将 $M$ 同时作为注意力模块的输入。如图3所示，这通过利用每个嵌入观测与其他过去观测的关系来变换该观测：


$$
\operatorname{Encoder}\left( M\right)  = \operatorname{AttBlock}\left( {M,M}\right) \tag{5}
$$



In this way, the model extracts the spatio-temporal dependencies in the memory.
通过这种方式，模型提取记忆中的时空依赖关系。


Decoder. The decoder is supposed to produce actions based on the current observation given the context $C$ ,which in our model is the encoded memory. As shown in Fig. 3, it applies similar machinery as the encoder, with the notable difference that the query in the attention layer is the embedding of the current observation $\psi \left( o\right)$ :
解码器。解码器应在给定上下文 $C$（在我们的模型中为编码后的记忆）的情况下基于当前观测生成动作。如图3所示，它应用与编码器类似的机制，显著不同之处在于注意力层中的查询是当前观测的嵌入 $\psi \left( o\right)$：


$$
\operatorname{Decoder}\left( {o,C}\right)  = \operatorname{AttBlock}\left( {\psi \left( o\right) ,C}\right) \tag{6}
$$



The final SMT output is a probability distribution over the action space $\mathcal{A}$ :
最终的 SMT 输出是在动作空间上的概率分布 $\mathcal{A}$：


$$
\pi \left( {a \mid  o,M}\right)  = \operatorname{Cat}\left( {\operatorname{softmax}\left( Q\right) }\right) \tag{7}
$$



$$
\text{ where }Q = \mathrm{{FC}}\left( {\mathrm{{FC}}\left( {\operatorname{Decoder}\left( {o,\operatorname{Encoder}\left( M\right) }\right) }\right) }\right)
$$



where Cat denotes categorical distribution.
其中 Cat 表示分类分布。


This gives us a stochastic policy from which we can sample actions. Empirically, this leads to more stable behaviors, which avoids getting stuck in suboptimal states.
这为我们提供了一个可采样动作的随机策略。经验上，这能带来更稳定的行为，避免陷入次优状态。


Discussion. The above SMT model is based on the encoder-decoder structure introduced in the Transformer model, which has seen successes on natural language processing (NLP) problems such as machine translation, text generation and summarization. The design principles of the model, supported by strong empirical results, transfer well from the NLP domain to the robot navigation setup, which is the primary motivation for adopting it.
讨论。上述 SMT 模型基于 Transformer 中引入的编码器-解码器结构，该结构已在机器翻译、文本生成和摘要等自然语言处理（NLP）问题上取得成功。模型的设计原则有强有力的经验支持，能很好地从 NLP 领域迁移到机器人导航设置，这是采用该结构的主要动机。


First, an agent moving in a large environment has to work with dynamically growing number of past observations. The encoder-decoder structure has shown strong performance exactly in the regime of lengthy textual inputs. Second, contrary to common RNNs or other structured external memories, we do not impose a predefined order or structure on the memory. Instead, we encode temporal and spatial information as part of the observation and let the policy learn to interpret the task-relevant information through the attention mechanism of the encoder-decoder structure.
首先，在大环境中移动的智能体必须处理动态增长的过去观测数量。编码器-解码器结构在处理冗长文本输入时表现优越。其次，与常见的 RNN 或其他结构化外部记忆不同，我们不对记忆施加预定义的顺序或结构。相反，我们将时序和空间信息作为观测的一部分编码，让策略通过编码器-解码器结构的注意力机制学习解释对任务相关的信息。


#### 3.2.3 Memory Factorization
#### 3.2.3 记忆分解


The computational complexity of the SMT is dominated by the number of query-key pairs in the attention mechanisms. Specifically,the time complexity is $O\left( {\left| M\right| }^{2}\right)$ for the encoder due to the self-attention,and $O\left( \left| M\right| \right)$ for the decoder. In long-horizon tasks, where the memory grows considerably, quadratic complexity can be prohibitive. Inspired by [25], we replace the self-attention block from Eq. (4) with a composition of two blocks of similar design but more tractable computation:
SMT 的计算复杂度由注意力机制中的查询-键对数量主导。具体而言，由于自注意力，编码器的时间复杂度为 $O\left( {\left| M\right| }^{2}\right)$，而解码器为 $O\left( \left| M\right| \right)$。在记忆大幅增长的长时间任务中，二次复杂度可能不可行。受文献[25]启发，我们将等式(4)中的自注意力块替换为两个设计相似但计算更易处理的块的组合：


$$
\operatorname{AttFact}\left( {M,\widetilde{M}}\right)  = \operatorname{AttBlock}\left( {M,\operatorname{AttBlock}\left( {\widetilde{M},M}\right) }\right) \tag{8}
$$



where we use a "compressed" memory $\widetilde{M}$ obtained via finding representative centers from $M$ . These centers need to be dynamically updated to maintain a good coverage of the stored observations. In practice, we can use any clustering algorithm. For the sake of efficiency, we apply iterative farthest point sampling (FPS) [34] to the embedded observations in $M$ ,in order to choose a subset of elements which are distant from each other in the feature space. The running time of FPS is $O\left( {\left| M\right| \left| \widetilde{M}\right| }\right)$ and the time complexity of AttFact is $O\left( {\left| M\right| \left| \widetilde{M}\right| }\right)$ . With a fixed number of centers,the overall time complexity becomes linear. The diagram of the encoder with memory factorization is shown in Fig. 3.
其中我们使用通过从 $M$ 中找到代表性中心得到的“压缩”记忆 $\widetilde{M}$。这些中心需要动态更新以保持对存储观测的良好覆盖。实际中我们可以使用任何聚类算法。为提高效率，我们对 $M$ 中的嵌入观测应用迭代最远点采样 (FPS) [34]，以选择在特征空间中彼此相距较远的元素子集。FPS 的运行时间为 $O\left( {\left| M\right| \left| \widetilde{M}\right| }\right)$，AttFact 的时间复杂度为 $O\left( {\left| M\right| \left| \widetilde{M}\right| }\right)$。在中心数量固定的情况下，整体时间复杂度变为线性。带有记忆分解的编码器示意图见图3。


### 3.3. Training
### 3.3. 训练


We train all model variants and baselines using the standard deep Q-learning algorithm [29]. We follow [29] in the use of an experience replay buffer, which has a capacity of 1000 episodes. The replay buffer is initially filled with episodes collected by a random policy and is updated every 500 training iterations. The update replaces the oldest episode in the buffer with a new episode collected by the updated policy. At every training iteration, we construct a batch of 64 episodes randomly sampled from the replay buffer. The model is trained with Adam Optimizer [24] with a learning rate of $5 \times  {10}^{-4}$ . All model parameters except for the embedding networks are trained end-to-end. During training, we continuously evaluate the updated policy on the validation set (as in Sec. 4.1). We keep training each model until we observe no improvement on the validation set.
我们使用标准的深度 Q 学习算法[29]训练所有模型变体和基线。我们遵循[29]使用经验回放缓冲区，其容量为1000个回合。回放缓冲区最初由随机策略收集的回合填充，并每500次训练迭代更新一次。更新用由更新后策略收集的新回合替换缓冲区中最旧的回合。在每次训练迭代中，我们从回放缓冲区随机采样构建64个回合的批次。模型使用学习率为 $5 \times  {10}^{-4}$ 的 Adam 优化器[24]进行训练。除嵌入网络外，所有模型参数端到端训练。在训练期间，我们持续在验证集上评估更新后的策略（如第4.1节）。我们在验证集上不再看到改进时停止训练每个模型。


The embedding networks are pre-trained using the SMT policy with the same training setup, with the only difference that the memory size is set to be 1. This leads to a SMT with no attention layers, as attention of size 1 is an identity mapping. In this way, the optimization is made easier so that the embedding networks can be trained end-to-end. After being trained to convergence, the parameters of the embedding networks are frozen for other models.
嵌入网络使用与 SMT 策略相同的训练设置进行预训练，唯一的区别是记忆大小设置为 1。这导致 SMT 不含注意力层，因为大小为 1 的注意力相当于恒等映射。这样可简化优化，使嵌入网络能端到端训练。训练收敛后，嵌入网络参数将在其他模型中冻结。


A major difference to RNN policies or other memory-based policies is that SMT does not need backpropagation through time (BPTT). As a result, the optimization is more stable and less computationally heavy. This enables training the model to exploit longer temporal dependencies.
与 RNN 策略或其他基于记忆的策略的主要区别在于 SMT 不需要时间反向传播（BPTT）。因此，优化更稳定且计算开销更小，从而能够训练模型利用更长的时间依赖。


### 3.4. Implementation Details
### 3.4. 实现细节


Image modalities are rendered as ${640} \times  {480}$ and subsampled by 10. Each image modality is embedded into 64- dimensional vectors using a modified ResNet-18 [17]. We reduce the numbers of filters of all convolutional layers by a factor of 4 and use stride of 1 for the first two convolutional layers. We remove the global pooling to better capture the spatial information and directly apply the fully-connected layer at the end. Both pose and action vectors are embedded using a single 16-dimensional fully-connected layer.
图像模态以 ${640} \times  {480}$ 呈现并下采样 10 倍。每种图像模态使用修改过的 ResNet-18 [17] 嵌入为 64 维向量。我们将所有卷积层的滤波器数量减少为原来的 1/4，前两层卷积使用步幅 1。为更好地捕捉空间信息，我们移除了全局池化，并在末端直接应用全连接层。位姿和动作向量均使用单层 16 维全连接层嵌入。


Attention blocks in SMT use multi-head attention mechanisms [42] with 8 heads. The keys and values are both 128- dimensional. All the fully connected layers in the attention blocks are 128-dimensional and use ReLU non-linearity.
SMT 中的注意力块使用 8 个头的多头注意力机制 [42]。键和值均为 128 维。注意力块中的所有全连接层均为 128 维并使用 ReLU 非线性。


A special caution is to be taken with the pose vector. First, at every time step all pose vectors in the memory are transformed to be in the coordinate system defined by the current agent pose. This is consistent with an ego-centric representation of the memory. Thus, the pose observations need to be re-embedded at every time step, while all the other observations are embedded once. Second, a pose vector $p = \left( {x,y,\theta }\right)$ at time $t$ is converted to a normalized version $p = \left( {x/\lambda ,y/\lambda ,\cos \theta ,\sin \theta ,{e}^{-t}}\right)$ ,embedding in addition its temporal information $t$ in a soft way in its last dimension. This allows the model to differentiate between recent and old observation, assuming that former could be more important than latter. The scaling factor $\lambda  = 5$ is used to reduce the magnitude of the coordinates.
位姿向量需要特别注意。首先，在每个时间步内，记忆中所有位姿向量均被转换到由当前智能体位姿定义的坐标系中。这与以自我为中心的记忆表示一致。因此，位姿观测需在每个时间步重新嵌入，而所有其他观测只嵌入一次。其次，时间为 $t$ 的位姿向量 $p = \left( {x,y,\theta }\right)$ 会被转换为归一化版本 $p = \left( {x/\lambda ,y/\lambda ,\cos \theta ,\sin \theta ,{e}^{-t}}\right)$，并在其最后一维以软方式嵌入其时间信息 $t$。这使模型能区分近期与较早的观测，假设前者可能比后者更重要。缩放因子 $\lambda  = 5$ 用于减小坐标幅度。


## 4. Experiments
## 4. 实验


We design our experiments to investigate the following topics: 1) How well does SMT perform on different long-horizon robot tasks 2) How important is its design properties compared to related methods? 3) Qualitatively, what agent behaviors does SMT learn?
我们设计实验以研究以下问题：1）SMT 在不同长时程机器人任务上的表现如何；2）与相关方法相比，其设计属性有多重要；3）从定性上看，SMT 学到哪些智能体行为？


### 4.1. Task Setup
### 4.1. 任务设置


To answer these questions, we consider three visual navigation tasks: roaming, coverage, and search. These tasks require the agent to summarize spatial and semantic information of the environment across long time horizons. All tasks share the same POMDP from Sec. 3.1 except that the reward functions are defined differently in each task.
为回答这些问题，我们考虑三种视觉导航任务：漫游、覆盖和搜索。这些任务要求智能体在长时间范围内总结环境的空间和语义信息。所有任务共享第 3.1 节的 POMDP，区别仅在各任务的奖励函数定义不同。


Roaming: The agent attempts to move forward as much as possible without colliding. In this basic navigation task, a memory should help the agent avoid cluttered areas and oscillating behaviors. The reward is defined as $R\left( {s,a}\right)  = 1$ iff $a =$ go_forward and no collision occurs.
漫游：智能体尝试尽可能多地向前移动且不发生碰撞。在这个基本导航任务中，记忆应帮助智能体避开拥挤区域和振荡行为。奖励定义为 $R\left( {s,a}\right)  = 1$ 当且仅当 $a =$ 执行 go_forward 且未发生碰撞。


Coverage: In many real-world application a robot needs to explore unknown environments and visit all areas of these environments. This task clearly requires a detailed memory as the robot is supposed to remember all places it has visited. To define the coverage task, we overlay a grid of cell size 0.5 over the floorplan of each environment. We would like the agent to visit as many unoccupied cells as possible, expressed by reward $R\left( {s,a}\right)  = 5$ iff robot entered unvisited cell after executing the action.
覆盖：在许多现实应用中，机器人需要探索未知环境并访问环境中的所有区域。该任务显然需要详细的记忆，因为机器人需记住其访问过的所有地点。为定义覆盖任务，我们在每个环境的平面图上覆盖网格，单元格大小为 0.5。我们希望智能体访问尽可能多的未占用单元格，用奖励 $R\left( {s,a}\right)  = 5$ 表示当且仅当机器人在执行动作后进入未访问单元格。


Search: To evaluate whether the policy can learn beyond knowledge about the geometry of the environment, we define a semantic version of the coverage tasks. In particular, for six target object classes ${}^{1}$ ,we want the robot to search for as many as possible of them in the house. Each house contains 1 to 6 target object classes, 4.9 classes in average. Specifically, an object is marked as found if more than 4% of pixels in an image has the object label (as in [45]) and the corresponding depth values are less than 2 meter. Thus, $R\left( {s,a}\right)  = {100}$ iff after taking action $a$ we find one of the six object classes which hasn't been found yet.
搜索：为评估策略是否能学到超越环境几何知识的能力，我们定义了覆盖任务的语义版本。具体地，对于六类目标物体 ${}^{1}$，我们希望机器人在房屋中尽可能多地搜索到它们。每个房屋包含 1 到 6 类目标物体，平均 4.9 类。具体地，若图像中超过 4% 的像素具有该物体标签（如 [45]），且对应深度值小于 2 米，则该物体被标记为已找到。因此，$R\left( {s,a}\right)  = {100}$ 当且仅当 在执行动作 $a$ 之后我们找到了尚未找到的六类目标之一。


We add a collision reward of -1 for each time the agent collides. An episode will be terminated if the agent runs into more than 50 collisions. To encourage exploration, we add coverage reward to the search task with a weight of 0.2 .
我们对每次与物体碰撞给予 -1 的碰撞奖励。若代理发生超过 50 次碰撞，则该回合终止。为鼓励探索，我们在搜索任务中加入覆盖奖励，权重为 0.2 。


The above tasks are listed in ascending order of complexity. The coverage and search tasks are studied in robotics, however, primarily in explored environments and are concerned about optimal path planning [12].
上述任务按复杂程度升序排列。覆盖和搜索任务在机器人学中有所研究，但主要是在已探索环境中，并且关注的是最优路径规划 {{FN}}。


Environment. We use SUNCG [37], a set of synthetic but visually realistic buildings. We use the same data split as chosen by [45] and remove the houses with artifacts, which gives us 195 training houses and 46 testing houses. We hold out ${20}\%$ of the training houses as a validation set for ablation experiments. We run 10 episodes in each house with a fixed random seed during testing and validation. The agent moves by a constant step size of 0.25 meters with go_forward. It turns by ${45}^{ \circ  }$ degree in place with turn_left or turn_right. Gaussian noise is added to simulate randomness in real-world dynamics.
环境。我们使用 SUNCG [37]，这是一组合成但视觉上逼真的建筑。我们使用与 [45] 相同的数据划分方式，并移除存在瑕疵的房屋，这样我们得到了 195 个训练房屋和 46 个测试房屋。我们留出 ${20}\%$ 个训练房屋作为消融实验的验证集。在测试和验证期间，我们在每个房屋中以固定随机种子运行 10 个回合。智能体通过 go_forward 以 0.25 米的恒定步长移动。它通过 turn_left 或 turn_right 原地旋转 ${45}^{ \circ  }$ 度。添加高斯噪声以模拟现实世界动态中的随机性。


Model Variants. To investigate the effect of different model aspects, we conduct experiments with three variants: SMT, SMT + Factorization, and SM + Pooling. The second model applies SMT with AttFact instead of AttBlock. Inspired by [11], the last model directly applies a max pooling over the elements in the scene memory instead of using the encoder-decoder structure of SMT.
模型变体。为研究不同模型要素的影响，我们进行了三种变体的实验：SMT、SMT + Factorization 和 SM + Pooling。第二种模型对 SMT 使用 AttFact 而非 AttBlock。受 [11] 启发，最后一种模型直接对场景记忆中的元素施加最大池化，而不是使用 SMT 的编码器-解码器结构。


Baselines. We use the following baselines for comparison. A Random policy uniformly samples one of the three actions. A Reactive policy is trained to directly compute Q values using a purely feedforward net. It is two fully-connected layers on top of the embedded observation at every step. A LSTM policy [27] is the most common memory-based policy. A model with arguably larger capacity, called FRMQN [31], maintains embedded observations in a fixed-sized memory, similarly as SMT. Instead of using the encode-decoder structure to exploit the memory, it uses an LSTM, whose input is current observation and output is used to attend over the memory.
基线。我们使用以下基线进行比较。Random 策略从三种动作中均匀采样一个。Reactive 策略通过纯前馈网络直接计算 Q 值。它在每步对嵌入观测上加两层全连接层。LSTM 策略 [27] 是最常见的基于记忆的策略。一个容量可能更大的模型，称为 FRMQN [31]，将嵌入的观测保存在固定大小的记忆中，类似于 SMT。它没有使用编码-解码结构来利用记忆，而是使用 LSTM，其输入为当前观测，输出用于对记忆进行注意。


<table><tr><td>Method</td><td>Reward</td><td>Distance</td><td>Collisions</td></tr><tr><td>Random</td><td>58.3</td><td>25.3</td><td>42.7</td></tr><tr><td>Reactive [27]</td><td>308.9</td><td>84.6</td><td>29.3</td></tr><tr><td>LSTM [27]</td><td>379.7</td><td>97.9</td><td>11.4</td></tr><tr><td>FRMQN [31]</td><td>384.2</td><td>99.5</td><td>13.8</td></tr><tr><td>SM + Pooling</td><td>366.8</td><td>96.7</td><td>20.1</td></tr><tr><td>SMT + Factorization</td><td>376.4</td><td>98.6</td><td>17.9</td></tr><tr><td>SMT</td><td>394.7</td><td>102.1</td><td>13.6</td></tr></table>
<table><tbody><tr><td>方法</td><td>回报</td><td>距离</td><td>碰撞</td></tr><tr><td>随机</td><td>58.3</td><td>25.3</td><td>42.7</td></tr><tr><td>反应式 [27]</td><td>308.9</td><td>84.6</td><td>29.3</td></tr><tr><td>LSTM [27]</td><td>379.7</td><td>97.9</td><td>11.4</td></tr><tr><td>FRMQN [31]</td><td>384.2</td><td>99.5</td><td>13.8</td></tr><tr><td>SM + 池化</td><td>366.8</td><td>96.7</td><td>20.1</td></tr><tr><td>SMT + 因式分解</td><td>376.4</td><td>98.6</td><td>17.9</td></tr><tr><td>SMT</td><td>394.7</td><td>102.1</td><td>13.6</td></tr></tbody></table>


Table 1. Performance on Roaming. The average of cumulative reward, roaming distance and number of collisions are listed.
表 1。漫游性能。列出累积奖励、漫游距离和碰撞次数的平均值。


<table><tr><td>Method</td><td>Reward</td><td>Covered Cells</td></tr><tr><td>Random</td><td>94.2</td><td>27.4</td></tr><tr><td>Reactive [27]</td><td>416.2</td><td>86.9</td></tr><tr><td>LSTM [27]</td><td>418.1</td><td>87.8</td></tr><tr><td>FRMQN [31]</td><td>397.7</td><td>83.2</td></tr><tr><td>SM + Pooling</td><td>443.9</td><td>91.5</td></tr><tr><td>SMT + Factorization</td><td>450.1</td><td>99.3</td></tr><tr><td>SMT</td><td>474.6</td><td>102.5</td></tr></table>
<table><tbody><tr><td>方法</td><td>奖励</td><td>覆盖单元格</td></tr><tr><td>随机</td><td>94.2</td><td>27.4</td></tr><tr><td>反应式 [27]</td><td>416.2</td><td>86.9</td></tr><tr><td>LSTM [27]</td><td>418.1</td><td>87.8</td></tr><tr><td>FRMQN [31]</td><td>397.7</td><td>83.2</td></tr><tr><td>SM + 池化</td><td>443.9</td><td>91.5</td></tr><tr><td>SMT + 因式分解</td><td>450.1</td><td>99.3</td></tr><tr><td>SMT</td><td>474.6</td><td>102.5</td></tr></tbody></table>


Table 2. Performance on Coverage. The average of cumulative reward and number of covered cells are listed.
表 2。覆盖性能。列出累积奖励和覆盖单元格数的平均值。


<table><tr><td>Method</td><td>Reward</td><td>Classes</td><td>Ratio</td></tr><tr><td>Random</td><td>140.5</td><td>1.79</td><td>36.3%</td></tr><tr><td>Reactive [27]</td><td>358.2</td><td>3.14</td><td>61.9%</td></tr><tr><td>LSTM [27]</td><td>339.4</td><td>3.07</td><td>62.6%</td></tr><tr><td>FRMQN [31]</td><td>411.2</td><td>3.53</td><td>70.2%</td></tr><tr><td>SM + Pooling</td><td>332.5</td><td>2.98</td><td>60.6%</td></tr><tr><td>SMT + Factorization</td><td>432.6</td><td>3.69</td><td>75.0%</td></tr><tr><td>SMT</td><td>428.4</td><td>3.65</td><td>74.2%</td></tr></table>
<table><tbody><tr><td>方法</td><td>奖励</td><td>类别</td><td>比例</td></tr><tr><td>随机</td><td>140.5</td><td>1.79</td><td>36.3%</td></tr><tr><td>反应式 [27]</td><td>358.2</td><td>3.14</td><td>61.9%</td></tr><tr><td>LSTM [27]</td><td>339.4</td><td>3.07</td><td>62.6%</td></tr><tr><td>FRMQN [31]</td><td>411.2</td><td>3.53</td><td>70.2%</td></tr><tr><td>SM + 池化</td><td>332.5</td><td>2.98</td><td>60.6%</td></tr><tr><td>SMT + 因式分解</td><td>432.6</td><td>3.69</td><td>75.0%</td></tr><tr><td>SMT</td><td>428.4</td><td>3.65</td><td>74.2%</td></tr></tbody></table>


Table 3. Performance on Search. The cumulative of total reward, number of found classes and ratio of found classes are listed.
表 3. 搜索性能。列出了总奖励的累积值、找到的类别数量和找到类别的比例。


For all methods, we use the same pretrained embedding networks and two fully-connected layers to compute Q values. We also use the same batch size of 64 during training. To train LSTM and FRMQN, we use truncated back propagation through time of 128 steps.
对于所有方法，我们使用相同的预训练嵌入网络和两个全连接层来计算 Q 值。在训练期间，我们也使用相同的批量大小 64。为了训练 LSTM 和 FRMQN，我们使用 128 步的截断时间反向传播。


### 4.2. Comparative Evaluation
### 4.2. 对比评估


The methods are compared across the three different tasks: roaming in Table 1, coverage in Table 2, and search in Table 3. For each task and method we show the attained reward and task specific metrics.
对这些方法在三个不同任务上进行了比较：表 1 中的漫游任务、表 2 中的覆盖任务和表 3 中的搜索任务。对于每个任务和方法，我们展示了获得的奖励和特定任务的指标。


Effect of memory designs. Across all tasks, SMT outperforms all other memory-based models. The relative performance gain compared to other approaches is most significant for coverage (14% improvements) and considerable for search (5% improvements). This is consistent with the notion that for coverage and search memorizing all past observations is more vital. On roaming, larger memory capacity (in SMT case) helps, however, all memory-based approaches perform in the same ballpark. This is reasonable in the sense that maintaining a straight collision free trajectory is a relatively short-sight task.
内存设计的影响。在所有任务中，SMT 的表现优于所有其他基于内存的模型。与其他方法相比，其相对性能提升在覆盖任务中最为显著（提高了 14%），在搜索任务中也相当可观（提高了 5%）。这与以下观点一致，即对于覆盖和搜索任务，记住所有过去的观测结果更为关键。在漫游任务中，更大的内存容量（如 SMT 情况）有帮助，然而，所有基于内存的方法表现相近。从保持无碰撞的直线轨迹是一个相对短视的任务这一意义上来说，这是合理的。


---



${}^{1}$ We use television,refrigerator,bookshelf,table,sofa,and bed.
${}^{1}$ 我们使用电视、冰箱、书架、桌子、沙发和床。


---



<img src="https://cdn.noedgeai.com/bo_d4nfr977aajc73frs8r0_6.jpg?x=171&y=217&w=650&h=332&r=0"/>



Figure 4. Found classes by time steps. For the search task, we show number of found target object classes across time steps.
图 4. 按时间步长找到的类别。对于搜索任务，我们展示了在各个时间步长中找到的目标对象类别的数量。


In addition to memory capacity, memory access via attention brings improvements. For all tasks SMT outperforms SM + Pooling. The gap is particularly large for object search (Table 3), where the task has an additional semantic complexity of finding objects. Similarly, having multiheaded attention and residual layers brings an improvement over a basic attention, as employed by FRMQN, which is demonstrated on both coverage and search.
除了内存容量，通过注意力机制进行内存访问也能带来性能提升。对于所有任务，SMT 的表现优于 SM + 池化方法。在对象搜索任务中（表 3），差距尤为明显，因为该任务在寻找对象方面还有额外的语义复杂性。同样，使用多头注意力和残差层比 FRMQN 所采用的基本注意力机制有所改进，这在覆盖和搜索任务中都得到了体现。


The proposed memory factorization brings computational benefits, at no or limited performance loss. Even if it causes drop sometimes, the reward is better than SM + Pooling and other baseline methods.
所提出的内存分解方法在不损失或仅有限损失性能的情况下带来了计算上的好处。即使有时会导致性能下降，其奖励也优于 SM + 池化和其他基线方法。


Implications of memory for navigation. It is also important to understand how memory aids us at solving navigation tasks. For this purpose, in addition to reward, we report number of covered cells (Table 2) and number of found objects (Table 3). For both tasks, a reactive policy presents a strong baseline, which we suspect learns general exploration principles. Adding memory via SMT helps boost the coverage by 18% over reactive, and 17% over LSTM policies and 23% over simpler memory mechanism (FRMQN). We also observe considerable boosts of number of found objects by $5\%$ in the search task.
内存对导航的影响。理解内存如何帮助我们解决导航任务也很重要。为此，除了奖励之外，我们还报告了覆盖的单元格数量（表 2）和找到的对象数量（表 3）。对于这两个任务，反应式策略提供了一个强大的基线，我们怀疑它学习到了通用的探索原则。通过 SMT 添加内存后，与反应式策略相比，覆盖范围提高了 18%，与 LSTM 策略相比提高了 17%，与更简单的内存机制（FRMQN）相比提高了 23%。我们还观察到在搜索任务中，找到 $5\%$ 对象的数量有显著增加。


The reported metrics above are for a fixed time horizon of 500 steps. For varying time horizons, we show the performance on search in Fig. 4. We see that memory-based policies with attention-based reads consistently find more object classes as they explore the environment, with SMT variants being the best. This is true across the full execution with performance gap increasing steadily up to 300 steps.
上述报告的指标是在 500 步的固定时间范围内的。对于不同的时间范围，我们在图 4 中展示了搜索任务的性能。我们发现，基于注意力读取的基于内存的策略在探索环境时始终能找到更多的对象类别，其中 SMT 变体表现最佳。在整个执行过程中都是如此，性能差距在达到 300 步之前稳步增大。


### 4.3. Ablation Analysis
### 4.3. 消融分析


Here, we analyze two aspects of SMT: (i) size of the scene memory, and (ii) importance of the different observation modalities and componenets.
在这里，我们分析 SMT 的两个方面：（i）场景内存的大小，以及（ii）不同观测模态和组件的重要性。


<img src="https://cdn.noedgeai.com/bo_d4nfr977aajc73frs8r0_6.jpg?x=913&y=204&w=690&h=674&r=0"/>



Figure 5. Ablation Experiments. (a) We sweep the memory capacity from 50 steps to 500 steps and evaluate the reward of trajectories of 500 steps. (b) We leave out one component at a time in our full model and evaluate the averaged reward for each task.
图 5. 消融实验。（a）我们将内存容量从 50 步扫描到 500 步，并评估 500 步轨迹的奖励。（b）我们在完整模型中每次去掉一个组件，并评估每个任务的平均奖励。


Memory capacity. While in the previous section we discussed memory capacity across models, here we look at the importance of memory size for SMT. Intuitively a memory-based policy is supposed to benefits more from larger memory over long time horizons. But in practice this depends on the task and the network capacity, as shown in Fig. 5 (a). All three tasks benefit from using larger scene memory. The performance of roaming grows for memory up to 300 elements. For coverage and search, the performance keeps improving constantly with larger memory capacities. This shows that SMT does leverage the provided memory.
内存容量。虽然在上一节中我们讨论了不同模型的内存容量，但在这里我们关注内存大小对 SMT 的重要性。直观地说，基于内存的策略在较长的时间范围内应该从更大的内存中受益更多。但实际上，这取决于任务和网络容量，如图 5（a）所示。所有三个任务都从使用更大的场景内存中受益。漫游任务的性能在内存达到 300 个元素之前不断提高。对于覆盖和搜索任务，性能随着内存容量的增大而持续提升。这表明 SMT 确实利用了所提供的内存。


Modalities and components. For the presented tasks, we have image observations, pose, previous actions. To understand their importance, we re-train SMT by leaving out one modality at a time. We show the resulting reward in Fig. 5 (b). Among the observation modalities, last action, pose and the depth image play crucial roles across tasks. This is probably because SMT uses relative pose and last action to reason about spatial relationships. Further, depth image is the strongest clue related to collision avoidance, which is crucial for all tasks. Removing segmentation and RGB observations leads to little effect on coverage and drop of 10 for roaming since these tasks are defined primarily by environment geometry. For search, however, where SMT needs to work with semantics, the drop is 15 and 20.
模态和组件。对于所提出的任务，我们有图像观测、姿态和先前动作。为了理解它们的重要性，我们每次去掉一种模态来重新训练 SMT。我们在图 5（b）中展示了得到的奖励。在观测模态中，上一个动作、姿态和深度图像在各个任务中都起着至关重要的作用。这可能是因为 SMT 使用相对姿态和上一个动作来推理空间关系。此外，深度图像是与避障相关的最有力线索，这对所有任务都至关重要。去掉分割和 RGB 观测对覆盖任务影响不大，对漫游任务的奖励降低了 10，因为这些任务主要由环境几何形状决定。然而，对于搜索任务，SMT 需要处理语义信息，奖励分别降低了 15 和 20。


We also show that the encoder structure brings performance boost to the tasks. Especially in the search task, which is most challenging in terms of reasoning and planning, the encoder boosts the task reward by 23.7.
我们还表明，编码器结构为这些任务带来了性能提升。特别是在搜索任务中，这在推理和规划方面是最具挑战性的，编码器使任务奖励提高了 23.7。


<img src="https://cdn.noedgeai.com/bo_d4nfr977aajc73frs8r0_7.jpg?x=141&y=222&w=1465&h=848&r=0"/>



Figure 6. Visualization of the agent behaviors. We visualize the trajectories from the top-down view as green curves. Starting point and ending point of each trajectory are plot in white and black. Navigable area are masked in dark purple with red lines indicating the collision boundaries. For the coverage task, we mark the covered cells in pink. For the search task, we mark target objects with yellow masks.
图 6. 智能体行为可视化。我们将轨迹从俯视图的角度可视化为绿色曲线。每条轨迹的起点和终点分别用白色和黑色表示。可导航区域用深紫色遮罩，红色线条表示碰撞边界。对于覆盖任务，我们将已覆盖的单元格标记为粉色。对于搜索任务，我们用黄色遮罩标记目标对象。


### 4.4. Qualitative Results
### 4.4. 定性结果


To better understand the learned behaviors of the agent, we visualize the navigation trajectories in Fig. 6. We choose reactive and LSTM policies as representatives of memoryless and memory-based baselines to compare with SMT.
为了更好地理解智能体学习到的行为，我们在图 6 中可视化了导航轨迹。我们选择反应式和 LSTM 策略作为无记忆和基于记忆的基线代表，与 SMT 进行比较。


In the roaming task, our model demonstrates better strategies to keep moving and avoid collisions. In many of the cases, the agent first finds a long clear path in the house, which lets it go straight forward without frequently making turns. Then the agent navigates back and forth along the same route until the end of the episode. As a result, SMT usually leads to compact trajectories as shown in Fig. 6, top row. In contrast, reactive policy and LSTM policy often wander around the scene with a less consistent pattern.
在漫游任务中，我们的模型展示了更好的持续移动和避免碰撞的策略。在许多情况下，智能体首先在房屋中找到一条长的畅通路径，使其能够直线前进而无需频繁转弯。然后，智能体沿着同一路线来回导航，直到回合结束。因此，如图 6 顶行所示，SMT 通常会产生紧凑的轨迹。相比之下，反应式策略和 LSTM 策略通常会在场景中以不太一致的模式徘徊。


In the coverage task, our model explores the unseen space more efficiently by memorizing regions that have been covered. As shown in Fig. 6, middle row, after most of the cells inside a room being explored, the agent switches to the next unvisited room. Note that the cells are invisible to the agent, it needs to make this decision solely based on its memory and observation of the environment. It also remembers better which rooms have been visited so that it does not enter a room twice.
在覆盖任务中，我们的模型通过记忆已覆盖的区域更有效地探索未探索的空间。如图 6 中间行所示，在一个房间内的大部分单元格被探索后，智能体切换到下一个未访问的房间。请注意，单元格对智能体是不可见的，它需要仅根据其记忆和对环境的观察来做出此决策。它还能更好地记住哪些房间已被访问，从而不会两次进入同一个房间。


In the search task, our model shows efficient exploration as well as effective strategies to find the target classes. The search task also requires the agent to explore rooms with the difference that the exploration is driven by target object classes. Therefore, after entering a new room, the agent quickly scans around the space instead of covering all the navigable regions. In Fig. 6, if the agent finds the unseen target it goes straight towards it. Once it is done, it will leave the room directly. Comparing SMT with baselines, our trajectories are straight and direct between two targets, while baseline policies have more wandering patterns.
在搜索任务中，我们的模型展示了高效的探索以及找到目标类别的有效策略。搜索任务还要求智能体探索房间，不同之处在于这种探索是由目标对象类别驱动的。因此，进入一个新房间后，智能体快速扫描周围空间，而不是覆盖所有可导航区域。在图 6 中，如果智能体发现了未见过的目标，它会直接朝目标前进。一旦完成，它将直接离开房间。将 SMT 与基线进行比较，我们的轨迹在两个目标之间是笔直且直接的，而基线策略有更多的徘徊模式。


## 5. Conclusion
## 5. 结论


This paper introduces Scene Memory Transformer, a memory-based policy to aggregate observation history in robotic tasks of long time horizons. We use attention mechanism to exploit spatio-temporal dependencies across past observations. The policy is trained on several visual navigation tasks using deep Q-learning. Evaluation shows that the resulting policy achieves higher performance to other established methods.
本文介绍了场景记忆变压器（Scene Memory Transformer），这是一种基于记忆的策略，用于在长时间范围的机器人任务中聚合观察历史。我们使用注意力机制来挖掘过去观察之间的时空依赖关系。该策略使用深度 Q 学习在多个视觉导航任务上进行训练。评估表明，得到的策略比其他已有的方法具有更高的性能。


Acknowledgement: We thank Anelia Angelova, Ashish Vaswani and Jakob Uszkoreit for constructive discussions. We thank Marek Fišer for the software development of the simulation environment, Oscar Ramirez and Ayzaan Wahid for the support of the learning infrastructure.
致谢：我们感谢阿内莉亚·安杰洛娃（Anelia Angelova）、阿什什·瓦斯瓦尼（Ashish Vaswani）和雅各布·乌兹科雷特（Jakob Uszkoreit）进行的建设性讨论。我们感谢马雷克·菲舍尔（Marek Fišer）为模拟环境进行的软件开发，感谢奥斯卡·拉米雷斯（Oscar Ramirez）和阿扎安·瓦希德（Ayzaan Wahid）对学习基础设施的支持。


## References
## 参考文献


[1] P. Anderson, A. X. Chang, D. S. Chaplot, A. Dosovitskiy, S. Gupta, V. Koltun, J. Kosecka, J. Malik, R. Mottaghi, M. Savva, and A. R. Zamir. On evaluation of embodied navigation agents. CoRR, abs/1807.06757, 2018. 1
[1] P. 安德森（P. Anderson）、A. X. 张（A. X. Chang）、D. S. 查普洛特（D. S. Chaplot）、A. 多索维茨基（A. Dosovitskiy）、S. 古普塔（S. Gupta）、V. 科尔图恩（V. Koltun）、J. 科塞卡（J. Kosecka）、J. 马利克（J. Malik）、R. 莫塔吉（R. Mottaghi）、M. 萨瓦（M. Savva）和 A. R. 扎米尔（A. R. Zamir）。关于具身导航智能体的评估。CoRR，abs/1807.06757，2018 年。1


[2] P. Anderson, Q. Wu, D. Teney, J. Bruce, M. Johnson, N. Sünderhauf, I. Reid, S. Gould, and A. van den Hengel. Vision-and-Language Navigation: Interpreting visually-grounded navigation instructions in real environments. In IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2018. 1, 2
[2] P. 安德森（P. Anderson）、Q. 吴（Q. Wu）、D. 特尼（D. Teney）、J. 布鲁斯（J. Bruce）、M. 约翰逊（M. Johnson）、N. 松德豪夫（N. Sünderhauf）、I. 里德（I. Reid）、S. 古尔德（S. Gould）和 A. 范登亨格尔（A. van den Hengel）。视觉与语言导航：在真实环境中解释基于视觉的导航指令。见 IEEE 计算机视觉与模式识别会议（CVPR），2018 年。1, 2


[3] K. Arulkumaran, M. P. Deisenroth, M. Brundage, and A. A. Bharath. A brief survey of deep reinforcement learning. CoRR, abs/1708.05866, 2017. 1
[3] K. 阿鲁库马兰（K. Arulkumaran）、M. P. 戴森罗思（M. P. Deisenroth）、M. 布伦戴奇（M. Brundage）和 A. A. 巴拉思（A. A. Bharath）。深度强化学习简要综述。CoRR，abs/1708.05866，2017 年。1


[4] J. L. Ba, J. R. Kiros, and G. E. Hinton. Layer normalization. arXiv preprint arXiv:1607.06450, 2016. 4
[4] J. L. 巴（J. L. Ba）、J. R. 基罗斯（J. R. Kiros）和 G. E. 辛顿（G. E. Hinton）。层归一化。arXiv 预印本 arXiv:1607.06450，2016 年。4


[5] B. Bakker. Reinforcement learning with long short-term memory. In Advances in Neural Information Processing Systems, pages 1475-1482, 2002. 1
[5] B. 巴克（B. Bakker）。基于长短期记忆的强化学习。见《神经信息处理系统进展》，第 1475 - 1482 页，2002 年。1


[6] F. Bonin-Font, A. Ortiz, and G. Oliver. Visual navigation for mobile robots: A survey. Journal of Intelligent and Robotic Systems, 53:263-296, 2008. 2
[6] F. Bonin-Font, A. Ortiz, 和 G. Oliver. 移动机器人视觉导航综述。Journal of Intelligent and Robotic Systems, 53:263-296, 2008. 2


[7] D. S. Chaplot, E. Parisotto, and R. Salakhutdinov. Active neural localization. arXiv preprint arXiv:1801.08214, 2018. 2
[7] D. S. Chaplot, E. Parisotto, 和 R. Salakhutdinov. 主动神经定位。arXiv 预印本 arXiv:1801.08214, 2018. 2


[8] A. J. Davison. Real-time simultaneous localisation and mapping with a single camera. In IEEE International Conference on Computer Vision (ICCV), 2003. 2
[8] A. J. Davison. 使用单摄像头的实时同步定位与建图。收录于 IEEE 国际计算机视觉会议 (ICCV), 2003. 2


[9] F. Dayoub, T. Morris, B. Upcroft, and P. Corke. Vision-only autonomous navigation using topometric maps. In Intelligent robots and systems (IROS), 2013 IEEE/RSJ international conference on, pages 1923-1929. IEEE, 2013. 2
[9] F. Dayoub, T. Morris, B. Upcroft, 和 P. Corke. 仅视觉的自主导航：基于拓扑度量地图。收录于 Intelligent robots and systems (IROS), 2013 IEEE/RSJ international conference on, 页 1923-1929. IEEE, 2013. 2


[10] G. N. DeSouza and A. C. Kak. Vision for mobile robot navigation: A survey. IEEE Transactions on Pattern Analysis and Machine Intelligence, 24(2):237-267, 2002. 2
[10] G. N. DeSouza 和 A. C. Kak. 移动机器人导航的视觉研究综述。IEEE Transactions on Pattern Analysis and Machine Intelligence, 24(2):237-267, 2002. 2


[11] S. M. A. Eslami, D. J. Rezende, F. Besse, F. Viola, A. S. Morcos, M. Garnelo, A. Ruderman, A. A. Rusu, I. Dani-helka, K. Gregor, D. P. Reichert, L. Buesing, T. Weber, O. Vinyals, D. Rosenbaum, N. C. Rabinowitz, H. King, C. Hillier, M. M. Botvinick, D. Wierstra, K. Kavukcuoglu, and D. Hassabis. Neural scene representation and rendering. Science, 360:1204-1210, 2018. 6
[11] S. M. A. Eslami, D. J. Rezende, F. Besse, F. Viola, A. S. Morcos, M. Garnelo, A. Ruderman, A. A. Rusu, I. Dani-helka, K. Gregor, D. P. Reichert, L. Buesing, T. Weber, O. Vinyals, D. Rosenbaum, N. C. Rabinowitz, H. King, C. Hillier, M. M. Botvinick, D. Wierstra, K. Kavukcuoglu, 和 D. Hassabis. 神经场景表征与渲染。Science, 360:1204-1210, 2018. 6


[12] E. Galceran and M. Carreras. A survey on coverage path planning for robotics. Robotics and Autonomous systems, 61(12):1258-1276, 2013. 6
[12] E. Galceran 和 M. Carreras. 机器人覆盖路径规划综述。Robotics and Autonomous systems, 61(12):1258-1276, 2013. 6


[13] D. Gordon, A. Kembhavi, M. Rastegari, J. Redmon, D. Fox, and A. Farhadi. Iqa: Visual question answering in interactive environments. arXiv preprint arXiv:1712.03316, 1, 2017. 2
[13] D. Gordon, A. Kembhavi, M. Rastegari, J. Redmon, D. Fox, 和 A. Farhadi. IQA：交互环境中的视觉问答。arXiv 预印本 arXiv:1712.03316, 1, 2017. 2


[14] A. Graves, G. Wayne, M. Reynolds, T. Harley, I. Danihelka, A. Grabska-Barwińska, S. G. Colmenarejo, E. Grefenstette, T. Ramalho, J. Agapiou, et al. Hybrid computing using a neural network with dynamic external memory. Nature, 538(7626):471, 2016. 2
[14] A. Graves, G. Wayne, M. Reynolds, T. Harley, I. Danihelka, A. Grabska-Barwińska, S. G. Colmenarejo, E. Grefenstette, T. Ramalho, J. Agapiou, 等. 使用带动态外部记忆的神经网络的混合计算。Nature, 538(7626):471, 2016. 2


[15] S. Gupta, J. Davidson, S. Levine, R. Sukthankar, and J. Malik. Cognitive mapping and planning for visual navigation. IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 7272-7281, 2017. 1, 2
[15] S. Gupta, J. Davidson, S. Levine, R. Sukthankar, 和 J. Malik. 视觉导航的认知映射与规划。IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 页 7272-7281, 2017. 1, 2


[16] S. Gupta, D. F. Fouhey, S. Levine, and J. Malik. Unifying map and landmark based representations for visual navigation. CoRR, abs/1712.08125, 2017. 1, 2
[16] S. Gupta, D. F. Fouhey, S. Levine, 和 J. Malik. 统一基于地图与基于地标的视觉导航表示。CoRR, abs/1712.08125, 2017. 1, 2


[17] K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning for image recognition. In IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 770-778, 2016. 5
[17] K. He, X. Zhang, S. Ren, 和 J. Sun. 用于图像识别的深度残差学习。收录于 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 页 770-778, 2016. 5


[18] J. F. Henriques and A. Vedaldi. Mapnet : An allocentric spatial memory for mapping environments. In IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2018. 1,2
[18] J. F. Henriques 和 A. Vedaldi. MapNet：用于环境映射的中心外空间记忆。收录于 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2018. 1,2


[19] S. Hochreiter and J. Schmidhuber. Long short-term memory. Neural computation, 9(8):1735-1780, 1997. 2
[19] S. Hochreiter 和 J. Schmidhuber. 长短期记忆。Neural computation, 9(8):1735-1780, 1997. 2


[20] R. Jonschkowski, D. Rastogi, and O. Brock. Differentiable particle filters: End-to-end learning with algorithmic priors. arXiv preprint arXiv:1805.11122, 2018. 2
[20] R. Jonschkowski, D. Rastogi, 和 O. Brock. 可微粒子滤波：具有算法先验的端到端学习。arXiv 预印本 arXiv:1805.11122, 2018. 2


[21] L. P. Kaelbling, M. L. Littman, and A. R. Cassandra. Planning and acting in partially observable stochastic domains. Artificial Intelligence 101, 101:99-134, 1998. 3
[21] L. P. Kaelbling, M. L. Littman, 和 A. R. Cassandra. 在部分可观测随机域中的规划与行动。Artificial Intelligence 101, 101:99-134, 1998. 3


[22] M. Kempka, M. Wydmuch, G. Runc, J. Toczek, and W. Jaśkowski. Vizdoom: A doom-based ai research platform for visual reinforcement learning. In Computational Intelligence and Games (CIG), 2016 IEEE Conference on, pages 1-8. IEEE, 2016. 1, 2
[22] M. Kempka, M. Wydmuch, G. Runc, J. Toczek, and W. Jaśkowski. Vizdoom：一个基于 Doom 的用于视觉强化学习的 AI 研究平台。见 Computational Intelligence and Games (CIG), 2016 IEEE Conference on, 页码 1-8。IEEE, 2016。1, 2


[23] A. Khan, C. Zhang, N. Atanasov, K. Karydis, V. Kumar, and D. D. Lee. Memory augmented control networks. arXiv preprint arXiv:1709.05706, 2017. 2
[23] A. Khan, C. Zhang, N. Atanasov, K. Karydis, V. Kumar, and D. D. Lee. Memory augmented control networks。arXiv 预印本 arXiv:1709.05706, 2017。2


[24] D. P. Kingma and J. Ba. Adam: A method for stochastic optimization. International Conference for Learning Representations, 2015. 5
[24] D. P. Kingma and J. Ba. Adam：一种用于随机优化的方法。International Conference for Learning Representations, 2015。5


[25] J. Lee, Y. Lee, J. Kim, A. R. Kosiorek, S. Choi, and Y. W. Teh. Set transformer. CoRR, abs/1810.00825, 2018. 5
[25] J. Lee, Y. Lee, J. Kim, A. R. Kosiorek, S. Choi, and Y. W. Teh. Set transformer。CoRR, abs/1810.00825, 2018。5


[26] P. J. Liu, M. Saleh, E. Pot, B. Goodrich, R. Sepassi, L. Kaiser, and N. Shazeer. Generating wikipedia by summarizing long sequences. arXiv preprint arXiv:1801.10198, 2018. 2
[26] P. J. Liu, M. Saleh, E. Pot, B. Goodrich, R. Sepassi, L. Kaiser, and N. Shazeer. 通过总结长序列生成维基百科。arXiv 预印本 arXiv:1801.10198, 2018。2


[27] P. W. Mirowski, R. Pascanu, F. Viola, H. Soyer, A. J. Ballard, A. Banino, M. Denil, R. Goroshin, L. Sifre, K. Kavukcuoglu, D. Kumaran, and R. Hadsell. Learning to navigate in complex environments. CoRR, abs/1611.03673, 2016. 1, 2, 6
[27] P. W. Mirowski, R. Pascanu, F. Viola, H. Soyer, A. J. Ballard, A. Banino, M. Denil, R. Goroshin, L. Sifre, K. Kavukcuoglu, D. Kumaran, and R. Hadsell. 学习在复杂环境中导航。CoRR, abs/1611.03673, 2016。1, 2, 6


[28] V. Mnih, A. P. Badia, M. Mirza, A. Graves, T. P. Lillicrap, T. Harley, D. Silver, and K. Kavukcuoglu. Asynchronous methods for deep reinforcement learning. In International Conference on Machine Learning (ICML), 2016. 2
[28] V. Mnih, A. P. Badia, M. Mirza, A. Graves, T. P. Lillicrap, T. Harley, D. Silver, and K. Kavukcuoglu. 用于深度强化学习的异步方法。见 International Conference on Machine Learning (ICML), 2016。2


[29] V. Mnih, K. Kavukcuoglu, D. Silver, A. A. Rusu, J. Ve-ness, M. G. Bellemare, A. Graves, M. A. Riedmiller, A. Fid-jeland, G. Ostrovski, S. Petersen, C. Beattie, A. Sadik, I. Antonoglou, H. King, D. Kumaran, D. Wierstra, S. Legg, and D. Hassabis. Human-level control through deep reinforcement learning. Nature, 518:529-533, 2015. 2, 5
[29] V. Mnih, K. Kavukcuoglu, D. Silver, A. A. Rusu, J. Ve-ness, M. G. Bellemare, A. Graves, M. A. Riedmiller, A. Fid-jeland, G. Ostrovski, S. Petersen, C. Beattie, A. Sadik, I. Antonoglou, H. King, D. Kumaran, D. Wierstra, S. Legg, and D. Hassabis. 通过深度强化学习实现人类级控制。Nature, 518:529-533, 2015。2, 5


[30] A. Mousavian, A. Toshev, M. Fiser, J. Kosecka, and J. Davidson. Visual representations for semantic target driven navigation. arXiv preprint arXiv:1805.06066, 2018. 2
[30] A. Mousavian, A. Toshev, M. Fiser, J. Kosecka, and J. Davidson. 用于语义目标驱动导航的视觉表示。arXiv 预印本 arXiv:1805.06066, 2018。2


[31] J. Oh, V. Chockalingam, S. P. Singh, and H. Lee. Control of memory, active perception, and action in minecraft. In International Conference on Machine Learning (ICML), 2016. 1, 2,6
[31] J. Oh, V. Chockalingam, S. P. Singh, and H. Lee. 在 Minecraft 中控制记忆、主动感知与动作。见 International Conference on Machine Learning (ICML), 2016。1, 2,6


[32] E. Parisotto and R. Salakhutdinov. Neural map: Structured memory for deep reinforcement learning. CoRR, abs/1702.08360, 2017. 1, 2
[32] E. Parisotto and R. Salakhutdinov. Neural map：用于深度强化学习的结构化记忆。CoRR, abs/1702.08360, 2017。1, 2


[33] R. Pascanu, T. Mikolov, and Y. Bengio. On the difficulty of training recurrent neural networks. In International Conference on Machine Learning (ICML), 2013. 1, 2
[33] R. Pascanu, T. Mikolov, and Y. Bengio. 关于训练循环神经网络的困难。见 International Conference on Machine Learning (ICML), 2013。1, 2


[34] C. R. Qi, L. Yi, H. Su, and L. J. Guibas. Pointnet++: Deep hierarchical feature learning on point sets in a metric space. In Advances in Neural Information Processing Systems, pages 5099-5108, 2017. 5
[34] C. R. Qi, L. Yi, H. Su, and L. J. Guibas. Pointnet++：在度量空间点集上的深层分层特征学习。见 Advances in Neural Information Processing Systems, 页码 5099-5108, 2017。5


[35] N. Savinov, A. Dosovitskiy, and V. Koltun. Semi-parametric topological memory for navigation. International Conference on Learning Representations, 2018. 2
[35] N. Savinov, A. Dosovitskiy, and V. Koltun. 用于导航的半参数拓扑记忆。International Conference on Learning Representations, 2018。2


[36] R. Sim and J. J. Little. Autonomous vision-based exploration and mapping using hybrid maps and rao-blackwellised particle filters. 2006 IEEE/RSJ International Conference on Intelligent Robots and Systems, pages 2082-2089, 2006. 2
[36] R. Sim and J. J. Little. 使用混合地图和拉奥-布莱克韦利粒子滤波的自主基于视觉的探索与建图。2006 IEEE/RSJ International Conference on Intelligent Robots and Systems, 页码 2082-2089, 2006。2


[37] S. Song, F. Yu, A. Zeng, A. X. Chang, M. Savva, and T. Funkhouser. Semantic scene completion from a single depth image. IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2017. 6
[37] S. Song, F. Yu, A. Zeng, A. X. Chang, M. Savva, and T. Funkhouser. 从单幅深度图进行语义场景补全。IEEE 计算机视觉与模式识别会议 (CVPR), 2017. 6


[38] S. Sukhbaatar, J. Weston, R. Fergus, et al. End-to-end memory networks. In Advances in Neural Information Processing Systems, pages 2440-2448, 2015. 2
[38] S. Sukhbaatar, J. Weston, R. Fergus, et al. 端到端记忆网络。载于《神经信息处理系统进展》(Advances in Neural Information Processing Systems), 页2440-2448, 2015. 2


[39] S. Thrun. Simultaneous localization and mapping. In Robotics and cognitive approaches to spatial mapping, pages 13-41. Springer, 2007. 2
[39] S. Thrun. 同时定位与地图构建。载于《机器人学与空间映射的认知方法》(Robotics and cognitive approaches to spatial mapping), 页13-41. Springer, 2007. 2


[40] M. Tomono. 3-d object map building using dense object models with sift-based recognition features. 2006 IEEE/RSJ International Conference on Intelligent Robots and Systems, pages 1885-1890, 2006. 2
[40] M. Tomono. 使用带有基于 SIFT 的识别特征的密集物体模型构建三维物体地图。2006 IEEE/RSJ 国际智能机器人与系统会议, 页1885-1890, 2006. 2


[41] T. H. Trinh, A. M. Dai, T. Luong, and Q. V. Le. Learning longer-term dependencies in rnns with auxiliary losses. In International Conference on Machine Learning (ICML), 2018. 2
[41] T. H. Trinh, A. M. Dai, T. Luong, and Q. V. Le. 通过辅助损失在 RNN 中学习更长时依赖。载于国际机器学习大会 (ICML), 2018. 2


[42] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. Kaiser, and I. Polosukhin. Attention is all you need. In Advances in Neural Information Processing Systems, 2017. 1, 2, 4, 5
[42] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. Kaiser, and I. Polosukhin. 注意力就是一切。载于《神经信息处理系统进展》(Advances in Neural Information Processing Systems), 2017. 1, 2, 4, 5


[43] D. Wierstra, A. Förster, J. Peters, and J. Schmidhuber. Recurrent policy gradients. Logic Journal of the IGPL, 18:620- 634, 2010. 1
[43] D. Wierstra, A. Förster, J. Peters, and J. Schmidhuber. 递归策略梯度。IGPL 逻辑期刊, 18:620-634, 2010. 1


[44] D. Wooden. A guide to vision-based map building. IEEE Robotics & Automation Magazine, 13:94-98, 2006. 2
[44] D. Wooden. 视觉驱动地图构建指南。IEEE 机器人与自动化杂志, 13:94-98, 2006. 2


[45] Y. Wu, Y. Wu, G. Gkioxari, and Y. Tian. Building generalizable agents with a realistic and rich 3d environment. CoRR, abs/1801.02209, 2018. 2, 6
[45] Y. Wu, Y. Wu, G. Gkioxari, and Y. Tian. 使用真实且丰富的三维环境构建可泛化代理。CoRR, abs/1801.02209, 2018. 2, 6


[46] F. Xia, A. R. Zamir, Z.-Y. He, A. Sax, J. Malik, and S. Savarese. Gibson env: Real-world perception for embodied agents. CoRR, abs/1808.10654, 2018. 1, 2
[46] F. Xia, A. R. Zamir, Z.-Y. He, A. Sax, J. Malik, and S. Savarese. Gibson 环境：面向具身智能体的真实世界感知。CoRR, abs/1808.10654, 2018. 1, 2


[47] J. Zhang, L. Tai, J. Boedecker, W. Burgard, and M. Liu. Neural slam. arXiv preprint arXiv:1706.09520, 2017. 2
[47] J. Zhang, L. Tai, J. Boedecker, W. Burgard, and M. Liu. 神经 SLAM。arXiv 预印本 arXiv:1706.09520, 2017. 2


[48] M. Zhang, Z. McCarthy, C. Finn, S. Levine, and P. Abbeel. Learning deep neural network policies with continuous memory states. In 2016 IEEE International Conference on Robotics and Automation (ICRA), pages 520-527. IEEE, 2016. 1
[48] M. Zhang, Z. McCarthy, C. Finn, S. Levine, and P. Abbeel. 带连续记忆状态的深度神经网络策略学习。载于2016 IEEE 国际机器人与自动化会议 (ICRA), 页520-527. IEEE, 2016. 1


[49] Y. Zhu, D. Gordon, E. Kolve, D. Fox, L. Fei-Fei, A. Gupta, R. Mottaghi, and A. Farhadi. Visual semantic planning using deep successor representations. IEEE International Conference on Computer Vision (ICCV), pages 483-492, 2017. 1, 2
[49] Y. Zhu, D. Gordon, E. Kolve, D. Fox, L. Fei-Fei, A. Gupta, R. Mottaghi, and A. Farhadi. 使用深度后继表示的视觉语义规划。IEEE 国际计算机视觉大会 (ICCV), 页483-492, 2017. 1, 2


[50] Y. Zhu, R. Mottaghi, E. Kolve, J. J. Lim, A. Gupta, L. Fei-Fei, and A. Farhadi. Target-driven visual navigation in indoor scenes using deep reinforcement learning. 2017 IEEE International Conference on Robotics and Automation (ICRA), pages 3357-3364, 2017. 1, 2
[50] Y. Zhu, R. Mottaghi, E. Kolve, J. J. Lim, A. Gupta, L. Fei-Fei, and A. Farhadi. 基于深度强化学习的室内场景目标驱动视觉导航。2017 IEEE 国际机器人与自动化会议 (ICRA), 页3357-3364, 2017. 1, 2