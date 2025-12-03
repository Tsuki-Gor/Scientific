# WEB AGENTS WITH WORLD MODELS: LEARNING AND LEVERAGING ENVIRONMENT DYNAMICS IN WEB NAVIGATION
# 具备世界模型的网络智能体：在网页导航中学习并利用环境动态


Hyungjoo Chae Namyoung Kim Kai Tzu-iunn Ong Minju Gwak Gwanwoo Song
蔡亨柱 金南英 翁子允 郭敏珠 宋冠佑


Yonsei University
延世大学


\{mapoout, namyoung.kim, donalee, jinyeo\}@yonsei.ac.kr
\{mapoout, namyoung.kim, donalee, jinyeo\}@yonsei.ac.kr


Ohttps://github.com/kyle8581/WMA-Agents
Ohttps://github.com/kyle8581/WMA-Agents


: https://hf.co/spaces/LangAGI-Lab/WMA-Agents
: https://hf.co/spaces/LangAGI-Lab/WMA-Agents


## ABSTRACT
## 摘要


Large language models (LLMs) have recently gained much attention in building autonomous agents. However, performance of current LLM-based web agents in long-horizon tasks is far from optimal, often yielding errors such as repeatedly buying a non-refundable flight ticket. By contrast, humans can avoid such an irreversible mistake, as we have an awareness of the potential outcomes (e.g., losing money) of our actions, also known as the "world model". Motivated by this, our study first starts with preliminary analyses, confirming the absence of world models in current LLMs (e.g., GPT-40, Claude-3.5-Sonnet, etc.). Then, we present a World-model-augmented (WMA) web agent, which simulates the outcomes of its actions for better decision-making. To overcome the challenges in training LLMs as world models predicting next observations, such as repeated elements across observations and long HTML inputs, we propose a transition-focused observation abstraction, where the prediction objectives are free-form natural language descriptions exclusively highlighting important state differences between time steps. Experiments on WebArena and Mind2Web show that our world models improve agents' policy selection without training and demonstrate superior cost- and time-efficiency compared to recent tree-search-based agents.
大型语言模型（LLMs）最近在构建自主智能体方面备受关注。然而，当前基于大语言模型的网络智能体在长周期任务中的表现远未达到最优，常常会出现诸如重复购买不可退款机票等错误。相比之下，人类能够避免此类不可挽回的错误，因为我们能意识到自身行为的潜在后果（例如损失金钱），这也被称为“世界模型”。受此启发，我们的研究首先进行了初步分析，证实当前的大语言模型（如 GPT - 40、Claude - 3.5 - Sonnet 等）中缺乏世界模型。然后，我们提出了一种世界模型增强（WMA）网络智能体，它可以模拟自身行为的结果，以便做出更好的决策。为了克服将大语言模型训练成预测下一次观测结果的世界模型时遇到的挑战，例如观测结果中存在重复元素以及 HTML 输入过长等问题，我们提出了一种聚焦于过渡的观测抽象方法，其预测目标是自由形式的自然语言描述，专门突出不同时间步之间的重要状态差异。在 WebArena 和 Mind2Web 上的实验表明，我们的世界模型无需训练即可改善智能体的策略选择，并且与最近基于树搜索的智能体相比，在成本和时间效率方面表现更优。


## 1 INTRODUCTION
## 1 引言


Large language models (LLMs) have been widely applied to solve tasks in diverse domains, including web navigation, where LLMs generate action sequences (e.g., click) to accomplish user goals on websites (Shi et al. 2017, Kim et al. 2024). Despite some success (Yao et al. 2022), LLM-based web agents' performance remains significantly poor in long-horizon environments such as WebArena (Zhou et al. 2023), where GPT-4 yields a task success rate of 14.41% whereas humans have a success rate of 78.24%. This raises a question: Why do LLMs, despite their advancements, perform much worse than humans in web navigation?
大型语言模型（LLMs）已被广泛应用于解决不同领域的任务，包括网页导航，在网页导航中，大语言模型会生成动作序列（例如点击）以实现用户在网站上的目标（Shi 等人，2017；Kim 等人，2024）。尽管取得了一些成功（Yao 等人，2022），但基于大语言模型的网络智能体在长周期环境（如 WebArena（Zhou 等人，2023））中的表现仍然显著不佳，在该环境中，GPT - 4 的任务成功率为 14.41%，而人类的成功率为 78.24%。这就引出了一个问题：尽管大语言模型取得了进展，但为何它们在网页导航中的表现远不如人类？


Humans avoid unwanted situations by considering the possible outcomes of our actions beforehand (Edwards, 1954). Such awareness of actions and outcomes is referred to as the "world model" (Forrester 1995). Meanwhile, existing LLM-based web agents rely heavily on trial and error to make decisions, as they lack world models to help them foresee the outcome of an action without actually performing it (LeCun, 2022), leading to sub-optimal decision-making that is irreversible (e.g., repeatedly buying a non-refundable item). Acknowledging the importance of world models, studies in robotics and reinforcement learning (RL) have proposed to incorporate world models for agents in navigation tasks. For instance, Du et al. (2023) and Yang et al. (2024) apply world models to simulate visual outcomes/observations of input texts or robot control. The Dreamer series use world models to predict latent state of images and use them to optimize policies, reducing the need for actual interactions in game environments (Hafner et al. 2019a, 2020; 2024).
人类会预先考虑自身行为的可能结果，从而避免出现不想要的情况（Edwards，1954）。这种对行为和结果的认知被称为“世界模型”（Forrester，1995）。与此同时，现有的基于大语言模型的网络智能体在决策时严重依赖试错法，因为它们缺乏世界模型来帮助它们在不实际执行动作的情况下预见其结果（LeCun，2022），这导致了次优且不可逆转的决策（例如，重复购买不可退款的物品）。认识到世界模型的重要性，机器人学和强化学习（RL）领域的研究已提议在导航任务的智能体中引入世界模型。例如，Du 等人（2023）和 Yang 等人（2024）应用世界模型来模拟输入文本或机器人控制的视觉结果/观测结果。Dreamer 系列使用世界模型来预测图像的潜在状态，并利用这些状态来优化策略，减少了在游戏环境中实际交互的需求（Hafner 等人，2019a，2020；2024）。


Motivated by these, this paper begins by investigating SOTA LLMs' understanding of "environment dynamics", i.e., the association between actions and environment states. We reveal that (i) current LLMs (e.g., GPT-40 and Claude-3.5-Sonnet) struggle with predicting the outcomes of their actions and (ii) the awareness of potential outcomes helps them make decisions aligning with user goals. Upon these findings, we present a World-Model-Augmented (WMA) web agent, which simulates the outcomes of its actions for better decision-making. However, naively training a world model to predict the next observation state (i.e., the entire webpage) can lead to a large amount of repeated elements across observations and long HTML inputs, negatively affecting model performance. Thus, we propose a novel transition-focused observation abstraction, where the world model is trained to generate free-form natural language descriptions exclusively highlighting important state differences between time steps (e.g., an updated price on the website). During inference, our agent first simulates the outcome (i.e., next observation) of each action candidate (from the policy model) using the world model. Then, a value function estimates the rewards of all simulated observations, helping the agent select a final action with the highest estimated reward. Our contributions are two-fold:
受此启发，本文首先考察了 SOTA 大型模型对“环境动态”（即动作与环境状态之间关联）的理解。我们揭示了 (i) 现有 LLM（如 GPT-40 与 Claude-3.5-Sonnet）在预测其动作结果方面存在困难，且 (ii) 对潜在结果的认知有助于其做出与用户目标一致的决策。基于这些发现，我们提出了一个世界模型增强（WMA）的网页代理，它通过模拟动作的结果来改进决策。然而，简单地训练世界模型去预测下一观测状态（即整个网页）会导致观测之间出现大量重复元素与冗长的 HTML 输入，从而对模型性能产生负面影响。因此，我们提出了一种新颖的以转变为中心的观测抽象，世界模型仅被训练生成自由形式的自然语言描述，专注突出时间步之间的重要状态差异（例如网站上更新的价格）。在推理时，代理首先用世界模型模拟每个动作候选（由策略模型给出）的结果（即下一观测）。随后，价值函数评估所有模拟观测的回报，帮助代理选择估计回报最高的最终动作。我们的贡献有两点：


- We are the first to pioneer world models in LLM-based web agents, laying the groundwork for policy adaptation through simulated environment feedback in web navigation.
- 我们首次在基于 LLM 的网页代理中引入世界模型，为通过模拟环境反馈进行策略适配在网页导航领域奠定基础。


- We present a novel transition-focused observation abstraction for training LLMs as world models. We show that using world models trained with this method can improve action selection by simulating the action candidates without training the policy models. Also, we demonstrate our agents' cost- and time-efficiency compared to recent tree-search-based agents (Koh et al. 2024), by 6.8x and 5.3x, respectively.
- 我们提出了一种用于训练 LLM 作为世界模型的新颖以转变为中心的观测抽象。我们展示了用此方法训练的世界模型可以通过模拟动作候选来改进动作选择，而无需训练策略模型。此外，我们证明了相较于最新的基于树搜索的代理（Koh et al. 2024），我们的代理在成本和时间上分别高效 6.8 倍与 5.3 倍。


## 2 RELATED WORK
## 2 相关工作


Benchmarks for web agents. Many benchmarks have been introduced to evaluate LLM-based agents' ability in web navigation (Kim et al. 2024). MiniWoB (Shi et al. 2017) and Mini-WoB++ (Liu et al. 2018) are among the first widely adopted benchmarks. More recently, Web-Shop (Yao et al. 2022) simulates e-commerce environments where agents are tested to execute tasks based on text instructions on the web. These early benchmarks are limited to specific and constrained domains. Mind2Web (Deng et al. 2024) curates web tasks across more diverse domains, and WebArena (Zhou et al. 2023) emphasizes functional correctness and more realistic scenarios (e.g., posting articles on Reddit) in simulated environment. We adopt Mind2Web and WebArena for evaluation for their generalizability and resemblance of real-world web interactions.
网页代理基准。近年来引入了许多基准用于评估基于 LLM 的代理在网页导航方面的能力（Kim et al. 2024）。MiniWoB（Shi et al. 2017）和 Mini-WoB++（Liu et al. 2018）是最早被广泛采用的基准之一。近来，Web-Shop（Yao et al. 2022）模拟了电子商务环境，在网页上测试代理根据文本指令执行任务。这些早期基准局限于特定且受限的领域。Mind2Web（Deng et al. 2024）策划了更为多样领域的网页任务，WebArena（Zhou et al. 2023）在模拟环境中强调功能正确性与更真实的场景（例如在 Reddit 发布文章）。我们采用 Mind2Web 与 WebArena 进行评估，因其更具普适性且更接近真实世界的网页交互。


LLM-based web agents. In recent years, LLM-based agents have become popular in the web navigation domain. However, since many powerful proprietary LLMs do not provide access to model parameters, many studies of web navigation have been focusing on training-free methods where LLMs directly learn from user inputs (i.e., prompts) without task-specific training (Sodhi et al. 2023; Zheng et al. 2023). For instance, Wilbur (Lutz et al. 2024) and Agent Workflow Memory (Wang et al. 2024b) leverage a verification model (Pan et al. 2024) with prompt-based methods to collect successful trajectory data for guiding the agent's policy at inference time. AutoEval (Pan et al. 2024) and Tree search agent (Koh et al. 2024) increase the number of trials and reasoning paths, further improving system performance. However, due to their trial-and-error nature, these approaches can not only be computationally inefficient in gathering trajectories as tasks become more complex but also are more prone to undesired results (e.g., booking a non-refundable ticket). Our WMA web agent reduces such risks via a world model, which predicts future observations and the rewards of their corresponding action candidates before actually making an action. Furthermore, our approach can be orthogonally applied to many of the existing methods.
基于 LLM 的网页代理。近年基于 LLM 的代理在网页导航领域变得流行。然而，由于许多强大的专有 LLM 不开放模型参数，很多网页导航研究聚焦于无训练方法，即 LLM 直接从用户输入（即提示）中学习而无需任务专用训练（Sodhi et al. 2023；Zheng et al. 2023）。例如，Wilbur（Lutz et al. 2024）与 Agent Workflow Memory（Wang et al. 2024b）利用验证模型（Pan et al. 2024）结合基于提示的方法收集成功轨迹数据以在推理时引导代理策略。AutoEval（Pan et al. 2024）与树搜索代理（Koh et al. 2024）通过增加试验与推理路径数量进一步提升系统性能。然而，由于其反复试验的特性，这些方法在任务变得更复杂时不仅在收集轨迹上计算效率低下，而且更容易产生不期望的结果（例如预订不可退票）。我们的 WMA 网页代理通过世界模型预测未来观测及其对应动作候选的回报，在实际执行动作前降低了此类风险。此外，我们的方法可以与许多现有方法正交地结合使用。


World models in autonomous agents. World models refer to systems that generate internal representations of the world, predicting the effects of their actions on environments (LeCun 2022). In RL, simulating observations and environmental feedback using world models allow the policy model to learn (Sutton 1990) or plan (Ha & Schmidhuber 2018; Hafner et al. 2019b) without actually interacting with the environment. While some world models are trained with raw observations (Oh et al. 2015, Chiappa et al. 2017), others are built on latent representations (Hafner et al. 2019a, 2020, Kipf et al. 2020). For instance, in the image domain, Hafner et al. (2020) train a world model by training it to first compute a posterior stochastic state based on the current image and then a prior stochastic state that tries to predict the posterior without access to the image. Within the field of LLMs, Zhang et al. (2024) convert visual observations into natural language and employs an LLM-based world model for text-based games, and Wang et al. (2024a) further transform observations into a structural format (e.g., JSON), improving LLMs' reasoning over state transition functions. In web navigation, environments are built upon not only natural language but on more complex text modalities such as HTML and DOM trees. We address this by transforming them to a novel free-form description, highlighting the state difference between each time step.
自主智能体中的世界模型。世界模型指的是生成世界内部表征的系统，用于预测其行动对环境的影响（LeCun 2022）。在强化学习（RL）中，使用世界模型模拟观测和环境反馈，可使策略模型在不与环境实际交互的情况下进行学习（Sutton 1990）或规划（Ha & Schmidhuber 2018；Hafner 等人 2019b）。有些世界模型通过原始观测进行训练（Oh 等人 2015，Chiappa 等人 2017），而另一些则基于潜在表征构建（Hafner 等人 2019a、2020，Kipf 等人 2020）。例如，在图像领域，Hafner 等人（2020）训练世界模型时，先基于当前图像计算后验随机状态，再计算一个尝试在不访问图像的情况下预测后验的先验随机状态。在大语言模型（LLMs）领域，Zhang 等人（2024）将视觉观测转换为自然语言，并采用基于大语言模型的世界模型用于文本游戏，Wang 等人（2024a）进一步将观测转换为结构化格式（如 JSON），提升了大语言模型对状态转移函数的推理能力。在网页导航中，环境不仅基于自然语言构建，还基于 HTML 和 DOM 树等更复杂的文本形式。我们通过将其转换为一种新颖的自由格式描述来解决这个问题，突出每个时间步之间的状态差异。


## 3 PRELIMINARY ANALYSES: ARE CURRENT LLMS AWARE OF ENVIRONMENT DYNAMICS IN WEB NAVIGATION?
## 3 初步分析：当前的大语言模型是否了解网页导航中的环境动态？


We first start with investigating whether LLMs can understand the association between actions and their effects on the environment, i.e., understand the environment dynamics. We conduct analyses addressing these two questions:
我们首先研究大语言模型是否能理解行动与其对环境影响之间的关联，即理解环境动态。我们进行分析以解决以下两个问题：


- Preliminary question I: Are LLMs aware of the outcomes of their actions?
- 初步问题 I：大语言模型是否了解其行动的结果？


- Preliminary question II: When having access to the outcome of each action candidate, can LLMs select an optimal action aligning with the user objective?
- 初步问题 II：当能够获取每个候选行动的结果时，大语言模型能否选择与用户目标一致的最优行动？


For the analyses, we sample 100 user instructions from WebArena and annotate human trajectories within the environment. Each instance has a user instruction, the current state, a human-annotated golden action, and the corresponding next state resulting from the golden action. We analyze 4 popular closed-source SOTA LLMs: GPT-40-mini (Zhu et al. 2023), GPT-40, GPT-4-Turbo (OpenAI 2023), and Claude-3.5-Sonnet (Anthropic 2024). More details are in Appendix B
为进行分析，我们从 WebArena 中抽取了 100 条用户指令，并标注了环境中的人类轨迹。每个实例包含一条用户指令、当前状态、人工标注的黄金行动以及该黄金行动对应的下一状态。我们分析了 4 种流行的闭源最先进大语言模型：GPT - 40 - mini（Zhu 等人 2023）、GPT - 40、GPT - 4 - Turbo（OpenAI 2023）和 Claude - 3.5 - Sonnet（Anthropic 2024）。更多细节见附录 B


3.1 PRELIMINARY ANALYSIS I - LLMS STRUGGLE WITH PREDICTING THE NEXT STATES CAUSED BY THEIR ACTIONS
3.1 初步分析 I - 大语言模型难以预测其行动导致的下一状态


Setups. We test LLMs' ability to predict the outcomes of actions on the web via a binary classification task. Given the current state and the golden action, the LLM is prompted to select the correct next state from (i) the golden next state and (ii) a lexically similar yet incorrect next state retrieved from the same trajectory. We calculate the lexical similarity with difflib (Python 2024). We assess classification accuracy.
设置。我们通过二分类任务测试大语言模型预测网页上行动结果的能力。给定当前状态和黄金行动，提示大语言模型从（i）黄金下一状态和（ii）从同一轨迹中检索到的词汇上相似但错误的下一状态中选择正确的下一状态。我们使用 difflib（Python 2024）计算词汇相似度。我们评估分类准确率。


<img src="https://cdn.noedgeai.com/bo_d4nfperef24c73bbffn0_2.jpg?x=1000&y=1308&w=485&h=197&r=0"/>



Figure 1: LLMs' performance in next state prediction.
图 1：大语言模型在下一状态预测中的表现。


Results. Figure 1 reveals that under vanilla settings, current LLMs cannot effectively predict the next states caused by their actions. First, all adopted LLMs (54.75% on average) lose significantly to humans. Also, Claude-3.5-Sonnet performs almost as badly as random guessing. These suggest that the world model, the ability to foresee the potential outcomes of actions taken, is absent in LLMs.
结果。图 1 显示，在常规设置下，当前的大语言模型无法有效预测其行动导致的下一状态。首先，所有采用的大语言模型（平均 54.75%）的表现明显逊于人类。此外，Claude - 3.5 - Sonnet 的表现几乎和随机猜测一样差。这些表明大语言模型缺乏预见所采取行动潜在结果的世界模型能力。


3.2 PRELIMINARY ANALYSIS II - LLMS MAKE BETTER ACTION SELECTION WHEN ACCESSING THE OUTCOME OF EACH ACTION CANDIDATE
3.2 初步分析 II - 当能够获取每个候选行动的结果时，大语言模型能更好地选择行动


Setups. We assess whether LLMs can select a correct action that aligns with the user goal when they are provided with the outcome of each action candidate. Given the current state, 10 action candidates, and their corresponding outcomes/next states, the LLM is prompted to differentiate the golden action from other 9 negative actions.
设置。我们评估当大语言模型获得每个候选行动的结果时，它们能否选择与用户目标一致的正确行动。给定当前状态、10 个候选行动及其对应的结果/下一状态，提示大语言模型从其他 9 个负行动中区分出黄金行动。


<img src="https://cdn.noedgeai.com/bo_d4nfperef24c73bbffn0_2.jpg?x=1035&y=1849&w=440&h=264&r=0"/>



Figure 2: LLMs' performance in action selection (w/ and w/o next states).
图 2：大语言模型在行动选择中的表现（有和没有下一状态）。


Results. Figure 2 compares LLMs' performance in differentiating the golden action from negative actions when they are/are not provided with the resulting next state of each candidate action. We find that current SOTA LLMs have difficulty in selecting correct actions when they can only rely on the current observations/states (striped bars), yielding an average accuracy of only 49%. However, when augmented with the corresponding next state of each action candidate, they demonstrate huge performance gains (up to ${38}\%$ improvement) in selecting correct actions. When only the current state and the user objective are provided, GPT-40 yields an accuracy of 53%. In contrast, when the next state is given, performance rises to ${73}\%$ .
结果。图 2 比较了大语言模型（LLMs）在有和没有提供每个候选动作的后续状态时，从负面动作中区分出最佳动作的表现。我们发现，当前最先进的大语言模型在只能依靠当前观察/状态（条纹条柱）来选择正确动作时存在困难，平均准确率仅为 49%。然而，当为其提供每个动作候选对应的后续状态时，它们在选择正确动作方面表现出巨大的性能提升（最高提升 ${38}\%$）。当仅提供当前状态和用户目标时，GPT - 40 的准确率为 53%。相比之下，当提供后续状态时，准确率升至 ${73}\%$。


<img src="https://cdn.noedgeai.com/bo_d4nfperef24c73bbffn0_3.jpg?x=307&y=238&w=1182&h=482&r=0"/>



Figure 3: Framework overview. We first collect training data for world models (top). After training, we perform policy optimization by selecting the action leading to an optimal next state (bottom).
图 3：框架概述。我们首先收集世界模型的训练数据（顶部）。训练完成后，我们通过选择能导向最优后续状态的动作来进行策略优化（底部）。


### 3.3 INSIGHTS FROM PRELIMINARY ANALYSES
### 3.3 初步分析的见解


Through our preliminary analyses, we have demonstrated that: (i) Web agents built with SOTA LLMs are bad at predicting how their actions affect next states; (ii) When being aware of how an action affects the next state, LLMs can make better decisions. These findings highlight the necessity of world models in LLM-based web agents, pointing out a promising direction for facilitating better web agents in complex, long-horizon navigation tasks.
通过初步分析，我们证明了：（i）基于最先进大语言模型构建的网络代理不擅长预测其动作如何影响后续状态；（ii）当了解动作如何影响后续状态时，大语言模型可以做出更好的决策。这些发现凸显了在基于大语言模型的网络代理中使用世界模型的必要性，为在复杂、长视野的导航任务中实现更优的网络代理指明了一个有前景的方向。


## 4 WORLD-MODEL-AUGMENTED WEB AGENTS
## 4 世界模型增强型网络代理


Motivated by the above insights, we present a novel framework for World-Model-Augmented (WMA) web agents, LLM-based web agents equipped with world models. The world models learn/leverage environment dynamics (i.e., association of actions and outcomes) to simulate plausible next observations of agents' actions, facilitating better decisions (i.e., polices) in web navigation.
受上述见解的启发，我们提出了一种用于世界模型增强型（WMA）网络代理的新颖框架，这是一种配备了世界模型的基于大语言模型的网络代理。世界模型学习/利用环境动态（即动作与结果的关联）来模拟代理动作可能产生的后续观察，从而促进在网络导航中做出更好的决策（即策略）。


Formulations. Since web agents access only information in the viewport (i.e., users' visible area), we model web navigation as a partially observable Markov decision process (POMDP). We consider a web environment $\mathcal{E}$ with: (i) a hidden state space $\mathcal{S}$ ; (ii) an action space $\mathcal{A}$ ,including language-guided actions (e.g., CLICK, TYPE, HOVER, etc.) and their descriptions; (iii) an observation space $\mathcal{O}$ representing an accessibility tree of the page,which is a simplified DOM tree (Zhou et al.,2023).
公式化。由于网络代理仅能访问视口（即用户可见区域）内的信息，我们将网络导航建模为部分可观察马尔可夫决策过程（POMDP）。我们考虑一个网络环境 $\mathcal{E}$，其具有：（i）一个隐藏状态空间 $\mathcal{S}$；（ii）一个动作空间 $\mathcal{A}$，包括语言引导的动作（如点击、输入、悬停等）及其描述；（iii）一个观察空间 $\mathcal{O}$，表示页面的可访问性树，这是一个简化的 DOM 树（Zhou 等人，2023）。


In the POMDP,the agent receives a new partial observation ${o}_{t + 1} \in  \mathcal{O}$ from $\mathcal{E}$ after performing an action ${a}_{t} \in  \mathcal{A}$ based on ${o}_{t}$ . Such state transition ${s}_{t} \rightarrow  {s}_{t + 1}$ is managed by a golden transition function $\mathcal{T} : \mathcal{S} \times  \mathcal{A} \rightarrow  \mathcal{S}$ provided in the environment.
在 POMDP 中，代理基于 ${o}_{t}$ 执行动作 ${a}_{t} \in  \mathcal{A}$ 后，会从 $\mathcal{E}$ 接收一个新的部分观察 ${o}_{t + 1} \in  \mathcal{O}$。这种状态转移 ${s}_{t} \rightarrow  {s}_{t + 1}$ 由环境中提供的最优转移函数 $\mathcal{T} : \mathcal{S} \times  \mathcal{A} \rightarrow  \mathcal{S}$ 管理。


### 4.1 WORLD MODEL TRAINING
### 4.1 世界模型训练


We hereby introduce the training process of our world models. As shown in Figure 3 (top), our training consists of three main steps:
在此，我们介绍世界模型的训练过程。如图 3（顶部）所示，我们的训练包括三个主要步骤：


#### 4.1.1 STEP I: HARVESTING AGENT-ENVIRONMENT INTERACTION DATA
#### 4.1.1 步骤 I：收集代理 - 环境交互数据


We start by collecting the dataset $\mathcal{D} = \mathop{\sum }\limits_{{t = 1}}^{n}\left\{  {I,{o}_{t},{a}_{t},{o}_{t + 1}}\right\}$ from the environment $\mathcal{E}$ for training world models. For that, we prompt an LLM as web agent to achieve the goal provided in the user instruction $I$ ,by iteratively predicting an action ${a}_{t}$ based on the current observation ${o}_{t}$ throughout all $n$ time steps. Consequently,we obtain $\mathcal{D}$ from trajectory $\tau  = \left\{  {{o}_{1},{a}_{1},{o}_{2},\ldots ,{a}_{n},{o}_{n + 1}}\right\}$ based on $I$ ,and environment states of $n$ time steps $\left\{  {{s}_{1},\ldots ,{s}_{n + 1}}\right\}   \subset  \mathcal{S}$ obtained via transition function $\mathcal{T}$ .
我们首先从环境 $\mathcal{E}$ 中收集数据集 $\mathcal{D} = \mathop{\sum }\limits_{{t = 1}}^{n}\left\{  {I,{o}_{t},{a}_{t},{o}_{t + 1}}\right\}$ 用于训练世界模型。为此，我们将一个大语言模型作为网络代理，通过在所有 $n$ 个时间步中基于当前观察 ${o}_{t}$ 迭代预测动作 ${a}_{t}$，来实现用户指令 $I$ 中提供的目标。因此，我们基于 $I$ 从轨迹 $\tau  = \left\{  {{o}_{1},{a}_{1},{o}_{2},\ldots ,{a}_{n},{o}_{n + 1}}\right\}$ 中获得 $\mathcal{D}$，并通过转移函数 $\mathcal{T}$ 获得 $n$ 个时间步的环境状态 $\left\{  {{s}_{1},\ldots ,{s}_{n + 1}}\right\}   \subset  \mathcal{S}$。


#### 4.1.2 STEP II: TRANSITION-FOCUSED OBSERVATION ABSTRACTION
#### 4.1.2 步骤 II：以转换为中心的观察抽象


With the collected data $\mathcal{D} = \mathop{\sum }\limits_{{t = 1}}^{n}\left\{  {I,{o}_{t},{a}_{t},{o}_{t + 1}}\right\}$ ,it is intuitive to train LLM-based world models to predict ${o}_{t + 1}$ ,which is expressed with texts (e.g.,HTML and accessibility tree) (Deng et al.,2024) Zhou et al. 2023). However, simply using textual observations to represent environment states and use them as training objectives may introduce the following downsides:
利用收集到的数据 $\mathcal{D} = \mathop{\sum }\limits_{{t = 1}}^{n}\left\{  {I,{o}_{t},{a}_{t},{o}_{t + 1}}\right\}$，直观的做法是训练基于 LLM 的世界模型来预测 ${o}_{t + 1}$，其以文本形式表示（例如 HTML 和无障碍树）(Deng et al.,2024) Zhou et al. 2023)。然而，单纯用文本观察来表示环境状态并将其作为训练目标可能引入以下缺点：


- Low information gain during training: State transitions in websitesoften involve altering only a part of the previous observation (e.g., a drop-down menu is clicked). As a result, most information in ${o}_{t + 1}$ remains the same as it is in ${o}_{t}$ . Therefore,predicting the entire textual observation from scratch may result in low information gain during training.
- 训练时信息增益低：网站中的状态转换通常只改变前一观察的一部分（例如点击了下拉菜单）。因此 ${o}_{t + 1}$ 中的大部分信息在 ${o}_{t}$ 中保持不变。于是从头预测整个文本观察在训练中可能导致信息增益低。


- Excessively long sequence length: Processing the whole text-based observations can lead to excessively long sequence length and consequently high computational costs. Indeed, this can be partially mitigated by replacing raw HTML with an accessibility tree (relatively simple), using it as LLMs' training objectives still introduce a long sequence length (4K tokens on average, see Figure 4).
- 序列长度过长：处理整段基于文本的观察会导致序列长度过长，从而带来高计算成本。确实，虽然用无障碍树替换原始 HTML（相对更简洁）可以部分缓解，但作为 LLM 的训练目标仍会引入较长序列（平均约 4K 令牌，见图 4）。


To address the above bottleneck in training text-based models (i.e., LLMs) as world models, we draw inspiration from how the RL community conventionally implements world models: using estimated latent vectors as summaries of raw visual observations, reducing memory footprints for effectively learning environment dynamics (Doerr et al. 2018; Hafner et al. 2019a) - We thus propose to abstract raw text observations, with a focus on state transition between consecutive observations, for obtaining better training objectives.
为了解决将基于文本的模型（即 LLM）作为世界模型训练时的上述瓶颈，我们借鉴强化学习社区实现世界模型的常规做法：使用估计的潜在向量来总结原始视觉观察，减少内存占用以更有效地学习环境动力学（Doerr et al. 2018; Hafner et al. 2019a）——因此我们提出对原始文本观察进行抽象，聚焦于连续观察之间的状态转换，以获得更好的训练目标。


<img src="https://cdn.noedgeai.com/bo_d4nfperef24c73bbffn0_4.jpg?x=989&y=1044&w=497&h=266&r=0"/>



Figure 4: Sequence length distribution of different observation representations.
图 4：不同观察表示的序列长度分布。


To collect abstracted next observations for training world models, one may simply run an off-the-shelf summarizer on ${o}_{t + 1}$ collected in Step I. However,while reducing sequence length,this does not address the low information gain caused by repeated elements between ${o}_{t}$ and ${o}_{t + 1}$ . Thus,instead of such a naive approach, as shown in Figure 5, we first (i) apply the Hungarian algorithm (Kuhn, 1995) to calculate a cost matrix for matching elements between ${o}_{t}$ and ${o}_{t + 1}$ and (ii) mechanically transform the results into a list of state transition $\Delta \left( {{o}_{t},{o}_{t + 1}}\right)$ ,pointing out UPDATED,DELETED, and ADDED elements on the web. After that,we prompt an LLM to convert the extracted $\Delta \left( {{o}_{t},{o}_{t + 1}}\right)$ into a free-from natural language description ${\widetilde{o}}_{t + 1}$ ,which highlights the difference between the new observation ${o}_{t + 1}$ and ${o}_{t}$ . Replacing ${o}_{t + 1}$ in $\mathcal{D} = \left\{  {I,{o}_{t},{a}_{t},{o}_{t + 1}}\right\}$ collected in Step I with ${\widetilde{o}}_{t + 1}$ we just acquired here,we get a final dataset $\widetilde{\mathcal{D}} = \mathop{\sum }\limits_{{t = 1}}^{n}\left\{  {I,{o}_{t},{a}_{t},{\widetilde{o}}_{t + 1}}\right\}$ for training world models.
为了收集用于训练世界模型的抽象后续观察，可以直接对步骤 I 中收集的 ${o}_{t + 1}$ 运行现成的摘要器。然而，尽管能缩短序列长度，这并不能解决 ${o}_{t}$ 与 ${o}_{t + 1}$ 之间重复元素导致的信息增益低的问题。因此，如图 5 所示，我们首先 (i) 应用匈牙利算法 (Kuhn, 1995) 计算 ${o}_{t}$ 与 ${o}_{t + 1}$ 之间元素匹配的代价矩阵，(ii) 机械地将结果转换为状态转换列表 $\Delta \left( {{o}_{t},{o}_{t + 1}}\right)$，指出网页上的 UPDATED、DELETED 和 ADDED 元素。之后，我们提示 LLM 将提取的 $\Delta \left( {{o}_{t},{o}_{t + 1}}\right)$ 转换为自由格式的自然语言描述 ${\widetilde{o}}_{t + 1}$，以突出新观察 ${o}_{t + 1}$ 与 ${o}_{t}$ 之间的差异。将步骤 I 中收集的 $\mathcal{D} = \left\{  {I,{o}_{t},{a}_{t},{o}_{t + 1}}\right\}$ 中的 ${o}_{t + 1}$ 替换为我们在此处获得的 ${\widetilde{o}}_{t + 1}$，即可得到用于训练世界模型的最终数据集 $\widetilde{\mathcal{D}} = \mathop{\sum }\limits_{{t = 1}}^{n}\left\{  {I,{o}_{t},{a}_{t},{\widetilde{o}}_{t + 1}}\right\}$。


#### 4.1.3 STEP III: Learning Environment Dynamics
#### 4.1.3 步骤 III：学习环境动力学


Lastly,using $\widetilde{\mathcal{D}}$ ,we proceed to train the internal world model $\phi$ of the web agent to learn the environment dynamics. Formally, an LLM working as the world model is trained to predict the abstracted observation $\widetilde{o}$ of the next state ${s}_{t + 1}$ ,given three inputs: the user instruction $I$ ,the current observation ${o}_{t}$ ,and the current action ${a}_{t}$ . This LLM is trained to minimize the following loss term via the next-token prediction objective:
最后，使用 $\widetilde{\mathcal{D}}$，我们继续训练 Web 代理的内部世界模型 $\phi$ 以学习环境动力学。形式化地，作为世界模型的 LLM 在给定三项输入：用户指令 $I$、当前观察 ${o}_{t}$ 和当前动作 ${a}_{t}$ 的情况下，被训练去预测下一状态 ${s}_{t + 1}$ 的抽象观察 $\widetilde{o}$。该 LLM 通过下一个令牌预测目标最小化以下损失项：


$$
{\mathcal{L}}_{\phi } =  - \log \mathop{\sum }\limits_{{\left( {\widetilde{o},o,a,I}\right)  \in  \widetilde{\mathcal{D}}}}p\left( {{\widetilde{o}}_{t + 1} \mid  {o}_{t},{a}_{t},I}\right) \tag{1}
$$



Through this process, this LLM learns the environment dynamics, working as a world model that helps the web agent to foresee the potential outcome (i.e., predict the next observation) of its action.
通过此过程，该大模型学习环境动态，作为一个世界模型，帮助网页代理预见其动作的潜在结果（即预测下一次观察）。


<img src="https://cdn.noedgeai.com/bo_d4nfperef24c73bbffn0_5.jpg?x=311&y=228&w=1182&h=246&r=0"/>



Figure 5: The overview of transition-focused observation abstraction.
图5：面向转移的观察抽象概览。


### 4.2 INFERENCE-TIME POLICY OPTIMIZATION WITH THE WORLD MODEL
### 4.2 使用世界模型进行推理时策略优化


In this section,we explain how we use the developed world model $\phi$ to improve LLM-based agents’ performance in web navigation. As illustrated in Figure 3 (bottom), the web agent consists of three main components: (i) a policy model $\theta$ ; (ii) our world model $\phi$ ; (iii) a value function $V$ . Note that the policy model $\theta$ is frozen,i.e.,we do not update its parameters.
在本节中，我们解释如何使用所构建的世界模型 $\phi$ 来提升基于大模型的网页代理在网页导航中的表现。如图3（下）所示，网页代理由三部分组成：(i) 策略模型 $\theta$；(ii) 我们的世界模型 $\phi$；(iii) 价值函数 $V$。注意策略模型 $\theta$ 是冻结的，即我们不更新其参数。


During inference at time $t$ with a current observation ${o}_{t}$ ,WMA web agent ought to utilize the world model $\phi$ to foresee how an action can affect the state (i.e.,predict ${\widetilde{o}}_{t + 1}^{i}$ ),and accordingly find an optimal action/policy ${a}_{t}$ from the policy model $\theta$ that can lead to the target goal defined in $\mathcal{I}$ .
在时间 $t$ 的推理过程中，面对当前观察 ${o}_{t}$，WMA 网页代理应利用世界模型 $\phi$ 预见一个动作如何影响状态（即预测 ${\widetilde{o}}_{t + 1}^{i}$），并据此从策略模型 $\theta$ 中找到能导致在 $\mathcal{I}$ 中定义目标的最优动作/策略 ${a}_{t}$。


We begin by sampling $k$ action candidates $\left\{  {{a}_{t}^{1},{a}_{t}^{2},\ldots ,{a}_{t}^{k}}\right\}$ from $\theta$ via top- $p$ decoding (Holtzman et al. 2019),to conduct diverse exploration on future observations $\left\{  {{o}_{t + 1}^{1},{o}_{t + 1}^{2},\ldots ,{o}_{t + 1}^{k}}\right\}$ similar to Koh et al. (2024). Next,we use the world model $\phi$ to "simulate" the potential next observation ${\widetilde{o}}_{t + 1}^{i}$ caused by each action candidate ${a}_{t}$ :
我们首先从 $\theta$ 通过 top- $p$ 解码（Holtzman et al. 2019）采样 $k$ 个动作候选项 $\left\{  {{a}_{t}^{1},{a}_{t}^{2},\ldots ,{a}_{t}^{k}}\right\}$，以对未来观测 $\left\{  {{o}_{t + 1}^{1},{o}_{t + 1}^{2},\ldots ,{o}_{t + 1}^{k}}\right\}$ 进行类似于 Koh et al. (2024) 的多样化探索。接着，我们使用世界模型 $\phi$ 来“模拟”每个动作候选 ${a}_{t}$ 所导致的潜在下一个观测 ${\widetilde{o}}_{t + 1}^{i}$：


$$
{\left\{  {\widetilde{o}}_{t + 1}^{i}\right\}  }_{i = 1}^{k} = {\left\{  \phi \left( {o}_{t},{a}_{t}^{i},I\right) \right\}  }_{i = 1}^{k} \tag{2}
$$



Note that each ${\widetilde{o}}_{t + 1}^{i}$ is a free-form description of the next observation,as shown in Figure 5 (right).
注意每个 ${\widetilde{o}}_{t + 1}^{i}$ 是对下一次观察的自由形式描述，如图5（右）所示。


Lastly, we decide the agent's action for actual operation by selecting the action leading to the most optimal future state ${s}_{t + 1}$ from all action candidates. Following Koh et al. (2024),we use an off-the-shelf LLM as a value function $V\left( \cdot \right)$ to estimate the reward yielded by each action candidate and select the action ${\widehat{a}}_{t}$ with the highest reward:
最后，我们通过从所有动作候选中选择导致最优未来状态 ${s}_{t + 1}$ 的动作，决定代理的实际操作。按照 Koh et al. (2024)，我们使用现成的大模型作为价值函数 $V\left( \cdot \right)$ 来估计每个动作候选带来的奖励，并选择奖励最高的动作 ${\widehat{a}}_{t}$：


$$
{\widehat{a}}_{t} = \mathop{\operatorname{argmax}}\limits_{{{a}_{t} \in  \left\{  {{a}_{t}^{1},\ldots ,{a}_{t}^{k}}\right\}  }}V\left( {I,{o}_{t},{a}_{t},{\widetilde{o}}_{t + 1}^{i}}\right) \tag{3}
$$



With this process, we are able to optimize the policy selection of web agents in inference time without training the policy models. This training-free augmentation of world models allows us to easily adapt our world model $\phi$ to existing web agents,including prompt-based (Pan et al.,2024) Wang et al. 2024b) and fine-tuned LLMs (Gur et al., 2023; Lai et al., 2024).
通过此过程，我们能够在推理时优化网页代理的策略选择，而无需训练策略模型。这种无训练的世界模型增强使我们可以轻松将世界模型 $\phi$ 适配到现有网页代理上，包括基于提示的（Pan et al.,2024；Wang et al. 2024b）和微调的大模型（Gur et al., 2023；Lai et al., 2024）。


## 5 EXPERIMENTS
## 5 实验


### 5.1 SETUPS AND IMPLEMENTATION DETAILS
### 5.1 设置与实现细节


Benchmarks and evaluation metrics. For evaluation, we use the official WebArena and Mind2Web benchmarks (Zhou et al. 2023, Deng et al. 2024). WebArena includes 812 real-life tasks in simulated environments across five different websites, spanning four key domains - e-commerce (Shopping), social forums (Reddit), collaborative software development (Gitlab), content management (CMS), and Map. Details of each domain are further explained in Appendix C.3 The main metric, Success Rate (SR), is calculated as the percentage of the user instructions that are successfully accomplished by the generated agent trajectory. On the other hand, Mind2Web (Deng et al. 2024) covers over 2,000 open-ended tasks, collected from 137 websites of 31 domains and crowd-sourced action sequences for the tasks. Along with the SR, Mind2Web also uses Step SR, which measures whether the predicted action selects both the correct action type (action ${\mathrm{F}}_{1}$ ) and element ID (element accuracy). When the agent succeeds in all steps in a trajectory, it is evaluated as success.
基准与评估指标。为评估我们使用官方的 WebArena 与 Mind2Web 基准（Zhou et al. 2023，Deng et al. 2024）。WebArena 包含在五个不同网站的模拟环境中共 812 个真实任务，覆盖四个主要领域——电子商务（Shopping）、社交论坛（Reddit）、协作软件开发（Gitlab）、内容管理（CMS）和地图。各领域详情在附录 C.3 中进一步说明。主要指标成功率（SR）被计算为被生成的代理轨迹成功完成的用户指令的百分比。另一方面，Mind2Web（Deng et al. 2024）涵盖 2000 多个开放式任务，收集自 31 个领域的 137 个网站，并为任务众包行动序列。除 SR 外，Mind2Web 还使用步骤成功率（Step SR），该指标衡量预测动作是否同时选择正确的动作类型（动作 ${\mathrm{F}}_{1}$）和元素 ID（元素准确性）。当代理在轨迹的所有步骤中都成功时，评估为成功。


Table 1: Agent performance in WebArena. $\Delta$ : relative performance gains from policy optimization.
表1：WebArena 中的代理性能。$\Delta$：来自策略优化的相对性能提升。


<table><tr><td rowspan="2">Policy LLMs</td><td rowspan="2">Methods</td><td rowspan="2">Max Actions</td><td colspan="2">Success Rate (SR)</td><td rowspan="2">$\Delta$</td></tr><tr><td>w/o Action Selection</td><td>w/ Action Selection</td></tr><tr><td></td><td rowspan="4">AutoEval (Pan et al. 2024) <br> BrowserGym (Drouin et al., 2024) <br> SteP (Sodhi et al. 2023 <br> AWM wang et al. 2024b)</td><td rowspan="4">30</td><td>20.2%</td><td>-</td><td>-</td></tr><tr><td>GPT-4</td><td>23.5%</td><td>-</td><td>-</td></tr><tr><td></td><td>35.8%</td><td>-</td><td>-</td></tr><tr><td></td><td>35.5%</td><td>-</td><td>-</td></tr><tr><td rowspan="3">GPT-4o</td><td>Vanilla CoT (Zhou et al. 2023)</td><td>30</td><td>13.1%</td><td>-</td><td>-</td></tr><tr><td>Tree search agent (Koh et al., 2024)</td><td>5</td><td>15.0%</td><td>19.2%</td><td>+28.0%</td></tr><tr><td>WMA web agent (ours)</td><td>5</td><td>12.8%</td><td>16.6%</td><td>+29.7%</td></tr><tr><td>GPT-4o-mini</td><td>WMA web agent (ours)</td><td>5</td><td>9.4%</td><td>13.5%</td><td>+43.6%</td></tr></table>
<table><tbody><tr><td rowspan="2">政策类 LLMs</td><td rowspan="2">方法</td><td rowspan="2">最大动作数</td><td colspan="2">成功率 (SR)</td><td rowspan="2">$\Delta$</td></tr><tr><td>无动作选择</td><td>含动作选择</td></tr><tr><td></td><td rowspan="4">AutoEval (Pan et al. 2024) <br/> BrowserGym (Drouin et al., 2024) <br/> SteP (Sodhi et al. 2023 <br/> AWM wang et al. 2024b)</td><td rowspan="4">30</td><td>20.2%</td><td>-</td><td>-</td></tr><tr><td>GPT-4</td><td>23.5%</td><td>-</td><td>-</td></tr><tr><td></td><td>35.8%</td><td>-</td><td>-</td></tr><tr><td></td><td>35.5%</td><td>-</td><td>-</td></tr><tr><td rowspan="3">GPT-4o</td><td>普通链式思路 (Vanilla CoT) (Zhou et al. 2023)</td><td>30</td><td>13.1%</td><td>-</td><td>-</td></tr><tr><td>树搜索代理 (Koh et al., 2024)</td><td>5</td><td>15.0%</td><td>19.2%</td><td>+28.0%</td></tr><tr><td>WMA 网页代理（本工作）</td><td>5</td><td>12.8%</td><td>16.6%</td><td>+29.7%</td></tr><tr><td>GPT-4o-mini</td><td>WMA 网页代理（本工作）</td><td>5</td><td>9.4%</td><td>13.5%</td><td>+43.6%</td></tr></tbody></table>


Table 2: Domain-specific performance of agents using GPT-40-mini as policy models
表 2：使用 GPT-40-mini 作为策略模型的代理在特定领域的性能


<table><tr><td>Methods / Domains</td><td>Shopping</td><td>CMS</td><td>Reddit</td><td>Gitlab</td><td>Map</td><td>Overall</td></tr><tr><td>Vanilla CoT (max actions = 5)</td><td>18.8%</td><td>8.2%</td><td>5.3%</td><td>3.1%</td><td>11.6%</td><td>9.4%</td></tr><tr><td>WMA web agent (ours)</td><td>19.3%</td><td>11.5%</td><td>7.9%</td><td>8.7%</td><td>22.3%</td><td>13.5%</td></tr><tr><td>$\Delta$</td><td>+3%</td><td>+40%</td><td>+49%</td><td>+181%</td><td>+92%</td><td>+44%</td></tr></table>
<table><tbody><tr><td>方法 / 领域</td><td>购物</td><td>内容管理系统</td><td>Reddit</td><td>GitLab</td><td>地图</td><td>总体</td></tr><tr><td>原始 CoT（最大操作数 = 5）</td><td>18.8%</td><td>8.2%</td><td>5.3%</td><td>3.1%</td><td>11.6%</td><td>9.4%</td></tr><tr><td>WMA 网络代理（我们的）</td><td>19.3%</td><td>11.5%</td><td>7.9%</td><td>8.7%</td><td>22.3%</td><td>13.5%</td></tr><tr><td>$\Delta$</td><td>+3%</td><td>+40%</td><td>+49%</td><td>+181%</td><td>+92%</td><td>+44%</td></tr></tbody></table>


Training data for world models. (i) For evaluation in WebArena: To facilitate applications in the real world, the training data for world models needs to cover a wide range of tasks/goals. Since a diverse and large-scale user instructions set is not available ${}^{1}$ we synthesize user instructions using an LLM. With these synthesized instructions of various goals, we are able to collect rich trajectories as training data, improving world models' generalization to diverse real-world situations. In practice, we sample $l$ trajectories for each $I\left\lbrack  \begin{array}{l} 2 \\   \end{array}\right\rbrack$ We generate 870 synthetic user instructions and gather 14K instances from WebArena using GPT-40-mini as the policy model. To avoid redundant learning, we filter out repeated state-action pairs. (ii) For evaluation in Mind2Web: we adopt the offline trajectory data from Mind2Web, following the setting of Wang et al. (2024b).
用于世界模型的训练数据。(i) 用于 WebArena 的评估：为了便于现实应用，世界模型的训练数据需涵盖广泛的任务/目标。由于没有多样且大规模的用户指令集可用${}^{1}$，我们使用大型语言模型合成用户指令。通过这些不同目标的合成指令，我们能够收集丰富的轨迹作为训练数据，提升世界模型对各种现实场景的泛化能力。实际操作中，我们为每个$I\left\lbrack  \begin{array}{l} 2 \\   \end{array}\right\rbrack$采样$l$条轨迹。我们生成了 870 条合成用户指令，并使用 GPT-40-mini 作为策略模型从 WebArena 收集了 14K 条实例。为避免冗余学习，我们过滤掉重复的状态-动作对。(ii) 用于 Mind2Web 的评估：我们采用 Mind2Web 的离线轨迹数据，遵循 Wang et al. (2024b) 的设置。


Baselines. For baselines, we adopt: (1) a prompting-based LLM (Zhou et al. 2023) powered by chain-of-thought prompting (Wei et al. 2022); (2) AutoEval (Pan et al. 2024). It refines agents' trajectories based on the feedback on the final state of the trajectory (i.e., succeed or fail) from a VLM evaluator (Shinn et al. 2024); (3) BrowserGym (Drouin et al. 2024) trains web agents with multi-modal observations, including HTML contents and the screenshot image of the browser; (4) SteP (Sodhi et al. 2023), a framework based on human-authored hierarchical policies injected to the agent; (5) HTML-T5 (Gur et al. 2023), the previous SOTA method on Mind2Web, uses LLMs pre-trained LLMs on HTML corpus. (6) Agent workflow memory (AWM Wang et al. (2024b)) leverages self-discovered workflow memory to guides its policy; (7) Tree search agent (Koh et al. 2024), the most competitive baseline that explores multiple trajectories and selects an optimal path via a tree search algorithm. The main difference between ours and Tree search agents is that ours only uses the predicted future states via simulation and does not actually explore diverse states.
基线方法。作为基线，我们采用：(1) 基于提示的大型语言模型 (Zhou et al. 2023)，使用链式思维提示 (Wei et al. 2022)；(2) AutoEval (Pan et al. 2024)，它基于来自视觉语言模型评估器 (Shinn et al. 2024) 关于轨迹最终状态（即成功或失败）的反馈来优化代理的轨迹；(3) BrowserGym (Drouin et al. 2024)，使用多模态观测训练网页代理，包括 HTML 内容和浏览器截图图像；(4) SteP (Sodhi et al. 2023)，一个将人工编写的分层策略注入代理的框架；(5) HTML-T5 (Gur et al. 2023)，Mind2Web 上的先前 SOTA 方法，使用在 HTML 语料上预训练的 LLM；(6) Agent workflow memory (AWM，Wang et al. (2024b))，利用自发现的工作流记忆指导其策略；(7) Tree search agent (Koh et al. 2024)，最具竞争力的基线，通过探索多条轨迹并通过树搜索算法选择最优路径。我们的方法与树搜索代理的主要区别在于，我们仅通过模拟使用预测的未来状态，而不实际探索多样的状态。


World model. We use Llama-3.1-8B-Instruct (Dubey et al. 2024) as the backbone LLM for our world models ${}^{3}$ For WebArena,we construct our dataset in online setting using the provided web environment. In Mind2Web, we use the offline trajectory data (i.e., the train set) following Wang et al. (2024b). For prompt-based world models (baselines) in our experiments, we use 2-shot demonstrations to instruct LLMs to predict the next state. More details are provided in Appendix C. 1
世界模型。我们使用 Llama-3.1-8B-Instruct (Dubey et al. 2024) 作为世界模型的骨干 LLM ${}^{3}$。在 WebArena 中，我们在提供的网页环境下以在线方式构建数据集。在 Mind2Web 中，我们使用离线轨迹数据（即训练集），遵循 Wang et al. (2024b)。对于实验中的基于提示的世界模型（基线），我们使用 2-shot 示范来指示 LLM 预测下一个状态。更多细节见附录 C。


Policy model. Following Koh et al. (2024), we adopt GPT-40 (gpt-40-0513) as the agent backbone for evaluation in WebArena. Additionally, we test with GPT-40-mini (gpt-40-mini-0718) to test our framework in relatively more resource-restricted scenarios.
策略模型。按照 Koh et al. (2024)，我们在 WebArena 的评估中采用 GPT-40 (gpt-40-0513) 作为代理骨干。此外，我们还使用 GPT-40-mini (gpt-40-mini-0718) 在计算资源相对受限的场景下测试我们的框架。


---



${}^{1}$ In WebArena,only test data (i.e.,instructions) is provided.
${}^{1}$ 在 WebArena 中，只提供测试数据（即指令）。


${}^{2}$ We empirically set $l = 5$ in our work. Further details on the whole data collection process are in Appendix. shttps://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct
${}^{2}$ 我们在工作中经验性地设置了$l = 5$。关于整个数据收集过程的更多细节见附录。shttps://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct


---



Table 3: Success rate on Mind2Web tests using GPT-3.5-Turbo as policy models. EA = element accuracy; $\mathrm{{EF}} =$ element filtering; ${\mathrm{{AF}}}_{1} =$ action ${\mathrm{F}}_{1}; *  =$ results from the original paper.
表 3：在 Mind2Web 测试中使用 GPT-3.5-Turbo 作为策略模型的成功率。EA = 元素准确率；$\mathrm{{EF}} =$ 元素过滤；${\mathrm{{AF}}}_{1} =$ 动作；${\mathrm{F}}_{1}; *  =$ 来源于原论文的结果。


<table><tr><td rowspan="2">Methods</td><td colspan="4">Cross-Task</td><td colspan="4">Cross-Website</td><td colspan="4">Cross-Domain</td></tr><tr><td>EA</td><td>${\mathrm{{AF}}}_{1}$</td><td>Step SR</td><td>SR</td><td>EA</td><td>${\mathrm{{AF}}}_{1}$</td><td>Step SR</td><td>SR</td><td>EA</td><td>${\mathrm{{AF}}}_{1}$</td><td>Step SR</td><td>SR</td></tr><tr><td>Synapse*</td><td>34.4%</td><td>-</td><td>30.6%</td><td>2.0%</td><td>28.8%</td><td>-</td><td>23.4%</td><td>1.1%</td><td>29.4%</td><td>-</td><td>25.9%</td><td>1.6%</td></tr><tr><td>HTML-T5-XL*</td><td>60.6%</td><td>81.7%</td><td>57.8%</td><td>10.3%</td><td>47.6%</td><td>71.9%</td><td>42.9%</td><td>5.6%</td><td>50.2%</td><td>74.9%</td><td>48.3%</td><td>5.1%</td></tr><tr><td>MindAct*</td><td>41.6%</td><td>60.6%</td><td>36.2%</td><td>2.0%</td><td>35.8%</td><td>51.1%</td><td>30.1%</td><td>2.0%</td><td>21.6%</td><td>52.8%</td><td>18.6%</td><td>1.0%</td></tr><tr><td>AWM (w/ EF)*</td><td>50.6%</td><td>57.3%</td><td>45.1%</td><td>4.8%</td><td>41.4%</td><td>46.2%</td><td>33.7%</td><td>2.3%</td><td>36.4%</td><td>41.6%</td><td>32.6%</td><td>0.7%</td></tr><tr><td>AWM (w/o EF)</td><td>78.3%</td><td>74.1%</td><td>62.8%</td><td>15.3%</td><td>74.7%</td><td>70.1%</td><td>58.6%</td><td>6.2%</td><td>74.8%</td><td>71.2%</td><td>60.7%</td><td>9.5%</td></tr><tr><td>AWM+WMA (ours)</td><td>79.9%</td><td>75.8%</td><td>67.0%</td><td>25.4%</td><td>75.7%</td><td>72.1%</td><td>61.3%</td><td>8.5%</td><td>75.9%</td><td>72.6%</td><td>63.4%</td><td>10.1%</td></tr></table>
<table><tbody><tr><td rowspan="2">方法</td><td colspan="4">跨任务</td><td colspan="4">跨网站</td><td colspan="4">跨领域</td></tr><tr><td>EA</td><td>${\mathrm{{AF}}}_{1}$</td><td>分步 SR</td><td>SR</td><td>EA</td><td>${\mathrm{{AF}}}_{1}$</td><td>分步 SR</td><td>SR</td><td>EA</td><td>${\mathrm{{AF}}}_{1}$</td><td>分步 SR</td><td>SR</td></tr><tr><td>Synapse*</td><td>34.4%</td><td>-</td><td>30.6%</td><td>2.0%</td><td>28.8%</td><td>-</td><td>23.4%</td><td>1.1%</td><td>29.4%</td><td>-</td><td>25.9%</td><td>1.6%</td></tr><tr><td>HTML-T5-XL*</td><td>60.6%</td><td>81.7%</td><td>57.8%</td><td>10.3%</td><td>47.6%</td><td>71.9%</td><td>42.9%</td><td>5.6%</td><td>50.2%</td><td>74.9%</td><td>48.3%</td><td>5.1%</td></tr><tr><td>MindAct*</td><td>41.6%</td><td>60.6%</td><td>36.2%</td><td>2.0%</td><td>35.8%</td><td>51.1%</td><td>30.1%</td><td>2.0%</td><td>21.6%</td><td>52.8%</td><td>18.6%</td><td>1.0%</td></tr><tr><td>AWM (含 EF)*</td><td>50.6%</td><td>57.3%</td><td>45.1%</td><td>4.8%</td><td>41.4%</td><td>46.2%</td><td>33.7%</td><td>2.3%</td><td>36.4%</td><td>41.6%</td><td>32.6%</td><td>0.7%</td></tr><tr><td>AWM (不含 EF)</td><td>78.3%</td><td>74.1%</td><td>62.8%</td><td>15.3%</td><td>74.7%</td><td>70.1%</td><td>58.6%</td><td>6.2%</td><td>74.8%</td><td>71.2%</td><td>60.7%</td><td>9.5%</td></tr><tr><td>AWM+WMA（本方法）</td><td>79.9%</td><td>75.8%</td><td>67.0%</td><td>25.4%</td><td>75.7%</td><td>72.1%</td><td>61.3%</td><td>8.5%</td><td>75.9%</td><td>72.6%</td><td>63.4%</td><td>10.1%</td></tr></tbody></table>


Value function. We fine-tune Llama-3.1-8B-Instruct to predict rewards using data from Mind2Web, where rewards (as training objective) are calculated based on the progress toward the goal,i.e., $t/\left( {\operatorname{len}\left( \tau \right) }\right)$ when ${a}_{t}$ is taken.
值函数。我们对 Llama-3.1-8B-Instruct 进行微调以使用 Mind2Web 的数据预测奖励，其中奖励（作为训练目标）是基于朝目标的进展计算的，即在采取 ${a}_{t}$ 时为 $t/\left( {\operatorname{len}\left( \tau \right) }\right)$。


### 5.2 MAIN RESULTS
### 5.2 主要结果


Agent performance in WebArena. In Table 1 (middle), we first compare our WMA web agent (16.6%) with vanilla CoT (13.1%) and observe significant improvements over almost all domains in WebArena as detailed in Table 2 Interestingly, when using GPT-40-mini as the policy model, our agent achieve 181% and 92% performance gains over CoT in Gitlab and Map, respectively. The relatively small improvement in Shopping might be due to the large-scale state space $\mathcal{S}$ in the domain, such as the diversity of searched item lists from different user queries, which makes it harder for the world model to properly learn environment dynamics. Regardless, the overall improvement suggests the effectiveness of leveraging learnt environment dynamics during inference time.
WebArena 中的代理表现。在表 1（中部），我们首先将我们的 WMA 网络代理（16.6%）与原始 CoT（13.1%）进行比较，观察到在 WebArena 几乎所有领域都有显著提升，详见表 2。有趣的是，当使用 GPT-40-mini 作为策略模型时，我们的代理在 Gitlab 和 Map 上分别比 CoT 实现了 181% 和 92% 的性能提升。在 Shopping 领域提升相对较小可能是由于该领域的大规模状态空间 $\mathcal{S}$，例如来自不同用户查询的搜索商品列表的多样性，这使得世界模型更难恰当地学习环境动态。无论如何，整体改进表明在推理时利用学习到的环境动态是有效的。


Next, we compare our approach with Tree search agent (Koh et al. 2024), which uses the oracle next state observation (i.e.,resulted by the gold transition function $\mathcal{T}$ from the environment) for policy selection instead of estimated observation via the world model. While the absolute SR of our WMA agent (16.6%) is slightly below Tree search agent (19.2%) when using GPT-40 as policy models, our policy optimization with the world model brings a larger performance gain to vanilla CoT than tree search (+29.7% vs. +28.0%). Also, in the later section (§5.3), we present ours' superior cost and time efficiency over Tree search agent.
接着，我们将我们的方法与 Tree search agent（Koh et al. 2024）比较，后者在策略选择时使用来自环境的 Oracle 下一态观察（即由金标准转移函数 $\mathcal{T}$ 产生的结果），而不是通过世界模型估计的观察。尽管当使用 GPT-40 作为策略模型时，我们的 WMA 代理的绝对成功率（16.6%）略低于 Tree search agent（19.2%），但我们通过世界模型进行的策略优化相比于原始 CoT 带来了更大的性能提升（+29.7% vs. +28.0%）。另外，在后文（§5.3）中，我们展示了我们在成本和时间效率上相对于 Tree search agent 的优势。


Agent performance in Mind2Web. We compare WMA web agent with MindAct (Deng et al. 2024) and AWM (Wang et al. 2024b), which are previous and current SQTAs on Mind2Web ${}^{4}$ Table 3 demonstrates that WMA web agent significantly outperforms AWM ${}^{5}$ achieving new SOTA performance. Furthermore, the results indicate that WMA web agent trained on Mind2Web data has a strong generalization capability. This makes our approach much more valuable in scenarios where collecting data for new web environments is non-trivial.
Mind2Web 中的代理表现。我们将 WMA 网络代理与 MindAct（Deng et al. 2024）和 AWM（Wang et al. 2024b）比较，二者分别为 Mind2Web 上的前任和现任 SOTA ${}^{4}$。表 3 显示 WMA 网络代理显著优于 AWM ${}^{5}$，达成了新的 SOTA 性能。此外，结果表明在 Mind2Web 数据上训练的 WMA 网络代理具有较强的泛化能力。这使得我们的方法在为新网页环境收集数据不易的场景中更具价值。


Our advantages besides performance gains. Based on the performance reported, we can conclude that our strategy of building world models (i.e., observation abstraction) is effective not only for accessibility tree format (WebArena) but also for HTML format (Mind2Web), underscoring the applicability of our approach across different representations of web data. Another advantage of our approach over others is that the developed world models can be incorporated into existing or future web agents without any additional training of policy models, enabling easy implementation.
除了性能提升外的优势。基于报告的性能，我们可以得出结论：构建世界模型（即观察抽象化）的策略不仅对无障碍树格式（WebArena）有效，对 HTML 格式（Mind2Web）同样有效，凸显了我们方法在不同网页数据表示间的适用性。我们方法的另一个优势是已开发的世界模型可以被纳入现有或未来的网页代理中，而无需对策略模型进行任何额外训练，从而便于实现。


### 5.3 ANALYSES OF TIME AND COST EFFICIENCY
### 5.3 时间和成本效率分析


We compare our WMA web agent with Tree search agent in terms of time and API cost efficiency. Results are shown in Table 4 To run one user instruction, Tree search agent spends about 748.3 seconds on average, as it involves the exploration of diverse future states while actually interacting with the environment. When it conducts backtracing to revert to the previous state, the whole sequence of previous actions has to be executed again. By contrast, WMA web agent only takes 140.3 seconds per instance by simulating the possible action candidates rather than actually executing them, which
我们在时间与 API 成本效率方面将 WMA 网络代理与 Tree search agent 进行比较。结果见表 4。处理一条用户指令时，Tree search agent 平均耗时约 748.3 秒，因为它涉及对多样的未来状态进行探索并实际与环境交互。当其回溯以恢复到先前状态时，之前整个动作序列必须再次被执行。相比之下，WMA 网络代理每个实例仅需 140.3 秒，因为它通过模拟可能的动作候选而不是实际执行这些动作，进而


---



${}^{4}$ Tree search agent is not applicable to this benchmark as the environment is not available.
${}^{4}$ 由于环境不可用，Tree search agent 不适用于该基准。


${}^{5}$ Surprisingly, we find element filtering (EF) of MindAct, applied to AWM in default, largely hindering its performance. Thus, in Table 3 we include the results without EF. A detailed discussion is in Appendix C. 6
${}^{5}$ 有趣的是，我们发现 MindAct 的元素过滤（EF），默认应用于 AWM，很大程度上阻碍了其性能。因此，在表 3 中我们包含了无 EF 的结果。详细讨论见附录 C。6


---



Table 4: Head-to-head comparison of Tree search agent (results are from Koh et al. (2024)) and ours regarding (i) SR and (ii) API cost, and (iii) inference time. We use GPT-40 for policy models.
表 4：Tree search agent（结果取自 Koh et al. (2024)）与我们在 (i) 成功率，(ii) API 成本，和 (iii) 推理时间方面的正面对比。我们使用 GPT-40 作为策略模型。


<table><tr><td>Methods</td><td>Shopping</td><td>CMS</td><td>Reddit</td><td>Gitlab</td><td>Map</td><td>API cost</td><td>Inference time (sec)</td></tr><tr><td>Tree search agent</td><td>28.1%</td><td>16.5%</td><td>10.5%</td><td>13.3%</td><td>25.8%</td><td>\$2.7</td><td>748.3</td></tr><tr><td>WMA (ours)</td><td>20.8%</td><td>14.3%</td><td>10.5%</td><td>13.3%</td><td>26.8%</td><td>\$0.4</td><td>140.3</td></tr></table>
<table><tbody><tr><td>方法</td><td>购物</td><td>内容管理系统</td><td>Reddit</td><td>Gitlab</td><td>地图</td><td>API 成本</td><td>推理时间（秒）</td></tr><tr><td>树搜索智能体</td><td>28.1%</td><td>16.5%</td><td>10.5%</td><td>13.3%</td><td>25.8%</td><td>\$2.7</td><td>748.3</td></tr><tr><td>WMA（本方法）</td><td>20.8%</td><td>14.3%</td><td>10.5%</td><td>13.3%</td><td>26.8%</td><td>\$0.4</td><td>140.3</td></tr></tbody></table>


Table 5: Results of the ablation study in WebArena.
表 5：WebArena 中消融研究的结果。


<table><tr><td rowspan="2">Settings</td><td colspan="2">World Model</td><td colspan="4">Success Rate (SR)</td></tr><tr><td>Use</td><td>Training</td><td>Shopping</td><td>Gitlab</td><td>Map</td><td>Overall</td></tr><tr><td>w/o next states in reward estimation (§4.2)</td><td>✘</td><td>✘</td><td>28.0%</td><td>6.0%</td><td>19.0%</td><td>18.0%</td></tr><tr><td>w/o training world models (§4.1)</td><td>✓</td><td>✘</td><td>30.0%</td><td>10.0%</td><td>15.0%</td><td>17.5%</td></tr><tr><td>w/o abstracting observations (§4.1.2)</td><td>✓</td><td>✓</td><td>22.0%</td><td>6.0%</td><td>15.0%</td><td>14.5%</td></tr><tr><td>WMA (ours)</td><td>✓</td><td>✓</td><td>32.0%</td><td>14.0%</td><td>21.0%</td><td>22.0%</td></tr></table>
<table><tbody><tr><td rowspan="2">设置</td><td colspan="2">世界模型</td><td colspan="4">成功率 (SR)</td></tr><tr><td>使用</td><td>训练</td><td>购物</td><td>Gitlab</td><td>地图</td><td>总体</td></tr><tr><td>在奖励估计中不使用下一个状态（§4.2）</td><td>✘</td><td>✘</td><td>28.0%</td><td>6.0%</td><td>19.0%</td><td>18.0%</td></tr><tr><td>不训练世界模型（§4.1）</td><td>✓</td><td>✘</td><td>30.0%</td><td>10.0%</td><td>15.0%</td><td>17.5%</td></tr><tr><td>不抽象观测（§4.1.2）</td><td>✓</td><td>✓</td><td>22.0%</td><td>6.0%</td><td>15.0%</td><td>14.5%</td></tr><tr><td>WMA（本文）</td><td>✓</td><td>✓</td><td>32.0%</td><td>14.0%</td><td>21.0%</td><td>22.0%</td></tr></tbody></table>


is 5.3 times faster than Tree search agent. Tree search agent requires 6.8 times more API cost due to its multi-modal inputs. To sum up, while showing comparable performance to Tree search agent in CMS, Reddit, Gitlab, and Map, our WMA web agent demonstrates superior cost and time efficiency.
比 Tree search agent 快 5.3 倍。由于其多模态输入，Tree search agent 需要 6.8 倍的 API 成本。总之，尽管在 CMS、Reddit、Gitlab 和 Map 上表现与 Tree search agent 相当，我们的 WMA web agent 在成本和时间效率上具有优势。


### 5.4 ABLATION STUDIES
### 5.4 消融研究


We conduct several ablation studies on our WMA web agent with 200 randomly sampled instances from WebArena (Shopping: 50; Gitlab: 50; Map: 100). We use GPT-40-mini as policy models.
我们在 WebArena 上随机抽取 200 个实例（Shopping: 50；Gitlab: 50；Map: 100）对 WMA web agent 进行多项消融研究。我们使用 GPT-40-mini 作为策略模型。


Accessing simulated next states in reward estimation improves agent performance. To assess the impact of incorporating the simulated next state when calculating the value score, we compare our reward estimation strategy to a Q-value function (Haarnoja et al. 2017) that predicts the reward based on only $\left( {{o}_{t},{a}_{t}}\right)$ . The results in Table 5 show that the information of the resulting next state helps the value function to predict rewards more accurately, resulting a better task performance.
在奖励估计中访问模拟的下一个状态能提升代理性能。为评估在计算价值分数时纳入模拟下一个状态的影响，我们将该奖励估计策略与仅基于 $\left( {{o}_{t},{a}_{t}}\right)$ 预测奖励的 Q 值函数（Haarnoja et al. 2017）进行比较。表 5 的结果表明，结果状态的信息有助于价值函数更准确地预测奖励，从而带来更好的任务表现。


Fine-tuning facilitates better world models than prompt-based approaches. To assess the effectiveness of our training approach for world models, we compare our framework with a variant, where we replace the trained world model (i.e., fine-tuned Llama-3.1-8B-Instruct) with a GPT-4o-mini prompted to predict the next observation solely based on 2-shot demonstrations (i.e., in-context learning) without training. The sub-optimal performance of this variant, as shown in Table 5 (2nd row), suggests that SOTA LLMs do not have sufficient knowledge of environment dynamics, which is consistent with our findings in §3.1
微调比基于提示的方法能构建更好的世界模型。为评估我们世界模型训练方法的有效性，我们将框架与一个变体比较：用一个通过 2-shot 演示（即上下文学习）提示但未训练的 GPT-4o-mini 代替训练好的世界模型（即微调的 Llama-3.1-8B-Instruct）来仅基于示例预测下一个观测。表 5（第 2 行）所示的该变体的次优表现表明，最先进的 LLM 并不具备足够的环境动力学知识，这与我们在 §3.1 的发现一致。


Table 6: Performance with different value models.
表 6：使用不同价值模型的表现。


<table><tr><td>Value Function</td><td>Training</td><td>SR</td></tr><tr><td>GPT-4o-mini</td><td>✘</td><td>12.7%</td></tr><tr><td>Llama-3.1-8B</td><td>✓</td><td>13.5%</td></tr></table>
<table><tbody><tr><td>值函数</td><td>训练</td><td>SR</td></tr><tr><td>GPT-4o-mini</td><td>✘</td><td>12.7%</td></tr><tr><td>Llama-3.1-8B</td><td>✓</td><td>13.5%</td></tr></tbody></table>


Abstracting observation elicits better next state prediction. We evaluate the effectiveness of our observation abstraction (§4.1.2), which focuses on state transition. For that, we train a world model that learns to predict the full accessibility tree, i.e., ${o}_{t + 1}$ instead of our transition-focused abstraction ${\widetilde{o}}_{t + 1}$ . As we expected, Table 5 (3rd row) reveals that generating the whole next observations (i.e., all elements in the viewport) results indeed hinder agent performance, yielding the worst SR among all ablations. This shows that processing redundant and repeated information across observations negatively affects the world model in capturing critical state changes compared to abstracted observations that exclusively highlight state transition.
抽象化观测可提升对下一状态的预测。我们评估了侧重于状态转移的观测抽象 (§4.1.2) 的有效性。为此，我们训练了一个世界模型来预测完整的可访问性树，即 ${o}_{t + 1}$，而非我们侧重转移的抽象 ${\widetilde{o}}_{t + 1}$。如预期，表5（第3行）显示生成完整的下一步观测（即视口内的所有元素）确实会削弱代理性能，导致所有消融中最低的成功率（SR）。这表明相比仅突出状态转移的抽象化观测，处理观测中冗余和重复信息会妨碍世界模型捕捉关键状态变化。


<img src="https://cdn.noedgeai.com/bo_d4nfperef24c73bbffn0_8.jpg?x=1104&y=1634&w=367&h=201&r=0"/>



Figure 6: Ablation on the number of sampled actions $\left( k\right)$ .
图6：关于采样动作数量的消融 $\left( k\right)$。


Choice of value functions. We compare the fine-tuned value model (i.e., Llama-3.1-8B-Instruct) used for implementing WMA web agents with prompted GPT-40-mini in Table 6 Ours lead to a slightly better agent performance compared to GPT-40-mini. This suggests fine-tuning the value function is a reasonable alternative in scenarios where API budgets are limited.
价值函数的选择。我们在表6中将用于实现 WMA 网页代理的微调价值模型（即 Llama-3.1-8B-Instruct）与通过提示使用的 GPT-40-mini 进行了比较。我们的模型相比 GPT-40-mini 带来了略优的代理表现。这表明在 API 预算受限的情形下微调价值函数是一个合理的替代方案。


Budget for exploration. Figure 6 shows that there is a positive trend between the number of sampled actions $\left( k\right)$ during inference-time policy optimization in §4.2 and the agents’ task performance (SR). These results suggest that our WMA web agent may benefit from more exploration of the future states when the budget is allowed.
探索预算。图6 显示在 §4.2 中推理时策略优化期间采样动作数量 $\left( k\right)$ 与代理任务表现（SR）之间存在正相关趋势。这些结果表明，当预算允许时，我们的 WMA 网页代理可能从对未来状态的更多探索中获益。


## 6 FURTHER ANALYSES
## 6 进一步分析


### 6.1 COMBINING SELF-REFINE WITH OUR WORLD MODELS
### 6.1 将自我精炼与我们的世界模型结合


Besides our inference-time policy optimization, another way of using world models is to refine its predicted action (Madaan et al. 2024), based on the outcome simulated by the world model. Such self-refinement has been showing promising performance in diverse LLM applications (Shinn et al. 2024, Chae et al., 2024). Here, we conduct a demonstrative experiment of combining self-refine with our world model in the Map domain from We-bArena. Since tasks in this domain involve a complex set of utility tools, such as sorting and zoom-in, we consider it suitable for testing self-refine. In this experiment,after the policy model $\theta$ produces an action ${a}_{t}$ ,we use our world model to simulate the next observation ${\widetilde{o}}_{t + 1}$ and prompt $\theta$ to refine the action based on ${\widetilde{o}}_{t}$ . Simply put,this setting allows $\theta$ to make adjustments to its output action when the predicted next observation is not optimal. Table 7 shows that refining with simulated environment feedback improves the agent's policy by 1.8% point in terms of accuracy compared to CoT. While this is a plausible direction for future work, our simulate-score-select paradigm yields an almost $2\mathrm{x}$ higher accuracy,making it our choice of the policy optimization method.
除了我们的推理时策略优化之外，另一种使用世界模型的方法是基于世界模型模拟的结果来精炼其预测的动作（Madaan 等，2024）。此类自我精炼在多种大模型应用中展现出良好效果（Shinn 等，2024；Chae 等，2024）。在此，我们在 We-bArena 的地图领域做了一个示范性实验，将自我精炼与我们的世界模型结合。由于该领域的任务涉及一套复杂的实用工具，如排序和放大，我们认为其适合测试自我精炼。在该实验中，当策略模型 $\theta$ 生成一个动作 ${a}_{t}$ 后，我们使用世界模型模拟下一步观测 ${\widetilde{o}}_{t + 1}$，并基于 ${\widetilde{o}}_{t}$ 提示 $\theta$ 来精炼动作。简而言之，这一设置允许 $\theta$ 在预测的下一步观测不理想时对其输出动作进行调整。表7 显示，与 CoT 相比，利用模拟的环境反馈进行精炼使代理策略在准确率上提升了 1.8 个百分点。尽管这是未来工作的可行方向，我们的 simulate-score-select 范式带来了几乎 $2\mathrm{x}$ 的更高准确率，因此成为我们选择的策略优化方法。


Table 7: Results of applying self-refine to GPT-40-mini using simulated environment feedback.
表7：将自我精炼应用于使用模拟环境反馈的 GPT-40-mini 的结果。


<table><tr><td>Methods</td><td>SR</td></tr><tr><td>Vanilla CoT</td><td>11.6%</td></tr><tr><td>Self-refine w/ our world model</td><td>13.4%</td></tr><tr><td>WMA (ours; Fig 3)</td><td>22.3%</td></tr></table>
<table><tbody><tr><td>方法</td><td>SR</td></tr><tr><td>普通 CoT</td><td>11.6%</td></tr><tr><td>使用我们的世界模型自我改进</td><td>13.4%</td></tr><tr><td>WMA（本文方法；图 3）</td><td>22.3%</td></tr></tbody></table>


### 6.2 TYPES OF ERRORS IN WORLD MODELS' PREDICTIONS
### 6.2 世界模型预测中的错误类型


To gain deeper insights into WMA web agents,we sample 50 erroneous predicted states (i.e., ${\widetilde{o}}_{t + 1}$ ) from world models in WebArena, and manually categorize the type of errors. Whether a predicted state is erroneous is judged by a CS major who manually compares the viewport and the predicted observation. Examples of each type and details on the sampled states are provided in Appendix D.2 Figure 7 shows the statistics of the following error types: (i) Correct yet overly generic statements (24%) - Statements such as "The user will see a comprehensive layout of various order-related functionalities", where the structure of the layout and what functionalities will be seen are not specified; (ii) Low competence in web elements/functions (26%) - Cases where the world model does not know how to use components on the web, e.g., expecting the search engine to show the desired items when the agent does not delete old texts on the search bar before entering a new keyword; (iii) Counterfactual imagination (42%) - Cases where the next observation predicted by the world model includes elements that are not supposed to occur/exist, e.g., making up products that are not sold in the store; (iv) others (8%) - other errors, such as skipping the next observation and predicting an observation that is further from the current time step.
为了深入了解 WMA 网络代理，我们从 WebArena 的世界模型中抽样 50 个错误预测状态（即 ${\widetilde{o}}_{t + 1}$），并对错误类型进行人工分类。是否为错误预测由一位计算机专业人员通过人工比较视口和预测观察结果来判断。每类示例和抽样状态的详细信息见附录 D.2。图 7 显示了以下错误类型的统计：（i）正确但过于泛泛的陈述（24%）——例如“用户将看到各种与订单相关功能的综合布局”，但未具体说明布局结构和可见的功能；（ii）对网页元素/功能能力不足（26%）——世界模型不知道如何使用网页组件，例如在代理未删除搜索栏中的旧文本就输入新关键词时，仍指望搜索引擎显示期望项目；（iii）反事实想象（42%）——世界模型预测的下一观察包含不应出现/不存在的元素，例如杜撰商店未出售的产品；（iv）其他（8%）——其他错误，如跳过下一观察而预测与当前时间步更远的观察。


<img src="https://cdn.noedgeai.com/bo_d4nfperef24c73bbffn0_9.jpg?x=993&y=1339&w=493&h=166&r=0"/>



Figure 7: Statistics of error types in erroneous observations predicted by $\phi$ .
图 7：由 $\phi$ 预测的错误观察中错误类型的统计。


## 7 CONCLUSIONS
## 7 结论


We are the first study that incorporates world models in LLM-based web agents, addressing the limitation of current SOTA LLMs in understanding environment dynamics. Through extensive experiments in WebArena and Mind2Web, we show that (i) our WMA web agent can demonstrate great efficacy in policy selection by simulating outcomes of its actions via world models trained using our approach (i.e., transition-focused observation abstraction). Moreover, (ii) our WMA web agent outperforms strong baselines (i.e., Tree search agent) with reduced cost and time for the exploration and (iii) achieves a new SOTA performance in Mind2Web. By augmenting LLM-based web agents with world models, we establish a strong foundation for future research in web navigation.
我们是首个将世界模型纳入基于大模型的网络代理的研究，解决了现有最先进大模型在理解环境动态方面的局限。通过在 WebArena 和 Mind2Web 上的大量实验证明：（i）我们的 WMA 网络代理通过使用我们的方法（即以转移为中心的观察抽象）训练的世界模型模拟其动作结果，在策略选择上表现出很高的效能；此外，（ii）我们的 WMA 网络代理在探索成本和时间上优于强基线（即树搜索代理）；（iii）并在 Mind2Web 上达到新的最先进性能。通过用世界模型增强基于大模型的网络代理，我们为未来在网页导航方面的研究奠定了坚实基础。


## ACKNOWLEDGMENTS
## 致谢


This work was supported by STEAM R&D Project, NRF, Korea (RS-2024-00454458) and Institute of Information & communications Technology Planning & Evaluation (IITP) grant funded by the Korea government(MSIT) (No. RS-2024-00457882, National AI Research Lab Project). Jinyoung Yeo is the corresponding author.
本工作由韩国 STEAM R&D 项目、NRF（RS-2024-00454458）及韩国政府（MSIT）资助的정보통신기획평가院(IITP) 资助（编号 RS-2024-00457882，国家人工智能研究实验室项目）支持。Jinyoung Yeo 为通讯作者。


## REFERENCES
## 参考文献


Anthropic. Introducing claude 3.5 sonnet, June 21 2024. URL https://www.anthropic.com/news/claude-3-5-sonnet Accessed: 2024-09-30.
Anthropic. Introducing claude 3.5 sonnet, 2024 年 6 月 21 日。URL https://www.anthropic.com/news/claude-3-5-sonnet 访问日期：2024-09-30。


Hyungjoo Chae, Taeyoon Kwon, Seungjun Moon, Yongho Song, Dongjin Kang, Kai Tzu-iunn Ong, Beong-woo Kwak, Seonghyeon Bae, Seung-won Hwang, and Jinyoung Yeo. Coffee-gym: An environment for evaluating and improving natural language feedback on erroneous code. arXiv preprint arXiv:2409.19715, 2024.
Hyungjoo Chae, Taeyoon Kwon, Seungjun Moon, Yongho Song, Dongjin Kang, Kai Tzu-iunn Ong, Beong-woo Kwak, Seonghyeon Bae, Seung-won Hwang, 和 Jinyoung Yeo。Coffee-gym：用于评估和改进对错误代码的自然语言反馈的环境。arXiv 预印本 arXiv:2409.19715, 2024。


Silvia Chiappa, Sébastien Racaniere, Daan Wierstra, and Shakir Mohamed. Recurrent environment simulators. arXiv preprint arXiv:1704.02254, 2017.
Silvia Chiappa, Sébastien Racaniere, Daan Wierstra, 和 Shakir Mohamed。递归环境模拟器。arXiv 预印本 arXiv:1704.02254, 2017。


Xiang Deng, Yu Gu, Boyuan Zheng, Shijie Chen, Sam Stevens, Boshi Wang, Huan Sun, and Yu Su. Mind2web: Towards a generalist agent for the web. Advances in Neural Information Processing Systems, 36, 2024.
Xiang Deng, Yu Gu, Boyuan Zheng, Shijie Chen, Sam Stevens, Boshi Wang, Huan Sun, 和 Yu Su。Mind2web：迈向通用网页代理。Advances in Neural Information Processing Systems, 36, 2024。


Andreas Doerr, Christian Daniel, Martin Schiegg, Duy Nguyen-Tuong, Stefan Schaal, Marc Tou-ssaint, and Sebastian Trimpe. Probabilistic recurrent state-space models. In International Conference on Machine Learning, 2018. URL https://api.semanticscholar.org/ CorpusID: 45425492
Andreas Doerr, Christian Daniel, Martin Schiegg, Duy Nguyen-Tuong, Stefan Schaal, Marc Toussaint, 和 Sebastian Trimpe。概率递归状态空间模型。收录于国际机器学习大会，2018。URL https://api.semanticscholar.org/ CorpusID: 45425492


Alexandre Drouin, Maxime Gasse, Massimo Caccia, Issam H Laradji, Manuel Del Verme, Tom Marty, Léo Boisvert, Megh Thakkar, Quentin Cappart, David Vazquez, et al. Workarena: How capable are web agents at solving common knowledge work tasks? arXiv preprint arXiv:2403.07718, 2024.
Alexandre Drouin, Maxime Gasse, Massimo Caccia, Issam H Laradji, Manuel Del Verme, Tom Marty, Léo Boisvert, Megh Thakkar, Quentin Cappart, David Vazquez 等。Workarena：网络代理在解决常见知识工作任务方面的能力如何？arXiv 预印本 arXiv:2403.07718, 2024。


Yilun Du, Mengjiao Yang, Bo Dai, Hanjun Dai, Ofir Nachum, Joshua B. Tenenbaum, Dale Schu-urmans, and Pieter Abbeel. Learning universal policies via text-guided video generation, 2023. URLhttps://arxiv.org/abs/2302.00111
杜弈伦、杨梦娇、戴博、戴瀚君、奥菲尔·纳楚姆、约书亚·B·特南鲍姆、戴尔·舒尔曼斯和彼得·阿贝埃尔。通过文本引导的视频生成学习通用策略，2023 年。URLhttps://arxiv.org/abs/2302.00111


Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024.
阿比曼纽·杜贝、阿比纳夫·焦赫里、阿比纳夫·潘迪、阿比舍克·卡迪安、艾哈迈德·阿尔 - 达勒、艾莎·莱特曼、阿基尔·马图尔、艾伦·舍尔滕、艾米·杨、安吉拉·范等。大羊驼 3 模型群。预印本 arXiv:2407.21783，2024 年。


Ward Edwards. The theory of decision making. Psychological bulletin, 51(4):380, 1954.
沃德·爱德华兹。决策理论。《心理学公报》，51(4):380，1954 年。


Jay W Forrester. Counterintuitive behavior of social systems. The System Dynamics Road Maps, 1995.
杰伊·W·福雷斯特。社会系统的反直觉行为。《系统动力学路线图》，1995 年。


Izzeddin Gur, Hiroki Furuta, Austin Huang, Mustafa Safdari, Yutaka Matsuo, Douglas Eck, and Aleksandra Faust. A real-world webagent with planning, long context understanding, and program synthesis. arXiv preprint arXiv:2307.12856, 2023.
伊泽丁·古尔、古田博树、奥斯汀·黄、穆斯塔法·萨夫达里、松尾丰、道格拉斯·埃克和亚历山德拉·福斯特。具有规划、长上下文理解和程序合成能力的现实世界网络智能体。预印本 arXiv:2307.12856，2023 年。


David Ha and Jürgen Schmidhuber. World models. arXiv preprint arXiv:1803.10122, 2018.
大卫·哈和于尔根·施密德胡伯。世界模型。预印本 arXiv:1803.10122，2018 年。


Tuomas Haarnoja, Haoran Tang, P. Abbeel, and Sergey Levine. Reinforcement learning with deep energy-based policies. In International Conference on Machine Learning, 2017. URL https: //api.semanticscholar.org/CorpusID:11227891
图奥马斯·哈阿诺亚、唐浩然、P·阿贝埃尔和谢尔盖·列维。基于深度能量策略的强化学习。见《国际机器学习会议》，2017 年。URL https://api.semanticscholar.org/CorpusID:11227891


Danijar Hafner, Timothy Lillicrap, Jimmy Ba, and Mohammad Norouzi. Dream to control: Learning behaviors by latent imagination. arXiv preprint arXiv:1912.01603, 2019a.
达尼贾尔·哈夫纳、蒂莫西·利利克拉普、吉米·巴和穆罕默德·诺鲁兹。梦想控制：通过潜在想象学习行为。预印本 arXiv:1912.01603，2019a 年。


Danijar Hafner, Timothy Lillicrap, Ian Fischer, Ruben Villegas, David Ha, Honglak Lee, and James Davidson. Learning latent dynamics for planning from pixels. In International conference on machine learning, pp. 2555-2565. PMLR, 2019b.
达尼贾尔·哈夫纳、蒂莫西·利利克拉普、伊恩·费舍尔、鲁本·比列加斯、大卫·哈、李宏拉克和詹姆斯·戴维森。从像素中学习用于规划的潜在动力学。见《国际机器学习会议》，第 2555 - 2565 页。机器学习研究会议录，2019b 年。


Danijar Hafner, Timothy Lillicrap, Mohammad Norouzi, and Jimmy Ba. Mastering atari with discrete world models. arXiv preprint arXiv:2010.02193, 2020.
达尼贾尔·哈夫纳、蒂莫西·利利克拉普、穆罕默德·诺鲁兹和吉米·巴。用离散世界模型掌握雅达利游戏。预印本 arXiv:2010.02193，2020 年。


Danijar Hafner, Jurgis Pasukonis, Jimmy Ba, and Timothy Lillicrap. Mastering diverse domains through world models, 2024. URL https://arxiv.org/abs/2301.04104
达尼贾尔·哈夫纳、尤尔吉斯·帕苏科尼斯、吉米·巴和蒂莫西·利利克拉普。通过世界模型掌握不同领域，2024 年。URL https://arxiv.org/abs/2301.04104


Ari Holtzman, Jan Buys, Li Du, Maxwell Forbes, and Yejin Choi. The curious case of neural text degeneration. arXiv preprint arXiv:1904.09751, 2019.
阿里·霍尔茨曼、扬·拜斯、杜立、麦克斯韦·福布斯和崔叶真。神经文本退化的奇怪案例。预印本 arXiv:1904.09751，2019 年。


Geunwoo Kim, Pierre Baldi, and Stephen McAleer. Language models can solve computer tasks. Advances in Neural Information Processing Systems, 36, 2024.
金根佑、皮埃尔·巴尔迪和斯蒂芬·麦卡利尔。语言模型可以解决计算机任务。《神经信息处理系统进展》，36，2024 年。


Thomas Kipf, Elise van der Pol, and Max Welling. Contrastive learning of structured world models, 2020.URL https://arxiv.org/abs/1911.12247
托马斯·基普夫、伊莉斯·范德波尔和马克斯·韦林。结构化世界模型的对比学习，2020 年。URL https://arxiv.org/abs/1911.12247


Jing Yu Koh, Stephen McAleer, Daniel Fried, and Ruslan Salakhutdinov. Tree search for language model agents. arXiv preprint arXiv:2407.01476, 2024.
 Koh 景宇、斯蒂芬·麦卡利尔、丹尼尔·弗里德和鲁斯兰·萨拉胡丁诺夫。语言模型智能体的树搜索。预印本 arXiv:2407.01476，2024 年。


Harold W Kuhn. The hungarian method for the assignment problem. 50 Years of Integer Programming 1958-2008, pp. 29, 1995.
哈罗德·W·库恩。分配问题的匈牙利算法。《整数规划 50 年（1958 - 2008）》，第 29 页，1995 年。


Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, and Ion Stoica. Efficient memory management for large language model serving with pagedattention. In Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles, 2023.
权宇锡、李卓翰、庄思远、盛莹、郑连民、余浩、约瑟夫·E·冈萨雷斯、张浩和伊恩·斯托伊卡。使用分页注意力实现大语言模型服务的高效内存管理。收录于《ACM SIGOPS第29届操作系统原理研讨会论文集》，2023年。


Hanyu Lai, Xiao Liu, Iat Long Iong, Shuntian Yao, Yuxuan Chen, Pengbo Shen, Hao Yu, Hanchen Zhang, Xiaohan Zhang, Yuxiao Dong, et al. Autowebglm: Bootstrap and reinforce a large language model-based web navigating agent. arXiv preprint arXiv:2404.03648, 2024.
赖汉宇、刘晓、应龙、姚顺天、陈宇轩、申鹏博、余浩、张寒晨、张晓晗、董雨潇等。Autowebglm：引导并强化基于大语言模型的网页导航代理。预印本arXiv:2404.03648，2024年。


Yann LeCun. A path towards autonomous machine intelligence version 0.9. 2, 2022-06-27. Open Review, 62(1):1-62, 2022.
扬·勒昆。迈向自主机器智能之路0.9版。2022年6月27日。《开放评审》，62(1):1 - 62，2022年。


Evan Zheran Liu, Kelvin Guu, Panupong Pasupat, Tianlin Shi, and Percy Liang. Reinforcement learning on web interfaces using workflow-guided exploration. arXiv preprint arXiv:1802.08802, 2018.
刘哲然、凯尔文·顾、帕努蓬·帕苏帕特、史天霖和梁珀西。使用工作流引导探索在网页界面上进行强化学习。预印本arXiv:1802.08802，2018年。


Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. Advances in neural information processing systems, 36, 2024.
刘浩天、李春元、吴庆阳和李永宰。视觉指令调优。《神经信息处理系统进展》，36，2024年。


Michael Lutz, Arth Bohra, Manvel Saroyan, Artem Harutyunyan, and Giovanni Campagna. Wilbur: Adaptive in-context learning for robust and accurate web agents. arXiv preprint arXiv:2404.05902, 2024.
迈克尔·卢茨、阿思·博拉、曼维尔·萨罗扬、阿尔乔姆·哈鲁图尼扬和乔瓦尼·坎帕尼亚。威尔伯：用于健壮准确网页代理的自适应上下文学习。预印本arXiv:2404.05902，2024年。


Aman Madaan, Niket Tandon, Prakhar Gupta, Skyler Hallinan, Luyu Gao, Sarah Wiegreffe, Uri Alon, Nouha Dziri, Shrimai Prabhumoye, Yiming Yang, et al. Self-refine: Iterative refinement with self-feedback. Advances in Neural Information Processing Systems, 36, 2024.
阿曼·马达安、尼克特·坦顿、普拉哈尔·古普塔、斯凯勒·哈利南、高璐宇、莎拉·维格雷夫、乌里·阿隆、努哈·齐里、什里迈·普拉布莫耶、杨一鸣等。自我精炼：通过自我反馈进行迭代精炼。《神经信息处理系统进展》，36，2024年。


Junhyuk Oh, Xiaoxiao Guo, Honglak Lee, Richard L Lewis, and Satinder Singh. Action-conditional video prediction using deep networks in atari games. Advances in neural information processing systems, 28, 2015.
吴俊赫、郭晓晓、李鸿拉克、理查德·L·刘易斯和萨廷德·辛格。在雅达利游戏中使用深度网络进行动作条件视频预测。《神经信息处理系统进展》，28，2015年。


OpenAI. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023.
OpenAI。GPT - 4技术报告。预印本arXiv:2303.08774，2023年。


Jiayi Pan, Yichi Zhang, Nicholas Tomlin, Yifei Zhou, Sergey Levine, and Alane Suhr. Autonomous evaluation and refinement of digital agents. arXiv preprint arXiv:2404.06474, 2024.
潘佳怡、张一驰、尼古拉斯·汤姆林、周逸飞、谢尔盖·列维京和阿兰·苏尔。数字代理的自主评估与精炼。预印本arXiv:2404.06474，2024年。


Python. difflib - helpers for computing deltas, 2024. URL https://docs.python.org/3/ library/difflib.html Python Documentation.
Python。difflib - 计算差异的辅助工具，2024年。网址https://docs.python.org/3/ library/difflib.html 《Python文档》。


Tianlin Shi, Andrej Karpathy, Linxi Fan, Jonathan Hernandez, and Percy Liang. World of bits: An open-domain platform for web-based agents. In International Conference on Machine Learning, pp. 3135-3144. PMLR, 2017.
史天霖、安德烈·卡帕西、范林希、乔纳森·埃尔南德斯和梁珀西。比特世界：一个基于网页的代理开放领域平台。收录于《国际机器学习会议》，第3135 - 3144页。PMLR，2017年。


Noah Shinn, Federico Cassano, Ashwin Gopinath, Karthik Narasimhan, and Shunyu Yao. Reflexion: Language agents with verbal reinforcement learning. Advances in Neural Information Processing Systems, 36, 2024.
诺亚·辛、费德里科·卡萨诺、阿什温·戈皮纳特、卡蒂克·纳拉辛汉和姚顺宇。反思：具有语言强化学习的语言代理。《神经信息处理系统进展》，36，2024年。


Paloma Sodhi, SRK Branavan, and Ryan McDonald. Heap: Hierarchical policies for web actions using llms. arXiv preprint arXiv:2310.03720, 2023.
帕洛玛·索迪、SRK·布拉纳万和瑞安·麦克唐纳。堆：使用大语言模型的网页动作分层策略。预印本arXiv:2310.03720，2023年。


Richard S. Sutton. Dyna, an integrated architecture for learning, planning, and reacting. SIGART Bull., 2:160-163, 1990. URL https://api.semanticscholar.org/CorpusID: 207162288]
理查德·S·萨顿。Dyna：一种用于学习、规划和反应的集成架构。《SIGART通报》，2:160 - 163，1990年。网址https://api.semanticscholar.org/CorpusID: 207162288]


Ruoyao Wang, Graham Todd, Ziang Xiao, Xingdi Yuan, Marc-Alexandre Côté, Peter Clark, and Peter Jansen. Can language models serve as text-based world simulators? arXiv preprint arXiv:2406.06485, 2024a.
王若瑶、格雷厄姆·托德、肖子昂、袁兴迪、马克 - 亚历山大·科特、彼得·克拉克和彼得·詹森。语言模型能否充当基于文本的世界模拟器？预印本arXiv:2406.06485，2024a。


Zora Zhiruo Wang, Jiayuan Mao, Daniel Fried, and Graham Neubig. Agent workflow memory. arXiv preprint arXiv:2409.07429, 2024b.
王若竹（Zora Zhiruo Wang）、毛佳媛（Jiayuan Mao）、丹尼尔·弗里德（Daniel Fried）和格雷厄姆·纽比格（Graham Neubig）。代理工作流记忆。预印本 arXiv:2409.07429，2024b。


Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems, 35:24824-24837, 2022.
杰森·魏（Jason Wei）、王学志（Xuezhi Wang）、戴尔·舒尔曼斯（Dale Schuurmans）、马腾·博斯马（Maarten Bosma）、夏飞（Fei Xia）、埃德·池（Ed Chi）、乐国伟（Quoc V Le）、丹尼·周（Denny Zhou）等。思维链提示在大语言模型中引发推理。《神经信息处理系统进展》，35:24824 - 24837，2022。


Mengjiao Yang, Yilun Du, Kamyar Ghasemipour, Jonathan Tompson, Leslie Kaelbling, Dale Schu-urmans, and Pieter Abbeel. Learning interactive real-world simulators, 2024. URL https: //arxiv.org/abs/2310.06114
杨梦娇（Mengjiao Yang）、杜奕伦（Yilun Du）、卡米亚尔·加塞米普尔（Kamyar Ghasemipour）、乔纳森·汤普森（Jonathan Tompson）、莱斯利·凯尔布林（Leslie Kaelbling）、戴尔·舒尔曼斯（Dale Schu - urmans）和彼得·阿比尔（Pieter Abbeel）。学习交互式现实世界模拟器，2024。网址 https://arxiv.org/abs/2310.06114


Shunyu Yao, Howard Yang Chen, John, and Karthik Narasimhan. Webshop: Towards scalable real-world web interaction with grounded language agents. 2022. doi: 10.48550/arXiv.2207.01206.
姚顺宇（Shunyu Yao）、霍华德·杨·陈（Howard Yang Chen）、约翰（John）和卡斯蒂克·纳拉辛汉（Karthik Narasimhan）。网络商店：使用基于实际场景的语言代理实现可扩展的现实世界网络交互。2022。doi: 10.48550/arXiv.2207.01206。


Alex Zhang, Khanh Nguyen, Jens Tuyls, Albert Lin, and Karthik Narasimhan. Language-guided world models: A model-based approach to ai control. arXiv preprint arXiv:2402.01695, 2024.
亚历克斯·张（Alex Zhang）、阮庆（Khanh Nguyen）、延斯·图尔斯（Jens Tuyls）、阿尔伯特·林（Albert Lin）和卡斯蒂克·纳拉辛汉（Karthik Narasimhan）。语言引导的世界模型：一种基于模型的人工智能控制方法。预印本 arXiv:2402.01695，2024。


Boyuan Zheng, Boyu Gou, Jihyung Kil, Huan Sun, and Yu Su. Gpt-4v (ision) is a generalist web agent, if grounded. arXiv preprint arXiv:2401.01614, 2024.
郑博远（Boyuan Zheng）、苟博宇（Boyu Gou）、吉亨·基尔（Jihyung Kil）、孙欢（Huan Sun）和苏羽（Yu Su）。如果有实际场景支撑，GPT - 4V（视觉）是一个通用网络代理。预印本 arXiv:2401.01614，2024。


Longtao Zheng, Rundong Wang, Xinrun Wang, and Bo An. Synapse: Trajectory-as-exemplar prompting with memory for computer control. In The Twelfth International Conference on Learning Representations, 2023.
郑龙涛（Longtao Zheng）、王润东（Rundong Wang）、王鑫润（Xinrun Wang）和安博（Bo An）。突触：用于计算机控制的带记忆的轨迹示例提示。收录于第十二届国际学习表征会议，2023。


Shuyan Zhou, Frank F Xu, Hao Zhu, Xuhui Zhou, Robert Lo, Abishek Sridhar, Xianyi Cheng, Yonatan Bisk, Daniel Fried, Uri Alon, et al. Webarena: A realistic web environment for building autonomous agents. arXiv preprint arXiv:2307.13854, 2023.
周舒妍（Shuyan Zhou）、弗兰克·徐（Frank F Xu）、朱浩（Hao Zhu）、周旭辉（Xuhui Zhou）、罗伯特·罗（Robert Lo）、阿比谢克·斯里达尔（Abishek Sridhar）、程先义（Xianyi Cheng）、约纳坦·比斯克（Yonatan Bisk）、丹尼尔·弗里德（Daniel Fried）、乌里·阿隆（Uri Alon）等。网络竞技场：用于构建自主代理的真实网络环境。预印本 arXiv:2307.13854，2023。


Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny. Minigpt-4: Enhancing vision-language understanding with advanced large language models. arXiv e-prints, pp. arXiv- 2304, 2023.
朱德耀（Deyao Zhu）、陈军（Jun Chen）、沈小倩（Xiaoqian Shen）、李翔（Xiang Li）和穆罕默德·埃尔霍塞尼（Mohamed Elhoseiny）。MiniGPT - 4：利用先进大语言模型增强视觉 - 语言理解。arXiv预印本，第 arXiv - 2304 页，2023。


## APPENDIX
## 附录


## A FURTHER ANALYSES
## A 进一步分析


Extending our world model to take multimodal input. This study focuses on building text-based world models and web agents. In web navigation, however, visual information can also play a critical role (Liu et al. 2024; Zheng et al. 2024). Although HTML and accessibility tree do represent the visual structure of a webpage to some degree, it is helpful to use visual information in addition to textual information for improving the learning of dynamics in the environment (Koh et al. 2024). Thus, we extend our world model to a multimodal setting, inspired by the recent success of multimodal web agents. For our experiments, we use the Mind2Web (cross-task) dataset with Qwen2-VL-2B as the backbone Vision Language Model.
将我们的世界模型扩展为接受多模态输入。本研究专注于构建基于文本的世界模型和网络代理。然而，在网络导航中，视觉信息也可能起到关键作用（Liu 等人，2024；Zheng 等人，2024）。尽管 HTML 和可访问性树在一定程度上代表了网页的视觉结构，但除了文本信息之外，使用视觉信息有助于改善环境动态的学习（Koh 等人，2024）。因此，受近期多模态网络代理成功的启发，我们将世界模型扩展到多模态设置。在我们的实验中，我们使用 Mind2Web（跨任务）数据集，以 Qwen2 - VL - 2B 作为骨干视觉语言模型。


The results are shown in Table 8 Despite using a smaller parameter size compared to Llama-3.1-8B, the multimodal input leads to notable improvements across all metrics. These results demonstrate two key findings: (1) our framework can readily adapt to multimodal settings, and (2) the addition of visual modality provides clear benefits. Given that we used a naive approach to image input, we expect further improvement when incorporating more sophisticated image prompting techniques, such as SeeAct or Set-of-Marks, which could further enhance performance.
结果如表 8 所示。尽管与 Llama - 3.1 - 8B 相比使用了较小的参数规模，但多模态输入在所有指标上都带来了显著改进。这些结果表明了两个关键发现：（1）我们的框架可以轻松适应多模态设置；（2）添加视觉模态带来了明显的好处。鉴于我们对图像输入采用了简单的方法，我们预计在纳入更复杂的图像提示技术（如 SeeAct 或标记集）时会有进一步的改进，这可能会进一步提高性能。


Table 8: Comparison of multimodal and text-only world models on Mind2Web (cross-task).
表 8：Mind2Web（跨任务）上多模态和仅文本世界模型的比较。


<table><tr><td>Method</td><td>Modality of WM</td><td>EA</td><td>${\mathbf{{AF}}}_{1}$</td><td>Step SR</td><td>SR</td></tr><tr><td>MindAct</td><td>-</td><td>-</td><td>-</td><td>17.4</td><td>0.8</td></tr><tr><td>AWM</td><td>-</td><td>78.3</td><td>74.1</td><td>62.8</td><td>15.3</td></tr><tr><td>AWM+WMA (Llama-3.1-8B)</td><td>Text</td><td>79.9</td><td>75.8</td><td>67.0</td><td>25.4</td></tr><tr><td>AWM+WMA (Qwen2-VL-2B)</td><td>Text</td><td>79.2</td><td>75.1</td><td>65.0</td><td>23.7</td></tr><tr><td>AWM+WMA (Qwen2-VL-2B)</td><td>Text+Image</td><td>83.0</td><td>78.9</td><td>72.8</td><td>36.7</td></tr></table>
<table><tbody><tr><td>方法</td><td>工作记忆模态</td><td>EA</td><td>${\mathbf{{AF}}}_{1}$</td><td>Step SR</td><td>SR</td></tr><tr><td>MindAct</td><td>-</td><td>-</td><td>-</td><td>17.4</td><td>0.8</td></tr><tr><td>AWM</td><td>-</td><td>78.3</td><td>74.1</td><td>62.8</td><td>15.3</td></tr><tr><td>AWM+WMA (Llama-3.1-8B)</td><td>文本</td><td>79.9</td><td>75.8</td><td>67.0</td><td>25.4</td></tr><tr><td>AWM+WMA (Qwen2-VL-2B)</td><td>文本</td><td>79.2</td><td>75.1</td><td>65.0</td><td>23.7</td></tr><tr><td>AWM+WMA (Qwen2-VL-2B)</td><td>文本+图像</td><td>83.0</td><td>78.9</td><td>72.8</td><td>36.7</td></tr></tbody></table>


Application of multi-step planning. Beyond searching for the best action in a single step, we also explore finding optimal paths through multi-step search by iteratively using the world model. However, prediction errors accumulate as search depth increases when using only world model simulations, limiting real-world applicability. To address this, we adopt a hybrid approach combining simulated interactions for width exploration and actual environmental interactions for depth exploration. We use an A*-like search algorithm following Koh et al. (2024) and conduct experiments using the same settings as shown in Figure 6 (Ablation on the number of sampled actions).
多步规划的应用。除了在单步中搜索最佳动作外，我们还通过迭代使用世界模型来探索通过多步搜索寻找最优路径。然而，仅使用世界模型模拟时，随着搜索深度增加，预测误差会累积，限制了在真实世界的适用性。为了解决这一问题，我们采用了一种混合方法，结合用于宽度探索的模拟交互和用于深度探索的实际环境交互。我们使用类似 A* 的搜索算法，遵循 Koh 等人 (2024) 的方法，并在与图 6（采样动作数量的消融）中相同的设置下进行实验。


Table 9 shows the results. We find that increasing the depth from $\left( {w = 3,d = 1}\right)$ to $\left( {w = 3,d = 3}\right)$ improves performance $\left( {{17.1} \rightarrow  {19.6}}\right)$ . However,when comparing settings with the same exploration budget- $\left( {w = 9,d = 1}\right)$ vs. $\left( {w = 3,d = 3}\right)$ -allocating more budget to width shows slightly better performance. A specific challenge explains this issue: Many errors occur during the execution of optimal action sequences due to mismatches between search time and execution time. Even when a browser loads the same content from the same URL, element IDs calculated by the backend can change upon page reload. This means that the same element might have different IDs between the search and execution phases.
表 9 显示了结果。我们发现将深度从 $\left( {w = 3,d = 1}\right)$ 提高到 $\left( {w = 3,d = 3}\right)$ 会提升性能 $\left( {{17.1} \rightarrow  {19.6}}\right)$ 。然而，在比较具有相同探索预算的设置——$\left( {w = 9,d = 1}\right)$ 与 $\left( {w = 3,d = 3}\right)$——时，将更多预算分配给宽度显示出略微更好的性能。一个具体的挑战可以解释该问题：在执行最优动作序列时，由于搜索时与执行时的不匹配，发生了许多错误。即使浏览器从相同的 URL 加载相同的内容，后端计算的元素 ID 在页面重新加载后也可能变化。这意味着同一元素在搜索阶段和执行阶段之间可能有不同的 ID。


Table 9: Success rates for different width $\left( w\right)$ and depth $\left( d\right)$ .
表 9：不同宽度 $\left( w\right)$ 和深度 $\left( d\right)$ 的成功率。


<table><tr><td>Width $\left( w\right) /$ Depth $\left( d\right)$</td><td>d = 1</td><td>d = 2</td><td>d = 3</td></tr><tr><td>$w = 1$</td><td>7.1</td><td>-</td><td>10.7</td></tr><tr><td>$w = 2$</td><td>10.1</td><td>12.6</td><td>-</td></tr><tr><td>$w = 3$</td><td>17.1</td><td>-</td><td>19.6</td></tr><tr><td>$w = 9$</td><td>20.5</td><td>-</td><td>-</td></tr></table>
<table><tbody><tr><td>宽度 $\left( w\right) /$ 深度 $\left( d\right)$</td><td>d = 1</td><td>d = 2</td><td>d = 3</td></tr><tr><td>$w = 1$</td><td>7.1</td><td>-</td><td>10.7</td></tr><tr><td>$w = 2$</td><td>10.1</td><td>12.6</td><td>-</td></tr><tr><td>$w = 3$</td><td>17.1</td><td>-</td><td>19.6</td></tr><tr><td>$w = 9$</td><td>20.5</td><td>-</td><td>-</td></tr></tbody></table>


In-depth analysis on the world model. We conduct in-depth analyses to evaluate how well our world model predicts the next observation, focusing primarily on coverage. Coverage measures how much information from the ground-truth observation is successfully captured in the model's prediction. We define coverage as the ratio of the information of the ground-truth observation covered by the predicted next observation. Specifically, the coverage score is calculated as:
对世界模型的深入分析。我们进行深入分析以评估世界模型预测下一个观测的能力，主要关注覆盖度。覆盖度衡量地面真实观测中有多少信息被模型的预测成功捕捉。我们将覆盖度定义为被预测的下一个观测所覆盖的地面真实观测信息的比例。具体地，覆盖度得分计算如下：(4)



<table><tr><td rowspan="2"></td><td>#sentences in ground-truth observation covered by the predicted observation</td></tr><tr><td>#total sentences in ground-truth observation</td></tr></table>
<table><tbody><tr><td rowspan="2"></td><td>被预测观测覆盖的地面实况句子数</td></tr><tr><td>地面实况中的句子总数</td></tr></tbody></table>


For evaluation, we employ an LLM-as-a-judge approach using GPT-40 as the judge LLM. We begin by separating both predicted and ground-truth observations into sentences. Then, we use an LLM to determine whether information from the target sentence can be found in the source text, which consists of a list of sentences. We run the evaluation on 100 samples used in our preliminary analysis and compare our world model with a few different LLMs, including GPT-40-mini and GPT-40.
在评估中，我们采用将 LLM 作为裁判的方法，使用 GPT-40 作为裁判 LLM。我们首先把预测观察和真实观察都拆分成句子。然后，使用 LLM 判断目标句子中的信息是否能在由多句构成的源文本中找到。我们在用于初步分析的 100 个样本上运行该评估，并将我们的世界模型与包括 GPT-40-mini 和 GPT-40 在内的几种不同 LLM 进行比较。


Table 10: Coverage comparison of different approaches of implementing world models.
表 10：不同实现世界模型方法的覆盖率比较。


<table><tr><td>Model</td><td>Coverage (%)</td></tr><tr><td>GPT-40-mini</td><td>33.50</td></tr><tr><td>GPT-40</td><td>33.85</td></tr><tr><td>Ours</td><td>42.99</td></tr></table>
<table><tbody><tr><td>模型</td><td>覆盖率（%）</td></tr><tr><td>GPT-40-mini</td><td>33.50</td></tr><tr><td>GPT-40</td><td>33.85</td></tr><tr><td>我们的</td><td>42.99</td></tr></tbody></table>


## B EXPERIMENTAL DETAILS OF PRELIMINARY ANALYSES
## B 预实验细节


### B.1 PRELIMINARY ANALYSIS I
### B.1 预备分析 I


We formulate next state prediction as a binary classification task rather than a generation task for an easier and more accurate evaluation (it is non-trivial to evaluate machine-generated accessibility tree or HTML). Measuring the next state prediction capability as a generation task requires an additional human evaluation or off-the-shelf LLM judges, but it might introduce evaluation bias and there is no consensus that LLMs can judge this capability.
我们将下一个状态预测表述为二分类任务而非生成任务，以便更简单且更准确地评估（评估机器生成的可访问性树或 HTML 并非易事）。将下一个状态预测作为生成任务来衡量需要额外的人类评估或现成的 LLM 判定器，但这可能引入评估偏差，且尚无共识认为 LLM 能胜任此类评判。


To collect training objectives for next state prediction, we use difflib python library to calculate the lexical similarity between the golden next state and similar yet incorrect next state. Then, we select the top-1 similar yet wrong state as the negative next state and randomly shuffle the answer choices. The prompt used for next state prediction is shown in Figure 15 The interface for human annotation is shown in Figure 8
为收集下一个状态预测的训练目标，我们使用 difflib python 库计算黄金下一个状态与相似但错误的下一个状态之间的词汇相似度。然后，我们选择相似度最高的错误状态作为负样本，并随机打乱答案选项。用于下一个状态预测的提示见图 15，人类标注界面见图 8


### B.2 PRELIMINARY ANALYSIS II
### B.2 预备分析 II


We use greedy decoding for sampling a sequence of 9 negative actions from GPT-40-mini. Specifically, the LLM is instructed to generate 9 negative action candidates with the 2-shot demonstration. Prompts used for action selection in preliminary analysis II are shown in Figure 16 and 17
我们对 GPT-40-mini 采样的 9 个负动作序列使用贪心解码。具体地，指示 LLM 在 2-shot 演示下生成 9 个负动作候选。用于预备分析 II 中动作选择的提示见图 16 和 17


## C IMPLEMENTATION DETAILS
## C 实现细节


### C.1 WORLD MODEL
### C.1 世界模型


#### C.1.1 DATASET CONSTRUCTION
#### C.1.1 数据集构建


Instruction and trajectory collection from WebArena. As mentioned in §5.1, WebArena does not provide anything other than the test set. We thus synthesize user instructions and accordingly collect trajectories. In total, we obtain 14,200 instances using GPT-40-mini with CoT prompt provided in Zhou et al. (2023). These instances are used to collect training data for world models in WebArena.
来自 WebArena 的指令和轨迹收集。如 §5.1 所述，WebArena 不提供除测试集外的内容。因此我们合成用户指令并相应收集轨迹。总计使用带有 Zhou et al. (2023) 提供的 CoT 提示的 GPT-40-mini 获得 14,200 个实例。这些实例用于为 WebArena 中的世界模型收集训练数据。


---



https://docs.python.org/3/library/difflib.html



---



Transition-focused observation abstraction. We implement the Hungarian algorithm (Kuhn, 1995) using munkres python package 7 Details of the algorithm are in Algorithm 1 TaO in Algorithm 1 stands for Transition-aware Observation, and denotes the direct observation output from the Hungarian algorithm used in §4.1.2 Then, using the output from the algorithm, we prompt an LLM to make a free-form description that captures the state transitions. The prompt used for producing free-form description is shown in Figure 18 and Figure 19
面向转换的观察抽象。我们使用 munkres python 包实现匈牙利算法 (Kuhn, 1995) 7 算法细节见算法 1。算法 1 中的 TaO 代表 Transition-aware Observation，表示在 §4.1.2 中使用的来自匈牙利算法的直接观察输出。然后，使用算法输出，我们提示 LLM 生成捕捉状态转换的自由形式描述。用于生成自由形式描述的提示见图 18 和图 19


Algorithm 1: Observation tree state matching for $\Delta \left( {{o}_{t},{o}_{t + 1}}\right)$ in $\text{ § }{4.1.2}$
算法 1：用于 $\Delta \left( {{o}_{t},{o}_{t + 1}}\right)$ 在 $\text{ § }{4.1.2}$ 中的观察树状态匹配


---



Input : States ${o}_{t} = \left\lbrack  {{e}_{0}^{t},\ldots ,{e}_{n - 1}^{t}}\right\rbrack  ,{o}_{t + 1} = \left\lbrack  {{e}_{0}^{t + 1},\ldots ,{e}_{m - 1}^{t + 1}}\right\rbrack$ . Each ${e}_{i}$ has name ${n}_{i}$ ,role ${r}_{i}$ ,
输入：状态 ${o}_{t} = \left\lbrack  {{e}_{0}^{t},\ldots ,{e}_{n - 1}^{t}}\right\rbrack  ,{o}_{t + 1} = \left\lbrack  {{e}_{0}^{t + 1},\ldots ,{e}_{m - 1}^{t + 1}}\right\rbrack$。每个 ${e}_{i}$ 具有 名称 ${n}_{i}$，角色 ${r}_{i}$，


		location ${l}_{i}$ . Weights ${\omega }_{n},{\omega }_{r},{\omega }_{l}$ .
		位置 ${l}_{i}$。权重 ${\omega }_{n},{\omega }_{r},{\omega }_{l}$。


Output: ${S}_{t + 1}^{\mathrm{{TaO}}}$
输出：${S}_{t + 1}^{\mathrm{{TaO}}}$


$U \leftarrow  \varnothing$



if $\operatorname{len}\left( {o}_{t + 1}\right)  \leq  \tau  \cdot  \operatorname{len}\left( {o}_{t}\right)$ then
如果 $\operatorname{len}\left( {o}_{t + 1}\right)  \leq  \tau  \cdot  \operatorname{len}\left( {o}_{t}\right)$ 然后


	#Construct cost matrix for Hungarian matching
	##为匈牙利匹配构建代价矩阵


	${C}_{i,j} \leftarrow  {\omega }_{n} \cdot  {\mathbf{1}}_{{n}_{i}^{t} = {n}_{j}^{t + 1}} + {\omega }_{r} \cdot  {\mathbf{1}}_{{r}_{i}^{t} = {r}_{j}^{t + 1}} + {\omega }_{l} \cdot  \left| {{l}_{i}^{t} - {l}_{j}^{t + 1}}\right|$



	#Apply Hungarian algorithm to find optimal matching
	##应用匈牙利算法以找到最优匹配


	${M}^{ * } \leftarrow  \mathop{\operatorname{argmin}}\limits_{M}\mathop{\sum }\limits_{{i,j}}{C}_{i,j} \cdot  {M}_{i,j}$



	#Identify unmatched elements
	##识别未匹配元素


	$U \leftarrow  \left\{  {j \mid  {M}_{i,j}^{ * } = 0,\forall i \in  \{ 0,\ldots ,n - 1\} }\right\}$



end
end


if $\operatorname{len}\left( U\right)  \geq  m - n$ or $U = \varnothing$ then
if $\operatorname{len}\left( U\right)  \geq  m - n$ or $U = \varnothing$ then


	${S}_{t + 1}^{\mathrm{{TaO}}} \leftarrow  {o}_{t + 1}$



else
else


	#Construct TaO state based on unmatched and nearby elements
	##基于未匹配及相邻元素构建 TaO 状态


	${S}_{t + 1}^{\mathrm{{TaO}}} \leftarrow  \left\lbrack  {{e}_{j}^{t + 1} \mid  j \in  U\text{ or }\left( {\operatorname{len}\left( U\right)  \leq  x\text{ and }\mathop{\min }\limits_{{u \in  U}}\left| {u - j}\right|  \leq  y}\right) }\right\rbrack$



end
end


---



#### C.1.2 TRAINING
#### C.1.2 训练


For world models and value functions, we use a learning rate of 1e-5 and spend around 3 GPU hours training them for 2 epochs on 8 RTX 4090 GPUs.
对于世界模型和值函数，我们使用 1e-5 的学习率，并在 8 块 RTX 4090 GPU 上训练约 3 GPU 小时，进行 2 个 epoch。


### C.2 INFERENCE
### C.2 推理


We use top- $p$ decoding with $p = {1.0}$ for sampling 20 actions from the model following (Koh et al., 2024). The three most frequent actions among the sampled actions are to be selected, and a next state prediction is to be performed for these actions. The prompt used for the next state prediction of world models is shown in Figure 20. For each predicted next state, a reward is calculated using the value function (the prompt is in Figure 21), and the action with the highest reward is finally selected. We use vLLM (Kwon et al. 2023) to run inference of fine-tuned LLMs.
我们采用带 $p = {1.0}$ 的 top-$p$ 解码，按 (Koh et al., 2024) 从模型中采样 20 个动作。从采样动作中选择出现频率最高的三个动作，并对这些动作进行下一状态预测。世界模型用于下一状态预测的提示见图 20。对于每个预测的下一状态，使用价值函数计算奖励（提示见图 21），最终选择奖励最高的动作。我们使用 vLLM (Kwon et al. 2023) 运行微调后大模型的推理。


### C.3 DETAILS ON WEBARENA
### C.3 WEBARENA 细节


To ensure fair comparison and reproducibility, we conduct our experiments using the WebArena environment. Specifically, we utilize an Amazon Web Services (AWS) EC2 instance pre-configured with the Docker environment for WebArena ${}^{8}$ This setup is identical to the experimental configuration employed by Zhou et al. (2023) in their original study. By using this standardized environment, we maintain consistency with previous research and facilitate direct comparisons of our results with those reported in the literature. The WebArena Docker environment encapsulates all necessary dependencies, web interfaces, and evaluation metrics, ensuring that our experiments are conducted under controlled and replicable conditions. Details of each domain are explained below.
为确保公平比较与可复现性，我们在 WebArena 环境中开展实验。具体地，我们使用预配置了 WebArena Docker 环境的 Amazon Web Services (AWS) EC2 实例 ${}^{8}$。该设置与 Zhou et al. (2023) 原始研究中使用的实验配置相同。通过使用此标准化环境，我们与先前研究保持一致，便于与文献中报告的结果直接比较。WebArena Docker 环境封装了所有必要依赖、网页接口和评估指标，确保我们的实验在受控且可复现的条件下进行。各领域的详细信息如下。


---



https://pypi.org/project/munkres/



${}^{8}$ https://github.com/web-arena-x/webarena/blob/main/environment_docker/README.md#pre-installed-amazon-machine-image
${}^{8}$ https://github.com/web-arena-x/webarena/blob/main/environment_docker/README.md#pre-installed-amazon-machine-image


---



- Shopping: E-commerce platforms supporting online shopping activities (e.g., Amazon, and eBay). In this website, the agent can search and make an order for realistic items.
- 购物：支持在线购物活动的电子商务平台（例如 Amazon 和 eBay）。在该网站中，代理可以搜索并为真实商品下单。


- CMS: Content Management Systems that manage the creation and revision of digital content (e.g., online store management).
- CMS：管理数字内容创建与修订的内容管理系统（例如，在线商店管理）。


- Reddit: Social forum platforms for opinion exchanges.
- Reddit：用于观点交流的社交论坛平台。


- Gitlab: Collaborative development platforms for software development.
- Gitlab：用于协同软件开发的平台。


- Map: Navigation and searching for information about points of interest such as institutions or locations. For Map domain, we use the online openstreetmap website ${}^{9}$ since the button for searching a route of the provided docker does not properly work. This issue is also raised in the official WebArena github 10
- Map：用于导航及检索机构或地点等兴趣点信息。对于 Map 领域，我们使用在线 openstreetmap 网站 ${}^{9}$，因为所提供 docker 中用于搜索路线的按钮无法正常工作。该问题也在官方 WebArena github 中被提出 10


### C.4 DETAILS ON MIND2WEB
### C.4 关于 MIND2WEB 的详细信息


For running our experiments on Mind2Web, we obtain Mind2Web data from the official project page ${}^{11}$ We use the implementation of Wang et al. (2024b) to calculate the evaluation metrics,EA, ${\mathrm{{AF}}}_{1}$ ,Step SR,and SR. Each action in the sequence comprises a (Target Element,Operation) pair, We measure Element Accuracy (EA) which compares the selected element with all ground-truth elements,and Action $\mathrm{F}1\left( {\mathrm{{AF}}}_{1}\right)$ that calculates token-level $\mathrm{F}1$ score for the predicted action. Each step of the task is evaluated independently with the ground-truth history provided. We then define Step Success Rate (Step SR) and Success Rate (for the whole task). For calculating Step Success Rate (Step SR) and Success Rate (SR), a step is regarded as successful only if both the selected element and the predicted action is correct. A task is regarded successful only if all steps have succeeded. For step-wise metrics, we report macro average across tasks.
在 Mind2Web 上运行实验时，我们从官方项目页面 ${}^{11}$ 获取 Mind2Web 数据。我们使用 Wang et al. (2024b) 的实现来计算评估指标 EA、${\mathrm{{AF}}}_{1}$、Step SR 和 SR。序列中的每个动作由 (Target Element,Operation) 对组成。我们衡量元素准确率（EA），将所选元素与所有真实元素比较，以及动作 $\mathrm{F}1\left( {\mathrm{{AF}}}_{1}\right)$，对预测动作计算基于标记的 $\mathrm{F}1$ 分数。任务的每一步在提供的真实历史下独立评估。然后我们定义步骤成功率（Step SR）和成功率（针对整个任务）。在计算 Step SR 和 SR 时，仅当所选元素和预测动作均正确时，该步骤才被视为成功。仅当所有步骤均成功时，任务才被视为成功。对于逐步指标，我们报告跨任务的宏平均。


### C.5 IMPLEMENTATION DETAILS OF BASELINES
### C.5 基线实现细节


Vanilla CoT (Zhou et al., 2023) For fair comparison, we first sample 20 actions with top- $p$ sampling similar to ours. We use the original CoT prompt from Zhou et al. (2023). Then we choose the most frequent action as the final action. We use the prompt in Figure 22.
Vanilla CoT (Zhou et al., 2023) 为了公平比较，我们首先以与我们相似的 top-$p$ 采样抽取 20 个动作。我们使用 Zhou et al. (2023) 的原始 CoT 提示，然后选择最频繁的动作作为最终动作。我们使用图 22 中的提示。


Tree Search Agent (Koh et al., 2024) We use the codes from the official Github repository for implementing Tree search agent ${}^{12}$ For time and cost analysis on this agent,we run Tree search agent on 10% of WebArena instances due to its excessive cost.
Tree Search Agent (Koh et al., 2024) 我们使用官方 Github 仓库中的代码来实现 Tree search agent ${}^{12}$。鉴于该代理的高昂成本，我们仅在 10% 的 WebArena 实例上运行 Tree search agent 以进行时间和成本分析。


Agent Workflow Memory (Wang et al. 2024b) We use the codes from the official github repository to implement Agent Workflow Memory (AWM). We use GPT-3.5-Turob to create workflow memory from the train data of Mind2Web dataset. During our experiments, we find that the candidate generation module of MindAct (Deng et al. 2024) significantly degrades the original performance. This module calculates the relevance score of each element to the query so that web agents can predict action with more shortened observation. We provide the results of both settings with and without the candidate generation module.
Agent Workflow Memory (Wang et al. 2024b) 我们使用官方 github 仓库中的代码来实现 Agent Workflow Memory (AWM)。我们使用 GPT-3.5-Turbo 从 Mind2Web 数据集的训练数据创建工作流记忆。在实验中，我们发现 MindAct (Deng et al. 2024) 的候选生成模块显著降低了原始性能。该模块计算每个元素与查询的相关性得分，使 Web 代理可以在更精简的观察下预测动作。我们提供了有该候选生成模块和无该模块两种设置的结果。


For certain baselines, we obtain the performance from the original papers, which are marked with "*" in the result tables.
对于某些基线，我们从原始论文中获取性能，这些在结果表中以“*”标注。


C. 6 ISSUE REGARDING THE ELEMENT FILTERING MODULE OF MINDACT
C.6 关于 MINDACT 元素过滤模块的问题


The element selection module proposed by Deng et al. (2024) used for filtering out irrelevant elements in the extremely long HTML content to avoid confusion. This element selection module is adapted to the suggested baseline in Mind2Web paper, MindAct and widely applied to the following methods (Wang et al. 2024b; Zheng et al. 2024) including the AWM baseline. However, we find that this module introduces a significant performance decrease, by removing not only the irrelevant items but also the relevant ones. Thus, we re-implemented AWM in both with and without the filtering module.
Deng et al. (2024) 提出的元素选择模块用于在极长的 HTML 内容中过滤掉无关元素以避免混淆。该元素选择模块被调整用于 Mind2Web 论文中建议的基线 MindAct，并广泛应用于后续方法（Wang et al. 2024b；Zheng et al. 2024），包括 AWM 基线。然而，我们发现该模块会显著降低性能，不仅移除了无关项，也移除了相关项。因此，我们对 AWM 进行了有该过滤模块和无该模块两种实现的重现。


---



\$https://www.openstreetmap.org/
\$https://www.openstreetmap.org/


"https://github.com/web-arena-x/webarena/issues/159
"https://github.com/web-arena-x/webarena/issues/159


"https://osu-nlp-group.github.io/Mind2Web/
"https://osu-nlp-group.github.io/Mind2Web/


${}^{12}$ https://github.com/kohjingyu/search-agents
${}^{12}$ https://github.com/kohjingyu/search-agents


---



C. 7 INSTANCE IDS OF ADAPTED TASKS FOR WEBARENA
C. 7 WebArena 适配任务的实例 ID


We randomly sampled 200 instances from WebArena (50, 50, and 100 instances from Shopping, Gitlab, and Map, respectively) We sample 100 instances from the Map domain as it is cost- and time-efficient due to its short inference time. We provide the full list of task ids below:
我们从 WebArena 中随机抽取了 200 个实例（分别从购物、Gitlab 和地图领域抽取了 50、50 和 100 个实例）。由于地图领域推理时间短，成本和时间效率高，我们从该领域抽取了 100 个实例。以下是完整的任务 ID 列表：


---



- Shopping: 49, 51, 96, 144, 146, 158, 162, 164, 165, 188, 189, 190, 226, 231, 235, 238,
- 购物：49、51、96、144、146、158、162、164、165、188、189、190、226、231、235、238


	263,274,278,281,300,313,319,333,337,352,355,362,376,385,386,387,432,467,



	468,469,506,509,511,513,515,517,518,521,528,529,530,531,587,589



- GitLab: 156, 174, 177, 178, 205, 207, 297, 305, 306, 311, 315, 317, 339, 341, 349, 357,
- GitLab：156、174、177、178、205、207、297、305、306、311、315、317、339、341、349、357


	389,395,396,416,418,422,441,452,475,482,483,523,524,535,537,552,553,563,



	564,566,569,658,662,664,669,670,736,751,783,787,789,800,803,810



- Map: 7, 8, 9, 10, 16, 17, 18, 19, 20, 33, 34, 35, 36, 37, 38, 40, 52, 53, 54, 55, 56, 57, 58,
- 地图：7、8、9、10、16、17、18、19、20、33、34、35、36、37、38、40、52、53、54、55、56、57、58


	60,61,70,71,72,73,75,76,80,81,82,83,84,86,87,88,89,90,91,92,93,97,98,99,



	100,101,137,138,139,140,151,153,154,218,219,220,221,222,223,224,236,248,



	249,250,251,252,253,254,256,257,287,356,364,365,366,367,369,371,372,373,



	377,378,380,381,382,383,757,758,759,760,761,762,763,764,765,766,767



---



## D DETAILS OF FURTHER ANALYSES
## D 进一步分析详情


### D.1 SELF-REFINE
### D.1 自我改进


We implement self-refine using the prompt shown in Figure 23 Specifically, we first generate a single action using the CoT prompt and we obtain the feedback from the value model used in our method. Then, we refine the action according to the feedback.
我们使用图 23 所示的提示实现自我改进。具体来说，我们首先使用思维链提示生成单个动作，并从我们方法中使用的值模型获取反馈。然后，根据反馈改进该动作。


### D.2 ERROR TYPE ANALYSIS AND EXAMPLES
### D.2 错误类型分析及示例


We sample 50 errors from the inference results in WebArena for our error analyses. The numbers of selected samples by domains are Shopping: 8, CMS: 11, Gitlab: 11, Reddit: 10, and Map: 10. The examples of the four error types are mentioned in §6.2 and are respectively shown below.
我们从 WebArena 的推理结果中抽取了 50 个错误进行错误分析。按领域选择的样本数量分别为：购物 8 个、CMS 11 个、Gitlab 11 个、Reddit 10 个和地图 10 个。§6.2 中提到了四种错误类型的示例，分别如下所示。


- Low competence in web elements/functions: Figure 9
- 对网页元素/功能的能力不足：图 9


- Counterfactual imagination: Figure 10
- 反事实想象：图 10


- Correct yet overly generic statement: Figure 11
- 正确但过于笼统的陈述：图 11


- Others: Figure 12
- 其他：图 12


## E EXAMPLES OF SUCCESSFUL INFERENCE
## E 成功推理示例


We provide several successful examples of our WMA web agents:
我们提供了几个 WMA 网络代理的成功案例：


- Inference on Mind2Web: Figure 13
- Mind2Web 上的推理：图 13


- Inference on WebArena: Figure 14
- WebArena 上的推理：图 14


## F PROMPTS
## F 提示


The following are prompts utilized in our study:
以下是我们研究中使用的提示：


- Prompt for next state prediction in preliminary analysis I in Figure 15.
- 图 15 中初步分析 I 里下一状态预测的提示。


- Prompts for action selection in preliminary analysis II in Figure 16 and Figure 17.
- 图 16 和图 17 中初步分析 II 里动作选择的提示。


- Prompt for refining TaO output in Figure 18
- 图 18 中细化 TaO 输出的提示


- Prompt for transition focused observation abstraction in Figure 19
- 图 19 中专注于过渡的观察抽象的提示


- Prompt used for the next state prediction of the world model in Figure 20
- 图 20 中世界模型下一状态预测使用的提示


- Prompt for reward calculation in value function in Figure 21
- 图 21 中价值函数里奖励计算的提示


- Prompt for baseline action prediction using accessibility tree with CoT in Figure 22
- 图 22 中使用带思维链的可达性树进行基线动作预测的提示


- Prompt for self-refining in Figure 23
- 图 23 中自我细化的提示


## UI Preview
## 用户界面预览


V Current Observation
V 当前观察


Tab 0 (current): Elmwood Inn Fine Teas, Orange Vanilla Caffeine-free Fruit Infusion, 16-Ounce Pouch [16131] RootWebArea 'Elmwood Inn Fine Teas, Orange Vanilla Caffeine-free Fruit Infusion, 16- Ounce Pouch' focused: True [16177] link 'My Account' [16178] link 'My Wish List 39 items' [16179] link 'Sign Out' [19601] StaticText 'Welcome to One Stop Market' [16152] link 'Skip to Content' [16146] link 'store logo' [16154] img 'one_stop_market_logo' [16155] link '\\ue611 My Cart' [16268] StaticText 'Search' [16219] combobox '\\ue615 Search' autocomplete: both hasPopup: listbox required: False expanded: False [16271] link 'Advanced Search' [16204] button 'Search' disabled: True [19588] tablist " multiselectable: False orientation: horizontal [19590] tabpanel " [17834] menu " orientation: vertical ...(omitted)
标签 0（当前）：埃尔姆伍德客栈优质茶，橙香草无咖啡因水果茶，16 盎司袋装 [16131] RootWebArea '埃尔姆伍德客栈优质茶，橙香草无咖啡因水果茶，16 盎司袋装' 聚焦：是 [16177] 链接 '我的账户' [16178] 链接 '我的愿望清单 39 项' [16179] 链接 '退出登录' [19601] 静态文本 '欢迎来到一站式市场' [16152] 链接 '跳至内容' [16146] 链接 '商店标志' [16154] 图片 '一站式市场标志' [16155] 链接 '\\ue611 我的购物车' [16268] 静态文本 '搜索' [16219] 组合框 '\\ue615 搜索' 自动完成：两者皆有 有弹出框：列表框 必需：否 展开：否 [16271] 链接 '高级搜索' [16204] 按钮 '搜索' 禁用：是 [19588] 标签列表 “ 多选：否 方向：水平 [19590] 标签面板 “ [17834] 菜单 “ 方向：垂直 ...(省略)


## Current Action
## 当前操作


click [16932] where [16932] is link 'Add to Wish List'
点击 [16932]，其中 [16932] 为链接 'Add to Wish List'


## Next State Choices
## 下一状态选择


Tab \{idx\} [19748] RootWebArea 'My Wish List' focused: True busy: 1 [19794] link 'My Account' [19795] link 'My Wish List' [19796] link 'Sign Out' [19769] link 'Skip to Content' [19763] link 'store logo' [19771] img 'one_stop_market_logo' [19772] link '\\ue611 My Cart' [19909] StaticText 'Search' [19845] combobox '\\ue615 Search' autocomplete: both hasPopup: listbox required: False expanded: False [19912] link 'Advanced Search' [19824] button 'Search' [19825] link 'Beauty & Personal Care' [19827] link 'Sports & Outdoors' [19829] link 'Clothing, Shoes & Jewelry' [19831] link 'Home & Kitchen' [19833] link 'Office Products' [19835] link 'Tools & Home Improvement' [21595] StaticText '9' ...(omitted) Tab \{idx\} [17045] RootWebArea 'My Wish List' focused: True busy: 1 [17078] link 'My Account' [17079] link 'My Wish List' [17080] link 'Sign Out' [17061] link 'Skip to Content' [17058] link 'store logo' [17063] img 'one_stop_market_logo' [17064] link '\\ue611 My Cart' [17206] StaticText 'Search' [17142] combobox '\\ue615 Search' autocomplete: both hasPopup: listbox required: False expanded: False [17209] link 'Advanced Search' [17121] button 'Search' [17122] link 'Beauty & Personal Care' [17124] link 'Sports & Outdoors' [17126] link 'Clothing, Shoes & Jewelry' [17128] link 'Home & Kitchen' [17130] link 'Office Products' [17132] link 'Tools & Home Improvement' [17134] link 'Health & Household' ...(omitted)
选项卡 \{idx\} [19748] 根网页区域 'My Wish List' 获焦：True 忙碌：1 [19794] 链接 'My Account' [19795] 链接 'My Wish List' [19796] 链接 'Sign Out' [19769] 链接 'Skip to Content' [19763] 链接 'store logo' [19771] 图片 'one_stop_market_logo' [19772] 链接 '\\ue611 My Cart' [19909] 静态文本 'Search' [19845] 组合框 '\\ue615 Search' 自动完成：both 有弹出：listbox 必需：False 展开：False [19912] 链接 'Advanced Search' [19824] 按钮 'Search' [19825] 链接 'Beauty & Personal Care' [19827] 链接 'Sports & Outdoors' [19829] 链接 'Clothing, Shoes & Jewelry' [19831] 链接 'Home & Kitchen' [19833] 链接 'Office Products' [19835] 链接 'Tools & Home Improvement' [21595] 静态文本 '9' ...(省略) 选项卡 \{idx\} [17045] 根网页区域 'My Wish List' 获焦：True 忙碌：1 [17078] 链接 'My Account' [17079] 链接 'My Wish List' [17080] 链接 'Sign Out' [17061] 链接 'Skip to Content' [17058] 链接 'store logo' [17063] 图片 'one_stop_market_logo' [17064] 链接 '\\ue611 My Cart' [17206] 静态文本 'Search' [17142] 组合框 '\\ue615 Search' 自动完成：both 有弹出：listbox 必需：False 展开：False [17209] 链接 'Advanced Search' [17121] 按钮 'Search' [17122] 链接 'Beauty & Personal Care' [17124] 链接 'Sports & Outdoors' [17126] 链接 'Clothing, Shoes & Jewelry' [17128] 链接 'Home & Kitchen' [17130] 链接 'Office Products' [17132] 链接 'Tools & Home Improvement' [17134] 链接 'Health & Household' ...(省略)


## Choose the next observation
## 选择下一个观察


1 ${\mathrm{A}}^{\left\lbrack  1\right\rbrack  }$ 1 ${\mathrm{B}}^{\left\lbrack  2\right\rbrack  }$



Figure 8: Human annotation interface for preliminary analysis I in §3.1
图 8：§3.1 中初步分析 I 的人工标注界面


<img src="https://cdn.noedgeai.com/bo_d4nfperef24c73bbffn0_20.jpg?x=407&y=329&w=984&h=607&r=0"/>



Figure 9: Erroneous example (Low competence in web elements/functions). Although the agent does not delete old texts on the search bar before entering the new keyword 'restaurants near CMU Posner Hall', the world model still expects the next observation to show the desired search results.
图 9：错误示例（对网页元素/功能能力低）。尽管代理在输入新关键词 'restaurants near CMU Posner Hall' 前未清除搜索栏中的旧文本，世界模型仍然期望下一观察展示期望的搜索结果。


<img src="https://cdn.noedgeai.com/bo_d4nfperef24c73bbffn0_20.jpg?x=305&y=1294&w=1127&h=597&r=0"/>



Figure 10: Erroneous example (Counterfactual imagination). The model predicts that specific products (96 TY CITY86 Bmw 740i Limited Collector Hoodie Men's Close; Toyota 86 Bad Institute Monkey Champagne Cup, Volkswagen A9 Bug Pick Dead Red) will appear in the next observation, while this specific page does not list them as the products for sell.
图 10：错误示例（反事实想象）。模型预测特定产品（96 TY CITY86 Bmw 740i Limited Collector Hoodie Men's Close；Toyota 86 Bad Institute Monkey Champagne Cup，Volkswagen A9 Bug Pick Dead Red）会出现在下一观察中，而该页面并未将这些列为在售产品。


<img src="https://cdn.noedgeai.com/bo_d4nfperef24c73bbffn0_21.jpg?x=375&y=358&w=1067&h=563&r=0"/>



Figure 11: Erroneous example (Correct yet overly generic statements). "Comprehensive layout" and "various order-related functionalities" are ambiguous and unclear expressions.
图 11：错误示例（正确但过于笼统的表述）。“综合布局”和“各种订单相关功能”是模糊且不清晰的表达。


<img src="https://cdn.noedgeai.com/bo_d4nfperef24c73bbffn0_21.jpg?x=382&y=1268&w=1040&h=670&r=0"/>



Figure 12: Erroneous example (Others). The predicted next state (i.e., contributions and activities) is actually several steps further away from the current time step.
图 12：错误示例（其他）。预测的下一个状态（即贡献和活动）实际上比当前时间步晚了好几步。


<img src="https://cdn.noedgeai.com/bo_d4nfperef24c73bbffn0_22.jpg?x=316&y=696&w=1166&h=870&r=0"/>



Figure 13: Successful example (Mind2Web). WMA web agent successfully inferences on the Mind2Web benchmark (menards task #0). Using the policy model (i.e., GPT-4o), WMA web agent selects the most proper action click [208] by leveraging its learned environment dynamics.
图 13：成功示例（Mind2Web）。WMA 网页代理在 Mind2Web 基准（menards 任务 #0）上成功推断。使用策略模型（即 GPT-4o），WMA 网页代理凭借其学得的环境动态选择了最合适的操作 click [208]。


<img src="https://cdn.noedgeai.com/bo_d4nfperef24c73bbffn0_23.jpg?x=312&y=280&w=1176&h=919&r=0"/>



Figure 14: Successful example (WebArena). WMA web agent successfully inferences on Gitlab domain in the WebArena benchmark (instance #175). Using the policy model (i.e., GPT-40), WMA web agent selects the most proper action click [88] by leveraging its learned environment dynamics.
图 14：成功示例（WebArena）。WMA 网页代理在 WebArena 基准的 Gitlab 域（实例 #175）上成功推断。使用策略模型（即 GPT-40），WMA 网页代理凭借其学得的环境动态选择了最合适的操作 click [88]。


---



<img src="https://cdn.noedgeai.com/bo_d4nfperef24c73bbffn0_23.jpg?x=329&y=1531&w=1146&h=463&r=0"/>



---



Figure 15: The prompt for preliminary analysis I in §3.1 Next state prediction
图 15：§3.1 中初步分析 I 的提示 文本 下一状态预测


Prompt for preliminary analysis II: action selection w/o next state
初步分析 II 的提示：无下一状态时的动作选择


You are an autonomous intelligent agent tasked with navigating a web browser. You will be given web-based tasks. These tasks will be accomplished by selecting the most appropriate action from a list of choices.
你是一个自主智能代理，负责在网页浏览器中导航。你将被分配基于网页的任务。通过从备选动作列表中选择最合适的动作来完成这些任务。


Here's the information you'll have:
以下是你将拥有的信息：


The user's objective: This is the task you're trying to complete.
用户的目标：这是你要完成的任务。


The current web page's accessibility tree: This is a simplified representation of the webpage, providing key information.
当前网页的无障碍树：这是网页的简化表示，提供关键信息。


The current web page's URL: This is the page you're currently navigating.
当前网页的 URL：这是你当前访问的页面。


The open tabs: These are the tabs you have open.
已打开的标签页：这些是你已打开的标签页。


The previous action: This is the action you just performed. It may be helpful to track your progress.
上一步操作：这是你刚执行的操作，有助于追踪进度。


For each step, you will be presented with 10 possible actions (A to J). Your task is to select the most appropriate
在每一步，你会看到 10 个可能的操作（A 到 J）。你的任务是选择最合适的


action to progress towards completing the user's objective.
操作以推进完成用户的目标。


The actions fall into several categories:
这些操作分为若干类别：


Page Operation Actions:
页面操作类：


Click: This action clicks on an element with a specific id on the webpage.
点击：此操作在网页上点击具有特定 id 的元素。


Type: Use this to type content into a field with a specific id. By default, the "Enter" key is pressed after typing unless specified otherwise.
输入：用于在具有特定 id 的字段中输入内容。默认在输入后按下“Enter”键，除非另有说明。


Hover: Hover over an element with a specific id.
悬停：将鼠标悬停在具有特定 id 的元素上。


Press: Simulates the pressing of a key combination on the keyboard (e.g., Ctrl+v).
按键：模拟键盘上的按键组合（例如 Ctrl+v）。


Scroll: Scroll the page up or down.
滚动：向上或向下滚动页面。


Tab Management Actions:
标签页管理类：


New tab: Open a new, empty browser tab.
新建标签页：打开一个新的空白浏览器标签页。


Tab focus: Switch the browser's focus to a specific tab using its index.
切换标签聚焦：使用索引将浏览器焦点切换到指定标签。


Close tab: Close the currently active tab.
关闭标签：关闭当前活动的标签。


URL Navigation Actions:
URL 导航操作：


Goto: Navigate to a specific URL.
前往：导航到指定 URL。


Go back: Navigate to the previously viewed page.
后退：导航到之前查看的页面。


Go forward: Navigate to the next page (if a previous 'go_back' action was performed).
前进：导航到下一页（如果先前执行过“后退”操作）。


Completion Action:
完成操作：


Stop: Select this action when you believe the task is complete. If the objective is to find a text-based answer, the answer will be included in the action description.
停止：当你认为任务已完成时选择此操作。如果目标是找到基于文本的答案，答案将包含在操作描述中。


Additional information:
附加信息：


If you want to visit other websites, check out the homepage at http://homepage.com.It has a list of websites you can visit.
如果你想访问其他网站，请查看主页 http://homepage.com。它列出了可访问的网站清单。


http://homepage.com/password.html lists all the account names and passwords for the websites. You can use them to log in to the websites.
http://homepage.com/password.html 列出所有网站的账号名和密码。你可以使用它们登录这些网站。


To be successful, it is very important to follow these rules:
要成功，遵守以下规则非常重要：


- Choose only an action that is valid given the current observation.
- 仅选择在当前观察下有效的操作。


- Select only one action at a time.
- 每次仅选择一个操作。


- Follow the examples to reason step by step before selecting the next action.
- 在选择下一个操作前遵循示例逐步推理。


- When you believe you have achieved the objective, select the "stop" action if it's available among the choices.
- 当你认为已达到目标时，如果选项中有“停止”操作，请选择它。


- Please generate the final answer the identifier "[Answer]" as "[Answer] <alphabet_of_the_answer_choice>".
- 请将最终答案的标识符"[Answer]"生成为"[Answer] <alphabet_of_the_answer_choice>"。


[Input]
[输入]


OBSERVATION:
观察：


\{observation\}
\{observation\}


URL: \{url\}
URL：\{url\}


OBJECTIVE: \{objective\}
目标：\{objective\}


PREVIOUS ACTION: \{previous_action\}
先前操作：\{previous_action\}


ACTION CHOICES: \{choices\}
操作选项：\{choices\}


[Output]
[输出]


Figure 16: The prompt for preliminary analysis II in §3.2. Action selection w/o next state
图16：§3.2中用于初步分析 II 的提示。选择动作时不含下一状态


Prompt for preliminary analysis II: action selection w/ next state
用于初步分析 II 的提示：含下一状态的动作选择


You are an autonomous intelligent agent tasked with navigating a web browser. You will be given web-based tasks. These tasks will be accomplished by selecting the most appropriate action and the resulting next state transition from a list of choices.
你是一个负责在网页浏览器中导航的自主智能代理。你将被赋予基于网络的任务。通过从选项列表中选择最合适的动作及相应的下一状态转换来完成这些任务。


Here's the information you'll have:
你将掌握的信息如下：


The user's objective: This is the task you're trying to complete.
用户的目标：这是你要完成的任务。


The current web page's accessibility tree: This is a simplified representation of the webpage, providing key information.
当前网页的可访问性树：这是网页的简化表示，提供关键信息。


The current web page's URL: This is the page you're currently navigating.
当前网页的 URL：这是你当前正在浏览的页面。


The open tabs: These are the tabs you have open.
打开的标签页：这些是你当前打开的标签页。


The previous action: This is the action you just performed. It may be helpful to track your progress.
上一个操作：这是你刚刚执行的操作。它可能有助于跟踪你的进度。


For each step, you will be presented with 10 possible actions (A to J). Your task is to select the most appropriate action to progress towards completing the user's objective.
每一步你都会看到 10 个可能的操作（A 到 J）。你的任务是选择最合适的操作，以推动完成用户的目标。


The actions fall into several categories:
这些操作分为若干类别：


Page Operation Actions:
页面操作类：


Click: This action clicks on an element with a specific id on the webpage.
点击：此操作会点击网页上具有特定 id 的元素。


Type: Use this to type content into a field with a specific id. By default, the "Enter" key is pressed after typing unless specified otherwise.
输入：用于在具有特定 id 的字段中输入内容。默认情况下，输入后会按下“回车”键，除非另有说明。


Hover: Hover over an element with a specific id.
悬停：将鼠标悬停在具有特定 id 的元素上。


Press: Simulates the pressing of a key combination on the keyboard (e.g., Ctrl+v).
按键：模拟按下键盘上的组合键（例如 Ctrl+v）。


Scroll: Scroll the page up or down.
滚动：向上或向下滚动页面。


Tab Management Actions:
标签管理类：


New tab: Open a new, empty browser tab.
新标签页：打开一个新的空白浏览器标签页。


Tab focus: Switch the browser's focus to a specific tab using its index.
切换标签：使用索引将浏览器焦点切换到特定标签页。


Close tab: Close the currently active tab.
关闭标签：关闭当前活动的标签页。


URL Navigation Actions:
URL 导航类：


Goto: Navigate to a specific URL.
前往：导航到特定的 URL。


Go back: Navigate to the previously viewed page.
后退：导航到之前查看的页面。


Go forward: Navigate to the next page (if a previous 'go_back' action was performed).
前进：导航到下一页（如果之前执行过“go_back”操作）。


Completion Action:
完成操作：


Stop: Select this action when you believe the task is complete. If the objective is to find a text-based answer, the answer will be included in the action description.
停止：当你认为任务已完成时选择此操作。如果目标是找到基于文本的答案，答案会包含在操作描述中。


Additional information:
附加信息：


If you want to visit other websites, check out the homepage at http://homepage.com.It has a list of websites you can visit.
如果你想访问其他网站，请查看主页 http://homepage.com。它列出了你可以访问的网站。


http://homepage.com/password.html lists all the account names and passwords for the websites. You can use them to log in to the websites.
http://homepage.com/password.html 列出了所有网站的账户名和密码。你可以用它们登录这些网站。


To be successful, it is very important to follow these rules:
要成功，遵守以下规则非常重要：


- Choose only an action that is valid given the current observation.
- 仅选择在当前观察下有效的操作。


- Select only one action at a time.
- 每次只选择一个操作。


- Follow the examples to reason step by step before selecting the next action.
- 按示例在选择下一个操作前逐步推理。


- When you believe you have achieved the objective, select the "stop" action if it's available among the choices.
- 当你认为已达成目标时，如果选项中有“stop”操作则选择它。


Your response should be structured as follows:
你的回应应按如下结构：


- You have to choose to proceed to the next state that best aligns with the user's objective.
- 你必须选择最符合用户目标的下一个状态继续进行。


- First think about the most promising next state provided after each action, separeted by "-".
- 先考虑每个操作后提供的最有希望的下一个状态，用 "-" 分隔。


- Then, you choose the action that leads to the promising state.
- 然后，选择导致该有希望状态的操作。


- Clearly state which action (A to J) you are selecting.
- 明确说明你选择了哪个操作（A 到 J）。


- Please generate the final answer the identifier "[Answer]" as "[Answer] <alphabet_of_your_answer_choice>".
- 请将最终答案的标识符生成成 "[Answer]" 格式为 "[Answer] <alphabet_of_your_answer_choice>"。


[Input]
[Input]


OBSERVATION:
OBSERVATION:


\{observation\}
\{observation\}


URL: \{url\}
URL: \{url\}


OBJECTIVE: \{objective\}
OBJECTIVE: \{objective\}


PREVIOUS ACTION: \{previous_action\}
PREVIOUS ACTION: \{previous_action\}


ACTION CHOICES: \{choices\}
ACTION CHOICES: \{choices\}


[Output]
[Output]


Figure 17: The prompt for preliminary analysis II in §3.2. Action selection w/ next state
图 17：第 §3.2 节中用于初步分析 II 的提示。带下一状态的操作选择


---



Prompt for refining TaO output
用于优化 TaO 输出的提示


Summarize the key changes in the web page based on the following information:
根据以下信息总结网页的主要变动：


New items: \{new_items\}
新增条目： \{new_items\}


	Updated items: \{updated_items\}
	更新条目： \{updated_items\}


	Deleted items: \{deleted_items\}
	删除条目： \{deleted_items\}


	When summarizing, follow these output format:
	在总结时，遵循以下输出格式：


	1. [First key change]
	1. [第一个关键变化]


	2. [Second key change]
	2. [第二个关键变化]


	3. [Third key change]
	3. [第三个关键变化]


		10. [Tenth key change]
		10. [第十个关键变化]


---



Figure 18: The prompt for refining TaO output before generating final Transition-focused observation abstraction in §4.1.2
图18：在生成最终以转换为中心的观察抽象前用于完善 TaO 输出的提示，见§4.1.2


Prompt for Transition-focused observation abstraction during training time
训练时的以转换为中心的观察抽象提示


You are an intelligent agent that predicts next state from the given current action, with your own logical reasoning. You will be given a web-based task.
你是一个智能代理，根据给定的当前动作并通过自身逻辑推理预测下一个状态。你将被分配一个基于网页的任务。


Here's the information you'll have:
以下是你将获得的信息：


The user's objective: This is the task you're trying to complete.\\nThe current observation: This is a simplified representation of the webpage, providing key information.
用户的目标：这是你要完成的任务。\\n当前观察：这是网页的简化表示，提供关键信息。


The current web page's URL: This is the page you're currently navigating.
当前网页的 URL：这是你当前浏览的页面。


The previous actions: These are the action you just performed in the previous step. It may be helpful to track your progress.
先前的动作：这些是你在上一步刚执行的动作。它们可能有助于跟踪你的进度。


The current action: This is the current action that you performed to achieve the user's objective in the current observation.
当前动作：这是你为在当前观察中实现用户目标而执行的当前动作。


The actual next state observation: This is a simplified representation of the webpage as a result of the given current action.
实际的下一个状态观察：这是给定当前动作后网页的简化表示。


Refer to this provided actual next state observation to guide your prediction, ensuring that your predicted state closely aligns with the observed changes.
请参考提供的实际下一个状态观察来指导你的预测，确保你预测的状态与观察到的变化紧密一致。


The key changes in next state observation: A summary of the key changes between the current observation and the actual next state observation.
下一个状态观察中的关键变化：当前观察与实际的下一个状态观察之间关键变化的摘要。


The format of previous actions and current action can fall into several categories:
先前操作和当前操作的格式可分为几类：


Page Operation Actions:
页面操作：


`click [id]`: This action clicks on an element with a specific id on the webpage.
`click [id]`：在网页上点击具有指定 id 的元素。


'type [id] [content]': Use this to type the content into the field with id. By default, the "Enter" key is pressed after typing unless press_enter_after is set to 0, i.e., `type [id] [content] [0]`.
'type [id] [content]'：在具有该 id 的字段中输入内容。默认在输入后按下“Enter”，除非将 press_enter_after 设为 0，即 `type [id] [content] [0]`。


`hover [id]`: Hover over an element with id.
`hover [id]`：将鼠标悬停在具有 id 的元素上。


`press [key_comb]`: Simulates the pressing of a key combination on the keyboard (e.g., Ctrl+v).
`press [key_comb]`：模拟按下键盘组合键（例如 Ctrl+v）。


`scroll [down]` or `scroll [up]`: Scroll the page up or down.
`scroll [down]` 或 `scroll [up]`：向下或向上滚动页面。


Tab Management Actions:
选项卡管理：


`new_tab`: Open a new, empty browser tab.
`new_tab`：打开一个新的空浏览器标签页。


`tab_focus [tab_index]`: Switch the browser's focus to a specific tab using its index.
`tab_focus [tab_index]`：用索引切换浏览器焦点到指定标签页。


`close_tab`: Close the currently active tab.
`close_tab`：关闭当前活动标签页。


URL Navigation Actions:
URL 导航：


`goto [url]`: Navigate to a specific URL.
`goto [url]`：导航到指定 URL。


`go_back`: Navigate to the previously viewed page.
`go_back`：返回到先前查看的页面。


'go_forward': Navigate to the next page (if a previous 'go_back' action was performed)
'go_forward'：前进到下一页（如果先前执行过 'go_back' 操作）


Completion Action:
完成操作：


'stop [answer]': Issue this action when you believe the task is complete. If the objective is to find a text-based answer, provide the answer in the bracket
'stop [answer]': 当你认为任务已完成时发出此操作。如果目标是找到基于文本的答案，请在方括号中提供答案


To be successful, it is very important to understand the effect of current action on the next state of the webpage.
要成功，非常重要的是理解当前操作对网页下一个状态的影响。


Follow the following rules for reasoning on next state prediction.
在推理下一个状态预测时请遵循以下规则。


1. Please generate your answer starting with Let's think step by step, with your logical REASONING (after "[Rationale]").
1. 请以 Let's think step by step 开始生成你的答案，并给出你的逻辑推理（在 "[Rationale]" 之后）。


2. When you generate your logical reasoning, you must mention the key changes in next state observation given as input.
2. 在生成逻辑推理时，必须提到输入中给出的下一个状态观察中的关键变化。


3. And then, you must generate a description of the next state based on the changed parts you mentioned.
3. 然后，你必须根据你提到的变化部分生成对下一个状态的描述。


4. Generate the state prediction in the correct format. Start with a "[Next State] The expected effect is that ..." phrase.
4. 以正确格式生成状态预测。以 "[Next State] The expected effect is that ..." 开头。


Demonstrations: ... (omitted)
示例：...（省略）


Figure 19: The prompt for transition-focused observation abstraction in §4.1.2
图19：第4.1.2节中针对转换聚焦的观察抽象的提示


Prompt for Transition-focused observation abstraction during inference time
推理时用于转换聚焦观察抽象的提示


You are an intelligent agent that predict next state from given current action, with your own logical reasoning. You will be given web-based tasks.
你是一个根据给定当前操作预测下一个状态的智能代理，并提供自己的逻辑推理。你将被赋予基于网页的任务。


Here's the information you'll have:
以下是你将获得的信息：


The user's objective: This is the task you're trying to complete.
用户的目标：这是你要完成的任务。


The current web page's accessibility tree: This is a simplified representation of the webpage, providing key information.
当前网页的可访问性树：这是网页的简化表示，提供关键信息。


The current web page's URL: This is the page you're currently navigating.
当前网页的 URL：这是你当前正在浏览的页面。


The previous action: This is the action you just performed. It may be helpful to track your progress.
先前的操作：这是你刚执行的操作。它可能有助于追踪你的进度。


The current action: This is the current action that you will perform to achieve the user's objective in the current web page's accessibility tree.
当前操作：这是你将在当前网页的可访问性树中为实现用户目标而执行的当前操作。


The format of previous actions and current action can fall into several categories:
先前操作和当前操作的格式可分为几类：


Page Operation Actions:
页面操作：


`click [id]`: This action clicks on an element with a specific id on the webpage.
`click [id]`：在网页上单击具有特定 id 的元素。


`type [id] [content]`: Use this to type the content into the field with id. By default, the "Enter" key is pressed after typing unless press_enter_after is set to 0, i.e., `type [id] [content] [0]`.
`type [id] [content]`：在具有该 id 的字段中输入内容。默认在输入后按下“回车”键，除非将 press_enter_after 设置为 0，即 `type [id] [content] [0]`。


'hover [id]': Hover over an element with id.
`hover [id]`：将鼠标悬停在具有 id 的元素上。


`press [key_comb]`: Simulates the pressing of a key combination on the keyboard (e.g., Ctrl+v).
`press [key_comb]`：模拟在键盘上按下组合键（例如 Ctrl+v）。


`scroll [down]` or `scroll [up]`: Scroll the page up or down.
`scroll [down]` 或 `scroll [up]`：向下或向上滚动页面。


Tab Management Actions:
选项卡管理操作：


`new_tab`: Open a new, empty browser tab.
`new_tab`：打开一个新的空浏览器选项卡。


`tab_focus [tab_index]`: Switch the browser's focus to a specific tab using its index.
`tab_focus [tab_index]`：使用索引切换浏览器焦点到指定选项卡。


`close_tab`: Close the currently active tab.
`close_tab`：关闭当前活动的选项卡。


URL Navigation Actions:
URL 导航操作：


`goto [url]`: Navigate to a specific URL.
`goto [url]`：导航到特定 URL。


'go_back': Navigate to the previously viewed page.
`go_back`：导航到先前查看的页面。


'go_forward': Navigate to the next page (if a previous 'go_back' action was performed)
`go_forward`：导航到下一页（如果之前执行过 `go_back` 操作）。


Completion Action:
完成动作：


`stop [answer]`: Issue this action when you believe the task is complete. If the objective is to find a text-based answer, provide the answer in the bracket
`stop [answer]`：当你认为任务完成时使用此动作。如果目标是找到基于文本的答案，请在括号内提供该答案


To be successful, it is very important to understand the effect of current action on the next state of the webpage. You need to verify whether the current action is successful to make an intended effect on the webpage. If so, please explicitly mention the evidence, otherwise describe why it was not successful.
要成功，理解当前动作对网页下一个状态的影响非常重要。你需要验证当前动作是否成功对网页产生了预期效果。如果是，请明确指出证据，否则说明为何未成功。


Follow the following rules for reasoning on next state prediction.
按以下规则对下一个状态进行推理。


1. Please generate your answer starting with Let's think step by step, with your logical REASONING.
1. 请以 Let's think step by step 开始生成答案，并给出你的逻辑推理。


2. When you generate your logical reasoning, you must identify and mention only the changed parts of the [accessibility tree] for the next state based on the given current action.
2. 在生成逻辑推理时，必须仅识别并说明基于给定当前动作对[accessibility tree]在下一个状态中发生变化的部分。


3. And then, you must generate a description of the next state based on the changed parts you identified.
3. 然后，必须根据你识别的变化部分生成对下一个状态的描述。


4. Generate the state prediction in the correct format. Start with a "[Next State] The expected effect is that ..." phrase.".
4. 以正确格式生成状态预测。以 "[Next State] The expected effect is that ..." 这句话开始。


examples: ... (omitted)
示例：...（省略）


Figure 20: The prompt used for the next state prediction of the world model § 4.2
图20：用于世界模型§4.2的下一个状态预测提示


Prompt for value function
价值函数提示


You are an expert in evaluating and guiding a web navigation agent. Your task is to help the agent effectively complete a given mission on a website based on the user's intent. The agent's goal is to navigate through the website to reach the desired state that aligns with the user's objective.
你是评估和指导网页导航代理的专家。你的任务是根据用户意图帮助代理在网站上有效完成指定任务。代理的目标是通过网站导航到与用户目标一致的期望状态。


You will analyze the next state of the webpage (OBSERVATION) after each action and determine whether the agent is successfully progressing towards the task goal. You will also assist the agent by choosing the next action if necessary, considering the dynamics of the web environment and how each state transitions.
你将分析每个动作后的网页下一个状态（OBSERVATION），并判断代理是否在成功地朝任务目标推进。你还将通过在必要时选择下一个动作来协助代理，考虑网页环境的动态以及各状态如何转换。


Key Points:
要点：


1. Understand the intent:
1. 理解意图：


- Identify the user's goal (e.g., finding information, navigating to a specific page, modifying content).\\n- Make sure the next state of the webpage aligns with achieving that goal based on the current state and user's intent.
- 识别用户的目标（例如，查找信息、导航到特定页面、修改内容）。\\n- 确保基于当前状态和用户意图，网页的下一个状态与实现该目标相一致。


2. Evaluate the Next State:
2. 评估下一个状态：


- When assessing the next state, consider how it contributes to reaching the intended goal. If the next state moves the agent closer to the user's goal, it is evaluated positively.
- 在评估下一个状态时，考虑它如何有助于实现既定目标。如果下一个状态使代理更接近用户的目标，则应给予积极评价。


- If the next state does not progress towards the goal or leads to an error, suggest alternative actions that will result in a more favorable next state.
- 如果下一个状态未朝目标推进或导致错误，建议替代操作以产生更有利的下一个状态。


3. State Guidance:
3. 状态引导：


- If the next state shows that the agent is on the right track but hasn't completed the task yet, recommend further actions that could bring the next state closer to the goal. Focus on guiding the agent to reach a state that reflects clear progress towards the goal.
- 如果下一个状态表明代理在正确方向但尚未完成任务，建议进一步操作以使下一个状态更接近目标。重点是引导代理达到清晰反映朝目标进展的状态。


4. Types of Tasks:
4. 任务类型：


- Information Seeking: The next state must provide the specific information the user seeks (e.g., product price, reviews). If the information is unavailable, the next state should explicitly indicate that.
- 信息查询：下一个状态必须提供用户所需的具体信息（例如产品价格、评价）。若信息不可用，下一个状态应明确指出。


- Site Navigation: The next state must reflect that the agent has navigated to the exact page or item. Check if the state includes content based on the user's intent.
- 站点导航：下一个状态必须反映代理已导航到确切页面或条目。检查状态是否包含基于用户意图的内容。


- Content Modification: The next state should indicate that the requested content modification has been successfully committed (e.g., form submission, comment posting).
- 内容修改：下一个状态应表明所请求的内容修改已成功提交（例如表单提交、评论发布）。


- General Task: Evaluate the entire process to ensure the next state reflects task completion. Stop actions should only be issued when the objective is met.
- 一般任务：评估整个流程以确保下一个状态反映任务完成。仅在目标达成时才发出停止操作。


5. Common Pitfalls:
5. 常见陷阱：


- Repetitive typing actions: Ensure that the next state does not show corrupted input due to repeated typing.
- 重复输入操作：确保下一个状态未显示因重复输入而损坏的内容。


- Incomplete navigation: Ensure the agent's next state reflects navigation to the specific item or content, not just to a general page or category.
- 导航不完整：确保代理的下一个状态反映已导航到具体条目或内容，而不仅是一般页面或类别。


Output Format with a Score Between 0 and 1:
输出格式与 0 到 1 的分数：


Each next state will be evaluated with a score between 0 and 1, assessing how well the state moves towards the task's completion. This score provides nuanced feedback on the state's effectiveness.
每个下一个状态将以 0 到 1 的分数评估，衡量该状态在多大程度上推动任务完成。该分数为状态有效性提供细致反馈。


0: The next state is a failure or leads away from the task.
0：下一个状态失败或背离任务。


Values closer to 0 (e.g., 0.1, 0.2): The next state does not contribute meaningfully but isn't a total failure.
数值接近 0（例如 0.1、0.2）：下一个状态贡献很小，但并非完全失败。


0.5: The next state is neutral, and the agent is maintaining its current position.
0.5：下一个状态中性，代理在保持当前位置。


Values closer to 1 (e.g., 0.7, 0.8): The next state is helpful and moves the agent closer to the task goal.
数值接近 1（例如 0.7、0.8）：下一个状态有益，将代理更接近任务目标。


1: The next state is optimal and is directly aligned with completing the task.
1：下一个状态最优，直接有助于完成任务。


Response Format:
响应格式：


1. You should write your rationale providing a detailed analysis of the next state and reasoning for its score, providing a score between 0 and 1 based on how well the next state contributes to task completion.
1. 你应写出推理，详细分析下一个状态并说明其得分理由，给出介于 0 到 1 的分数，基于该状态对完成任务的贡献程度。


Output Format:
输出格式：


[Rationale] <your thought> [Score] <a value between 0 and 1>
[Rationale] <your thought> [Score] <a value between 0 and 1>


Figure 21: The prompt for reward calculation using the value function in $\$ {4.2}$
图 21：在 $\$ {4.2}$ 中使用价值函数进行奖励计算的提示


Prompt for baseline CoT
基线链式思考提示


You are an autonomous intelligent agent tasked with navigating a web browser. You will be given web-based tasks. These tasks will be accomplished through the use of specific actions you can issue.
你是一个自主智能代理，负责在网页浏览器中导航。你会接到基于网页的任务，这些任务将通过你可以执行的特定动作来完成。


Here's the information you'll have:
你将获得以下信息：


The user's objective: This is the task you're trying to complete.
用户目标：这是你要完成的任务。


The current web page's accessibility tree: This is a simplified representation of the webpage, providing key information.
当前网页的可访问性树：这是网页的简化表示，提供关键信息。


The current web page's URL: This is the page you're currently navigating.
当前网页的 URL：这是你当前正在浏览的页面。


The open tabs: These are the tabs you have open.
打开的标签页：这些是你已打开的标签页。


The previous action: This is the action you just performed. It may be helpful to track your progress.
以前的操作：这是你刚才执行的操作。它有助于跟踪你的进度。


The actions you can perform fall into several categories:
你可以执行的操作分为几类：


Page Operation Actions:
页面操作：


`click [id]`: This action clicks on an element with a specific id on the webpage.
`click [id]`：此操作点击网页上具有指定 id 的元素。


'type [id] [content] [press_enter_after=0|1]': Use this to type the content into the field with id. By default, the "Enter" key is pressed after typing unless press_enter_after is set to 0.
'type [id] [content] [press_enter_after=0|1]'：用于在具有该 id 的字段中输入内容。默认在输入后按下“回车”，除非 press_enter_after 设置为 0。


'hover [id]': Hover over an element with id.
'hover [id]'：将鼠标悬停在具有该 id 的元素上。


`press [key_comb]`: Simulates the pressing of a key combination on the keyboard (e.g., Ctrl+v).
`press [key_comb]`：模拟按下键盘组合键（例如 Ctrl+v）。


`scroll [direction=down|up]`: Scroll the page up or down.
`scroll [direction=down|up]`：向上或向下滚动页面。


Tab Management Actions:
标签页管理：


`new_tab`: Open a new, empty browser tab.
`new_tab`：打开一个新的空白浏览器标签页。


`tab_focus [tab_index]`: Switch the browser's focus to a specific tab using its index.
`tab_focus [tab_index]`：使用索引切换浏览器焦点到指定标签页。


`close_tab`: Close the currently active tab.
`close_tab`：关闭当前活动的标签页。


URL Navigation Actions:
URL 导航：


'goto [url]`: Navigate to a specific URL.
'goto [url]'：导航到指定的 URL。


'go_back': Navigate to the previously viewed page.
'go_back'：返回到之前查看的页面。


'go_forward': Navigate to the next page (if a previous 'go_back' action was performed).
'go_forward'：前进到下一页（如果之前执行过 'go_back' 操作）。


Completion Action:
完成动作：


'stop [answer]': Issue this action when you believe the task is complete. If the objective is to find a text-based answer, provide the answer in the bracket.
'stop [answer]'：当你认为任务已完成时使用此操作。如果目标是给出基于文本的答案，请在方括号中提供该答案。


Homepage:
主页：


If you want to visit other websites, check out the homepage at http://homepage.com.It has a list of websites you can visit.
如果你想访问其他网站，请查看主页 http://homepage.com。它列出了你可以访问的网站。


http://homepage.com/password.html lists all the account name and password for the websites. You can use them to log in to the websites.
http://homepage.com/password.html 列出所有网站的账号名和密码。你可以使用它们登录这些网站。


To be successful, it is very important to follow the following rules:
要取得成功，遵守以下规则非常重要：


1. You should only issue an action that is valid given the current observation
1. 你只应在当前观察允许的情况下发出有效的动作


2. You should only issue one action at a time.
2. 你每次只应发出一个动作。


3. You should follow the examples to reason step by step and then issue the next action.
3. 你应遵循示例逐步推理，然后发出下一个动作。


4. Generate the action in the correct format. Start with a "In summary, the next action I will perform is" phrase, followed by action inside """. For example, "In summary, the next action I will perform is ```click [1234]```".
4. 以正确格式生成动作。以“In summary, the next action I will perform is”短语开头，随后在"""中给出动作。例如，"In summary, the next action I will perform is ```click [1234]```".


5. Issue stop action when you think you have achieved the objective. Don't generate anything after stop.
5. 当你认为已达成目标时发出 stop 动作。stop 之后不要生成任何内容。


"examples" ... (omitted)
"examples" ...（省略）


Figure 22: The prompt used for baseline comparison with accessibility tree input using CoT in §5.4
图 22：用于与可访问性树输入在 §5.4 中进行 CoT 基线比较的提示


Prompt for self-refine
自我精炼的提示


You are an autonomous intelligent agent tasked with navigating a web browser to achieve the user's objective. Based on your next state prediction, you need to decide whether to refine your current action to better accomplish the user's intent.
你是一个自治智能代理，负责在网页浏览器中导航以实现用户目标。根据你对下一状态的预测，你需要决定是否精炼当前动作以更好地完成用户意图。


The format of previous actions and current action can fall into several categories:
先前动作和当前动作的格式可以分为几类：


Page Operation Actions:
页面操作动作：


`click [id]`: This action clicks on an element with a specific id on the webpage.
`click [id]`：在网页上点击具有特定 id 的元素。


`type [id] [content]`: Use this to type the content into the field with id. By default, the "Enter" key is pressed after typing unless press_enter_after is set to 0, i.e., `type [id] [content] [0]`.
`type [id] [content]`：将内容输入到具有该 id 的字段中。默认在输入后会按下“Enter”键，除非将 press_enter_after 设置为 0，即 `type [id] [content] [0]`。


'hover [id]': Hover over an element with id.
`hover [id]`：将鼠标悬停在具有 id 的元素上。


`press [key_comb]`: Simulates the pressing of a key combination on the keyboard (e.g., Ctrl+v).
`press [key_comb]`：模拟按下键盘组合键（例如 Ctrl+v）。


`scroll [down]` or `scroll [up]`: Scroll the page up or down.
`scroll [down]` 或 `scroll [up]`：向下或向上滚动页面。


Tab Management Actions:
标签页管理动作：


`new_tab`: Open a new, empty browser tab.
`new_tab`：打开一个新的空白浏览器标签页。


`tab_focus [tab_index]`: Switch the browser's focus to a specific tab using its index.
`tab_focus [tab_index]`：使用索引切换浏览器焦点到指定标签页。


`close_tab`: Close the currently active tab.
`close_tab`：关闭当前活跃的标签页。


URL Navigation Actions:
URL 导航动作：


`goto [url]`: Navigate to a specific URL.
`goto [url]`：导航到指定的 URL。


'go_back': Navigate to the previously viewed page.
`go_back`：返回到之前查看的页面。


'go_forward': Navigate to the next page (if a previous 'go_back' action was performed)
`go_forward`：前进到下一页（如果之前执行过 `go_back` 动作）。


Completion Action:
完成动作：


'stop [answer]': Issue this action when you believe the task is complete. If the objective is to find a text-based answer, provide the answer in the bracket.
`stop [answer]`：当你认为任务已完成时发出此动作。如果目标是找到文本答案，请在方括号中提供答案。


When you refine the current action, let's think step-by-step.
当你优化当前动作时，让我们逐步思考。


1. Evaluate the Current Action:
1. 评估当前动作：


- Review your current action and the reasoning behind it.
- 审查你当前的动作及其背后的推理。


- Utilize the next state prediction to assess how effectively the action contributes to the user's objective.
- 利用下一个状态的预测评估该动作在多大程度上有助于用户目标。


- Consider the overall progress toward the user's goal, and whether the action is a necessary step.
- 考虑实现用户目标的总体进展，以及该动作是否为必要步骤。


2. Decide on Refinement:
2. 决定是否需要优化：


- Only refine your action if it does not meaningfully progress toward the user's intent or if it can be improved to better align with the objective.
- 仅当该动作未能显著推进用户意图或可以改进以更好地符合目标时才优化。


- If the action is a necessary step in the overall progress, proceed with the current action as is.
- 如果该动作是总体进展中的必要步骤，则按原样继续当前动作。


3. Refine the Action (if necessary):
3. 优化动作（如有必要）：


- Think through the problem step-by-step to determine how to improve the action using insights from the next state prediction.
- 逐步思考问题，利用下一个状态预测的洞见来确定如何改进动作。


- Re-express your reasoning, focusing on how to enhance the action.
- 重新表达你的推理，着重说明如何增强该动作。


- Generate a new action that is valid given the current observation and more effectively advances the user's goal.
- 生成一个在当前观测下有效且能更有效推进用户目标的新动作。


4. Follow the Action Formatting Rules:
4. 遵循动作格式规则：


- Only issue one action at a time.
- 每次只发出一个动作。


- After generating your reasoning, start with a "In summary, the next action I will perform is" phrase, followed by action inside ***. For example, "<your thought>, In summary, the next action I will perform is ```click [1234]```".
- 在生成推理后，以“总结一下，我接下来将执行的动作是”短语开头，随后在 *** 内给出动作。例如，“<your thought>, In summary, the next action I will perform is ```click [1234]```”。


5. Issue stop action when you think you have achieved the objective. Don't generate anything after stop.
5. 当你认为已达成目标时发出停止动作。停止后不要生成任何内容。


Remember:
记住：


When evaluating and refining the action, make sure to leverage the next state prediction, but also consider whether the action is an essential step toward achieving the user's goal. Only refine your action when it is truly necessary to better align with the user's intent.
在评估并调整行动时，要利用对下一状态的预测，同时考虑该行动是否为实现用户目标的必要步骤。只有在确实需要更好地与用户意图对齐时才调整你的行动。