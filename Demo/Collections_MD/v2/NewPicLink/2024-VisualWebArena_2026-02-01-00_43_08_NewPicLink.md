# VisualWebArena: Evaluating Multimodal Agents on Realistic Visually Grounded Web Tasks
# VisualWebArena: 在真实视觉驱动网页任务上评估多模态智能体


Jing Yu Koh Robert Lo* Lawrence Jang* Vikram Duvvur*
Jing Yu Koh Robert Lo* Lawrence Jang* Vikram Duvvur*


Ming Chong Lim* Po-Yu Huang* Graham Neubig Shuyan Zhou
Ming Chong Lim* Po-Yu Huang* Graham Neubig Shuyan Zhou


Ruslan Salakhutdinov Daniel Fried
Ruslan Salakhutdinov Daniel Fried


Carnegie Mellon University
卡内基梅隆大学


\{jingyuk,rsalakhu,dfried\}@cs.cmu.edu
\{jingyuk,rsalakhu,dfried\}@cs.cmu.edu


## Abstract
## 摘要


Autonomous agents capable of planning, reasoning, and executing actions on the web offer a promising avenue for automating computer tasks. However, the majority of existing benchmarks primarily focus on text-based agents, neglecting many natural tasks that require visual information to effectively solve. Given that most computer interfaces cater to human perception, visual information often augments textual data in ways that text-only models struggle to harness effectively. To bridge this gap, we introduce VisualWebArena, a benchmark designed to assess the performance of multimodal agents on realistic visually grounded web tasks. VisualWebArena comprises of diverse and complex web-based tasks that evaluate various capabilities of autonomous multimodal agents. To perform well, agents need to accurately process image-text inputs, interpret natural language instructions, and execute actions on websites to accomplish user-defined objectives. We evaluate state-of-the-art LLM-based autonomous agents, including several multimodal agents. Our analysis reveals several limitations of text-based LLM agents, gaps in the capabilities of state-of-the-art multimodal language agents, and insights towards building stronger autonomous agents for the web.
能够在网页上进行规划、推理和执行操作的自主智能体为自动化计算机任务提供了一条极具前景的途径。然而，现有的大多数基准测试主要关注基于文本的智能体，忽略了许多需要视觉信息才能有效解决的自然任务。鉴于大多数计算机界面都迎合人类感知，视觉信息通常以纯文本模型难以有效利用的方式增强文本数据。为了弥补这一差距，我们推出了 VisualWebArena，这是一个旨在评估多模态智能体在真实的视觉驱动网页任务中表现的基准。VisualWebArena 包含多种复杂的基于网页的任务，用以评估自主多模态智能体的各种能力。为了表现出色，智能体需要准确处理图文输入，理解自然语言指令，并在网站上执行操作以完成用户定义的目标。我们评估了最先进的基于大语言模型（LLM）的自主智能体，包括几种多模态智能体。我们的分析揭示了基于文本的 LLM 智能体的若干局限性、最先进的多模态语言智能体在能力上的差距，以及构建更强大的网页自主智能体的见解。


## 1 Introduction
## 1 引言


Automating routine computer tasks with autonomous agents is a long standing goal of artificial intelligence research (Franklin and Graesser, 1996; Jennings et al., 1998). To achieve this, we need agents that can navigate computers effectively, process visual and textual inputs, handle high-level natural language instructions, and execute actions to achieve desired goals. As digital interfaces today are primarily built for human eyes, effective visual understanding is necessary for many routine computer tasks. For example, humans frequently perform tasks on the web which involve visual references, such as "Help me order a green polo shirt from Amazon," or rely on pictures rather than text to communicate. However, many agent benchmarks today focus on text-based tasks, neglecting the evaluation (and consequently the development) of multimodal agents. To address this gap, we propose VisualWebArena (Fig. 1), a benchmark suite designed to rigorously assess and advance the visual and textual capabilities of autonomous agents. VisualWebArena builds off the WebArena (Zhou et al., 2024) framework, leveraging reproducible self-hosted environments and execution-based evaluations. VisualWebArena introduces a set of unique tasks that emphasize integrating visual understanding with language processing, closely simulating human interaction with modern computing interfaces. Our contributions are summarized as follows:
利用自主智能体自动化常规计算机任务是人工智能研究的长期目标（Franklin 和 Graesser，1996；Jennings 等，1998）。为实现这一目标，我们需要能够有效导航计算机、处理视觉和文本输入、处理高级自然语言指令并执行操作以达成预期目标的智能体。由于当今的数字界面主要是为人类视觉构建的，有效的视觉理解对于许多常规计算机任务必不可少。例如，人类经常在网页上执行涉及视觉参考的任务，如“帮我从亚马逊订购一件绿色 Polo 衫”，或者依赖图片而非文本进行交流。然而，当今许多智能体基准测试侧重于基于文本的任务，忽略了对多模态智能体的评估（以及随之而来的开发）。为了填补这一空白，我们提出了 VisualWebArena（图 1），这是一套旨在严格评估和提升智能体视觉与文本能力的基准套件。VisualWebArena 基于 WebArena（Zhou 等，2024）框架构建，利用可复现的自托管环境和基于执行的评估。VisualWebArena 引入了一系列独特任务，强调视觉理解与语言处理的整合，密切模拟人类与现代计算界面的交互。我们的贡献总结如下：


- We introduce VisualWebArena, a set of 910 realistic tasks over three diverse web environments: Classifieds, Shopping, and Reddit. The Classifieds environment is a new contribution with real world data, while the Shopping and Reddit environments are inherited from WebArena. All tasks we introduce are visually grounded, and require visual understanding of webpage content to effectively solve (while WebArena does not). 25.2% of the tasks also include images as input (Fig. 1), and require understanding interleaved image-text inputs.
- 我们推出了 VisualWebArena，这是一套包含三个不同网页环境（分类广告、购物和 Reddit）的 910 个真实任务。分类广告环境是包含真实世界数据的新贡献，而购物和 Reddit 环境继承自 WebArena。我们引入的所有任务都是视觉驱动的，需要对网页内容进行视觉理解才能有效解决（而 WebArena 则不需要）。25.2% 的任务还包含图像作为输入（图 1），需要理解交错的图文输入。


- We extensively benchmark the autonomous capabilities of state-of-the-art (SOTA) large language models (LLM) and vision-language models (VLMs), demonstrating that strong VLMs outperform text-based LLMs. The best VLM agents achieve a success rate of 16.4% on VisualWebArena, which is still significantly below human performance of 88.7%.
- 我们对最先进（SOTA）的大语言模型（LLM）和视觉语言模型（VLM）的自主能力进行了广泛的基准测试，证明了强大的 VLM 优于基于文本的 LLM。表现最好的 VLM 智能体在 VisualWebArena 上的成功率为 16.4%，仍远低于 88.7% 的人类表现。


---



*Equal contribution.
*同等贡献。


---



<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/02/2026_02_18__00_19_37_cc0236.jpg"/>



Figure 1: VisualWebArena is a benchmark suite of 910 realistic, visually grounded tasks on self-hosted web environments that involve web navigation and visual understanding.
图 1：VisualWebArena 是一个由 910 个真实的、视觉驱动的任务组成的基准套件，这些任务在自托管的网页环境中进行，涉及网页导航和视觉理解。


- We propose a new VLM agent inspired by Set-of-Marks prompting (Yang et al., 2023a), simplifying the action space of the model. We show that this model substantially outperforms other baseline LLM agents, especially on sites that are more visually complex.
- 我们受 Set-of-Marks 提示（Yang 等，2023a）启发提出了一种新型 VLM 智能体，简化了模型的动作空间。我们展示了该模型显著优于其他基准 LLM 智能体，特别是在视觉复杂度较高的网站上。


## 2 Related Work
## 2 相关工作


Language-Guided Web Agent Benchmarks The development of reproducible environments for autonomous agents has seen considerable progress in recent years. Earlier efforts introduced reinforcement learning environments (Brockman et al., 2016), and extended into web domains (Shi et al., 2017; Liu et al., 2018). Recent web agent benchmarks introduced tasks involving actions on static internet pages (Deng et al., 2023) as well as interaction in simulated web environments (Yao et al., 2022; Zhou et al., 2024). AgentBench (Liu et al., 2023c) extends the scope of agents for computer interaction beyond the web, exploring database management and operating system functionalities.
语言引导的网络智能体基准测试 近年来，为自主智能体创建可复现环境的工作取得了显著进展。早期的工作引入了强化学习环境（Brockman 等人，2016 年），并将其拓展到网络领域（Shi 等人，2017 年；Liu 等人，2018 年）。近期的网络智能体基准测试引入了涉及在静态网页上执行操作的任务（Deng 等人，2023 年），以及在模拟网络环境中的交互任务（Yao 等人，2022 年；Zhou 等人，2024 年）。AgentBench（Liu 等人，2023c）将用于计算机交互的智能体的范围拓展到了网络之外，探索数据库管理和操作系统功能。


LLM Agents There has been significant recent interest in using Large Language Models (LLMs) for developing autonomous agents (Xi et al., 2023; Wang et al., 2023a). State-of-the-art LLMs (Google, 2023; OpenAI, 2023; Chowdhery et al., 2023; Rae et al., 2021; Zhang et al., 2022; Touvron et al., 2023a,b; Jiang et al., 2023, 2024) based on the Transformer (Vaswani et al., 2017) architecture have demonstrated impressive abilities in learning from in-context examples (Brown et al., 2020; Chan et al., 2022), reasoning (Wei et al., 2022; Yao et al., 2023; Wang et al., 2023c; Besta et al., 2023), following instructions (Chung et al., 2022; Longpre et al., 2023; Ouyang et al., 2022), and operating over long-context sequences (Tay et al., 2021; Bertsch et al., 2023; Tworkowski et al., 2023). Several recent works leverage these abilities for building autonomous web agents: Kim et al. (2023) propose a recursive prompting method to improve GPT-4 performance on MiniWoB++ (Liu et al., 2018). Liu et al. (2023d) propose a method of orchestrating multiple LLM agents to improve performance on the WebShop (Yao et al., 2022) environment. Zeng et al. (2023) fine-tunes the LLaMA-2 models on interaction trajectories with instructions, improving over baseline agents.
大语言模型智能体 近期，人们对利用大语言模型（LLMs）开发自主智能体产生了浓厚兴趣（Xi 等人，2023 年；Wang 等人，2023a）。基于 Transformer 架构（Vaswani 等人，2017 年）的先进大语言模型（Google，2023 年；OpenAI，2023 年；Chowdhery 等人，2023 年；Rae 等人，2021 年；Zhang 等人，2022 年；Touvron 等人，2023a、b；Jiang 等人，2023 年、2024 年）已在从上下文中的示例学习（Brown 等人，2020 年；Chan 等人，2022 年）、推理（Wei 等人，2022 年；Yao 等人，2023 年；Wang 等人，2023c；Besta 等人，2023 年）、遵循指令（Chung 等人，2022 年；Longpre 等人，2023 年；Ouyang 等人，2022 年）以及处理长上下文序列（Tay 等人，2021 年；Bertsch 等人，2023 年；Tworkowski 等人，2023 年）方面展现出了令人印象深刻的能力。近期有几项研究利用这些能力构建自主网络智能体：Kim 等人（2023 年）提出了一种递归提示方法，以提高 GPT - 4 在 MiniWoB++ 上的性能（Liu 等人，2018 年）。Liu 等人（2023d）提出了一种协调多个大语言模型智能体的方法，以提高在 WebShop 环境中的性能（Yao 等人，2022 年）。Zeng 等人（2023 年）在带有指令的交互轨迹上对 LLaMA - 2 模型进行微调，相较于基线智能体有所改进。


Vision-Language Models Finally, our work builds off advances in vision-language models (VLMs), used for many multimodal tasks such as image captioning (Vinyals et al., 2015), visual question answering (Antol et al., 2015), and other benchmarks (Mialon et al., 2023; Yue et al., 2023; Tong et al., 2024). Frozen (Tsimpoukelli et al., 2021) was one of the first approaches to demonstrate the effectiveness of finetuning a visual encoder to map images into the embedding space of a LLM, introducing compelling few-shot multimodal abilities. Alayrac et al. (2022) introduced cross-attention layers and scaled up model sizes and training data. Wang et al. (2023b) introduced trainable visual expert modules to improve vision-language fusion. Liu et al. (2023b) proposed fine-tuning on images paired with instructions to improve text generation performance on several multimodal tasks. GPT-4V (OpenAI, 2023) introduces visual processing to the GPT-4 family of models. Gemini (Google, 2023) is multimodal from the beginning (in contrast to post-hoc fine-tuned models), and can handle text interleaved with visual and audio inputs. Several recent work have also explored using VLMs to build agents for mobile platforms (Zhan and Zhang, 2023; Chu et al., 2023;
视觉 - 语言模型 最后，我们的工作建立在视觉 - 语言模型（VLM）的进展之上，这些模型用于许多多模态任务，如图像描述（Vinyals 等人，2015）、视觉问答（Antol 等人，2015）和其他基准测试（Mialon 等人，2023；Yue 等人，2023；Tong 等人，2024）。Frozen（Tsimpoukelli 等人，2021）是最早证明微调视觉编码器以将图像映射到大型语言模型（LLM）嵌入空间有效性的方法之一，引入了引人注目的少样本多模态能力。Alayrac 等人（2022）引入了交叉注意力层，并扩大了模型规模和训练数据。Wang 等人（2023b）引入了可训练的视觉专家模块以改进视觉 - 语言融合。Liu 等人（2023b）提出对与指令配对的图像进行微调，以提高多个多模态任务的文本生成性能。GPT - 4V（OpenAI，2023）为 GPT - 4 系列模型引入了视觉处理能力。Gemini（Google，2023）从一开始就是多模态的（与事后微调的模型形成对比），并且可以处理与视觉和音频输入交错的文本。最近的一些工作还探索了使用 VLM 为移动平台构建智能体（Zhan 和 Zhang，2023；Chu 等人，2023;


Yang et al., 2023b) and the web (Gur et al., 2023; Hong et al., 2023). Zheng et al. (2024) is contemporaneous work which performs action grounding to identify appropriate HTML elements for enabling agents to execute actions. In contrast, our proposed SoM agent uses JavaScript to produce a Set-of-Marks (Yang et al., 2023a) for the VLM to directly use as an observation and action space.
杨等人，2023b）以及网络（古尔等人，2023；洪等人，2023）。郑等人（2024）的研究是同期工作，该工作进行动作定位以识别合适的 HTML 元素，使智能体能够执行动作。相比之下，我们提出的 SoM 智能体使用 JavaScript 生成标记集（杨等人，2023a），供视觉语言模型直接用作观察和动作空间。


## 3 VisualWebArena Environment
## 3 VisualWebArena 环境


In order to ensure reproducibility, realism, and determinism, all websites in the VisualWebArena framework are provided as standalone self-hosted web applications. The textual and visual content are acquired from real world counterparts, while the code is based off open-source infrastructure commonly used in real websites. We formally define the environment, observation space, and action space below, but encourage readers to refer to We-bArena (Zhou et al., 2024) for more details.
为了确保可复现性、真实性和确定性，VisualWebArena 框架中的所有网站均作为独立的自托管 Web 应用程序提供。文本和视觉内容获取自真实世界副本，而代码则基于真实网站中常用的开源基础设施。我们在下文正式定义了环境、观测空间和动作空间，但建议读者参考 WebArena (Zhou et al., 2024) 以获取更多细节。


The environment and agent can be modeled as a partially observable Markov decision process (POMDP): $\mathcal{E} = \left( {S,A,\Omega ,T}\right)$ ,where $S$ represents the set of states, $A$ represents the set of actions (Sec. 3.2),and $\Omega$ represents the set of observations (Sec. 3.1). The transition function is defined as $T : S \times  A \rightarrow  S$ ,with deterministic transitions between states conditioned on actions. At each time step $t$ ,the environment is in some state ${s}_{t}$ (e.g.,a particular page),with a partial observation ${o}_{t} \in  \Omega$ . An agent issues an action ${a}_{t} \in  A$ conditioned on ${o}_{t}$ ,which results in a new state ${s}_{t + 1} \in  S$ and a new partial observation ${o}_{t + 1} \in  \Omega$ of the resulting page. The action ${a}_{t}$ may be an action to be executed on the webpage (Tab. 1), or it may simply be a string output for information seeking tasks (Sec. 3.3).
环境和智能体可以建模为部分可观测马尔可夫决策过程 (POMDP)：$\mathcal{E} = \left( {S,A,\Omega ,T}\right)$，其中 $S$ 表示状态集，$A$ 表示动作集（第 3.2 节），$\Omega$ 表示观测集（第 3.1 节）。转移函数定义为 $T : S \times  A \rightarrow  S$，状态之间的转移基于动作且具有确定性。在每个时间步 $t$，环境处于某种状态 ${s}_{t}$（例如特定页面），并具有部分观测 ${o}_{t} \in  \Omega$。智能体根据 ${o}_{t}$ 发布动作 ${a}_{t} \in  A$，从而产生新状态 ${s}_{t + 1} \in  S$ 和结果页面的新部分观测 ${o}_{t + 1} \in  \Omega$。动作 ${a}_{t}$ 可以是在网页上执行的操作（表 1），也可以仅是信息检索任务的字符串输出（第 3.3 节）。


Finally,we define the reward function $R : S \times \; A \rightarrow  \{ 0,1\}$ (Sec. 3.3) to measure the success of a task execution. In VisualWebArena, the reward function returns 1 at the final step if the state transitions align with the expectations of the task objective (i.e., the goal is achieved), and 0 otherwise.
最后，我们定义奖励函数 $R : S \times \; A \rightarrow  \{ 0,1\}$（第 3.3 节）来衡量任务执行的成功程度。在 VisualWebArena 中，如果状态转移符合任务目标的预期（即目标达成），奖励函数在最后一步返回 1，否则返回 0。


### 3.1 Observation Space
### 3.1 观测空间


The observation space $\Omega$ is modeled after a realistic web browsing experience. Observations include the webpage URLs, opened tabs (possibly multiple tabs of different websites), and the webpage content of the focused tab. In 25.2% of tasks, the intent also involves one or more input images (e.g., the first and third tasks in Fig. 1). The webpage content can be represented in several ways:
观测空间 $\Omega$ 模仿真实的网页浏览体验。观测内容包括网页 URL、已打开的标签页（可能涉及不同网站的多个标签页）以及当前焦点标签页的网页内容。在 25.2% 的任务中，意图还涉及一个或多个输入图像（例如图 1 中的第一个和第三个任务）。网页内容可以通过多种方式表示：


<table><tr><td>Action Type $a$</td><td>Description</td></tr><tr><td>click [elem]</td><td>Click on element elem.</td></tr><tr><td>hover [elem]</td><td>Hover on element elem.</td></tr><tr><td>type [elem] [text]</td><td>Type text on element elem.</td></tr><tr><td>press [key_comb]</td><td>Press a key combination.</td></tr><tr><td>new_tab</td><td>Open a new tab.</td></tr><tr><td>tab_focus [index]</td><td>Focus on the i-th tab.</td></tr><tr><td>tab_close</td><td>Close current tab.</td></tr><tr><td>goto [url]</td><td>Open url.</td></tr><tr><td>go_back</td><td>Click the back button.</td></tr><tr><td>go_forward</td><td>Click the forward button.</td></tr><tr><td>scroll [up|down]</td><td>Scroll up or down the page.</td></tr><tr><td>stop [answer]</td><td>End the task with an output.</td></tr></table>
<table><tbody><tr><td>动作类型 $a$</td><td>描述</td></tr><tr><td>点击 [elem]</td><td>点击元素 elem。</td></tr><tr><td>悬停 [elem]</td><td>悬停在元素 elem 上。</td></tr><tr><td>输入 [elem] [text]</td><td>在元素 elem 中输入文本。</td></tr><tr><td>按键 [key_comb]</td><td>按下组合键。</td></tr><tr><td>新建标签页</td><td>打开新标签页。</td></tr><tr><td>聚焦标签页 [index]</td><td>聚焦到第 i 个标签页。</td></tr><tr><td>关闭标签页</td><td>关闭当前标签页。</td></tr><tr><td>跳转 [url]</td><td>打开 url。</td></tr><tr><td>后退</td><td>点击后退按钮。</td></tr><tr><td>前进</td><td>点击前进按钮。</td></tr><tr><td>滚动 [up|down]</td><td>向上或向下滚动页面。</td></tr><tr><td>停止 [answer]</td><td>以输出结果结束任务。</td></tr></tbody></table>


Table 1: Set of possible actions $A$ .
表 1：可能的操作集 $A$ 。


1. Raw web page HTML as a Document Object Model (DOM) tree, used in previous works on autonomous web agents (Shi et al., 2017; Liu et al., 2018; Deng et al., 2023).
1. 原始网页 HTML 作为文档对象模型 (DOM) 树，用于以往关于自主网络智能体的研究 (Shi et al., 2017; Liu et al., 2018; Deng et al., 2023)。


2. The accessibility tree, ${}^{1}$ which provides a structured and simplified representation of the web-page content that is optimized for assistive technologies. This is the primary representation that WebArena (Zhou et al., 2024) uses for its baseline LLM agents.
2. 可访问性树， ${}^{1}$ 它提供了针对辅助技术优化的网页内容的结构化且简化的表示。这是 WebArena (Zhou et al., 2024) 为其基准 LLM 智能体使用的主要表示形式。


3. Web screenshots as RGB arrays, which has demonstrated efficacy in prior work (Gur et al., 2023; Hong et al., 2023; Yan et al., 2023).
3. 网页截图作为 RGB 数组，这在先前的研究中已证明了有效性 (Gur et al., 2023; Hong et al., 2023; Yan et al., 2023)。


4. We introduce a new visual representation inspired by Set-of-Marks (SoM) prompting (Yang et al., 2023a). For every interactable element on the webpage, we label it with a bounding box and an ID (Fig. 2), producing a screenshot for visual agents to reference elements on the page using their unique ID. We provide more details and analysis in Sec. 5.3.
4. 我们引入了一种受标记集 (SoM) 提示 (Yang et al., 2023a) 启发的新视觉表示。对于网页上的每个可交互元素，我们用边界框和 ID（图 2）对其进行标记，生成一张截图供视觉智能体使用其唯一 ID 来引用页面上的元素。我们在第 5.3 节中提供了更多细节和分析。


### 3.2 Action Space
### 3.2 操作空间


The full set of actions $A$ is summarized in Tab. 1. The arguments for action ${a}_{t}$ is the unique element ID from the current observation ${o}_{t}$ . An advantage of this representation (over predicting $\left( {x,y}\right)$ coordinates) is that it allows us to focus on high level reasoning rather than low-level control, as many SOTA VLMs and LLMs were not explicitly trained for referencing elements at such fine granularity. For the agents with accessibility tree representations, the argument is the element ID in the tree. For the SoM representation, we use the unique IDs assigned in the current page (see Fig. 2).
全套操作 $A$ 总结在表 1 中。操作 ${a}_{t}$ 的参数是来自当前观测值 ${o}_{t}$ 的唯一元素 ID。这种表示形式（相比于预测 $\left( {x,y}\right)$ 坐标）的一个优势是它允许我们专注于高层推理而非底层控制，因为许多 SOTA VLM 和 LLM 并未显式针对如此细粒度的元素引用进行训练。对于具有可访问性树表示的智能体，参数是树中的元素 ID。对于 SoM 表示，我们使用当前页面中分配的唯一 ID（见图 2）。


---



${}^{1}$ https://developer.mozilla.org/en-US/docs/ Glossary/Accessibility_tree
${}^{1}$ https://developer.mozilla.org/en-US/docs/ Glossary/Accessibility_tree


---



<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/02/2026_02_18__00_19_37_748e7b.jpg"/>



Figure 2: Set-of-Marks (Yang et al., 2023a) augmented webpage screenshot. Every interactable element is highlighted with a bounding box and a unique ID.
图 2：标记集 (Yang et al., 2023a) 增强的网页截图。每个可交互元素都用边界框和唯一 ID 突出显示。


### 3.3 Evaluation
### 3.3 评估


In order to evaluate performance on VisualWebArena, we introduce new visually grounded evaluation metrics to the functional evaluation paradigm of WebArena. These allow us to comprehensively evaluate the correctness of execution traces on open ended visually grounded tasks. The rewards for each task are hand designed functions using the primitives described below.
为了评估在 VisualWebArena 上的性能，我们在 WebArena 的功能评估范式中引入了新的视觉定位评估指标。这些指标允许我们全面评估开放式视觉定位任务上执行轨迹的正确性。每个任务的奖励是使用下述原语手工设计的函数。


Information Seeking Tasks Information seeking tasks (e.g., the first task in Tab. 2) expect a string output $\widehat{a}$ from the model. We adopt similar reward functions as WebArena for measuring text correctness against a groundtruth output ${a}^{ * }$ :
信息获取任务 信息获取任务（例如表 2 中的第一个任务）期望模型输出字符串 $\widehat{a}$ 。我们采用与 WebArena 类似的奖励函数来衡量文本相对于标准答案输出 ${a}^{ * }$ 的正确性：


- exact_match: This can be defined as ${\mathbf{1}}_{\left\{  \widehat{a} = {a}^{ * }\right\}  }$ . Only outputs that are exactly equal to the groundtruth are given a score of 1 . This is used in tasks where an exact response (e.g., a numerical answer) is expected.
- exact_match：这可以定义为 ${\mathbf{1}}_{\left\{  \widehat{a} = {a}^{ * }\right\}  }$ 。只有与标准答案完全相等的输出才被给予 1 分。这用于期望精确回答（例如数值答案）的任务。


- must_include: This reward function gives a score of 1 if all elements in ${a}^{ * }$ are contained in $\widehat{a}$ and 0 otherwise. For example,if $\widehat{a} =$ "\$1.99,\$2.50,\$10.00" and ${a}^{ * } =$ \{"1.99","2.50","10.00"\},the task is awarded a score of 1 as all expected elements are present in the output. This is primarily used in tasks where we expect an unordered list of outputs, or we expect text output to contain a particular keyword.
- must_include：如果 ${a}^{ * }$ 中的所有元素都包含在 $\widehat{a}$ 中，此奖励函数将给出 1 分，否则给出 0 分。例如，如果 $\widehat{a} =$ 为 "\$1.99,\$2.50,\$10.00" 且 ${a}^{ * } =$ 为 \{"1.99","2.50","10.00"\}，由于输出中包含所有预期元素，该任务将被授予 1 分。这主要用于我们期望无序输出列表，或期望文本输出包含特定关键词的任务。


- fuzzy_match: This function queries a LLM (GPT-4-Turbo in our implementation) to evaluate whether ${a}^{ * }$ and $\widehat{a}$ are semantically equal. The LLM is prompted to output "correct", "incorrect", or "partially correct", and we assign a reward of 1 if the output is "correct". ${}^{2}$ This evaluation is useful for more open ended settings where we are only concerned with semantic rather than exact equivalence, such as asking the user to add a comment describing an image.
- fuzzy_match：此函数查询 LLM（在我们的实现中为 GPT-4-Turbo）以评估 ${a}^{ * }$ 和 $\widehat{a}$ 是否在语义上相等。LLM 被提示输出 "correct"、"incorrect" 或 "partially correct"，如果输出为 "correct"，我们分配 1 的奖励。 ${}^{2}$ 这种评估对于更开放的设置非常有用，在这些设置中我们只关注语义而非精确的等价性，例如要求用户添加描述图像的评论。


- must_exclude: We introduce this function, which is the converse of must_include. A reward of 0 is assigned if any element from a set ${a}^{ * }$ is found in $\widehat{a}$ (and 1 otherwise). For instance,if $\widehat{a} =$ "\$1.99,\$2.50,\$10.00" and ${a}^{ * } =$ \{"1.50","2.00"\},the reward is 1 as none of the prohibited elements are in the output.
- must_exclude：我们引入了此函数，它是 must_include 的反函数。如果在 $\widehat{a}$ 中发现集合 ${a}^{ * }$ 中的任何元素，则奖励为 0（否则为 1）。例如，如果 $\widehat{a} =$ 为 "\$1.99,\$2.50,\$10.00" 且 ${a}^{ * } =$ 为 \{"1.50","2.00"\}，由于输出中不包含任何禁用元素，奖励为 1。


In addition, we also introduce several new visual functions for measuring open ended tasks:
此外，我们还引入了几个用于衡量开放式任务的新视觉函数：


- eval_vqa: Similar to fuzzy_match, this function queries a VLM capable of performing visual question answering (VQA) (Antol et al., 2015). We use BLIP-2-T5XL (Li et al., 2023) in our implementation. We query the VLM with an image and a question. If the output of the VLM contains the groundtruth answer ${a}^{ * }$ ,a reward of 1 is assigned. This is useful for evaluating more open ended tasks, e.g., "Buy me a green hoodie under \$10.". There are many possible products that satisfy this objective, and it would be infeasible to enumerate all their IDs.
- eval_vqa：类似于 fuzzy_match，此函数查询具有视觉问答 (VQA) 能力的 VLM (Antol et al., 2015)。我们在实现中使用 BLIP-2-T5XL (Li et al., 2023)。我们向 VLM 提供图像和问题，如果 VLM 的输出包含标准答案 ${a}^{ * }$，则奖励为 1。这对于评估更具开放性的任务非常有用，例如“帮我买一件 10 美元以下的绿色连帽衫”。有许多产品可以满足这一目标，枚举所有 ID 是不可行的。


---



${}^{2}$ We do not consider non-binary rewards in this work,but this would be a valuable direction to explore in the future towards introducing more continuous scales of performance.
${}^{2}$ 我们在本文中不考虑非二元奖励，但这是未来探索更连续性能指标的一个有价值的方向。


---



- eval_fuzzy_image_match: This function checks whether a query image is similar to a groundtruth image according to the structural similarity index measure (SSIM) (Wang et al., 2004). If the SSIM between the query and groundtruth images is higher than a threshold $t \in  \left\lbrack  {0,1}\right\rbrack$ ,a reward of 1 is assigned.
- eval_fuzzy_image_match：此函数根据结构相似性指数 (SSIM) (Wang et al., 2004) 检查查询图像是否与标准图像相似。如果查询图像与标准图像之间的 SSIM 高于阈值 $t \in  \left\lbrack  {0,1}\right\rbrack$，则奖励为 1。


Navigation and Actions Many tasks in VisualWebArena require navigating through multiple webpages, and executing actions to change the underlying state $s$ of the environment. To accurately evaluate certain objectives, we require reward functions that examine the final webpage state to determine whether the task was successfully accomplished. Each evaluator consists of a locator as well as a URL. The URL can be a specific page, or a function (e.g., the last page that the agent navigated to). The locator describes the object on the page that should be examined (e.g., all img elements, or all elements with the .product-image-photo class). During evaluation, we use the locator to retrieve the corresponding image or text content, and reuse the functions from the information seeking tasks to check for correctness.
导航与操作 VisualWebArena 中的许多任务需要导航多个网页并执行操作以更改环境的底层状态 $s$。为了准确评估某些目标，我们需要通过检查最终网页状态来确定任务是否成功完成的奖励函数。每个评估器由一个定位器和一个 URL 组成。URL 可以是特定页面或函数（例如智能体导航到的最后一个页面）。定位器描述页面上应检查的对象（例如所有 img 元素，或所有带有 .product-image-photo 类的元素）。在评估过程中，我们使用定位器检索相应的图像或文本内容，并重用信息检索任务中的函数来检查正确性。


## 4 Curating Visually Grounded Tasks
## 4 策划视觉对齐任务


### 4.1 Web Environments
### 4.1 网络环境


VisualWebArena is designed around three realistic web environments that involve visually rich content. Several tasks require referencing information from a self-hosted Wikipedia knowledge base, and others involve interacting across more than one website.
VisualWebArena 围绕三个包含丰富视觉内容的真实网络环境设计。一些任务需要参考来自自托管维基百科知识库的信息，而另一些任务则涉及跨多个网站的交互。


Classifieds We introduce a new Classifieds web-site in VisualWebArena, inspired by real world marketplaces such as Craigslist and Facebook Marketplace. The Classifieds site contains 65,955 listings and provides a distinct environment compared to existing ones in WebArena, introducing visually grounded tasks centered around user interactions typical in classifieds websites (posting, searching, commenting). The site's infrastructure uses OS-Class, a robust open-source Content Management System (CMS) designed for classifieds ads, used in multiple real world sites. OSClass enables functions such as search, posting, commenting, and leaving reviews and ratings. More details about the environment are provided in Appendix. D.
分类广告 我们在 VisualWebArena 中引入了一个新的分类广告网站，其灵感来自 Craigslist 和 Facebook Marketplace 等真实市场。该分类广告网站包含 65,955 条列表，与 WebArena 中的现有环境相比提供了一个独特环境，引入了以分类广告网站典型用户交互（发布、搜索、评论）为中心的视觉对齐任务。该站点的基础设施使用 OS-Class，这是一个强大的开源内容管理系统 (CMS)，专为分类广告设计，并用于多个真实网站。OSClass 支持搜索、发布、评论以及留下评价和评分等功能。有关该环境的更多详细信息见附录 D。


Shopping The Shopping site follows the e-commerce environment from WebArena (Zhou et al., 2024), with product information and content scraped from Amazon and released in Web-Shop (Yao et al., 2022). Visual understanding of product images is required for successfully navigating and completing tasks on e-commerce platforms, making this a natural choice for VisualWebArena.
购物 购物网站遵循 WebArena (Zhou et al., 2024) 的电子商务环境，其产品信息和内容从 Amazon 抓取并在 Web-Shop (Yao et al., 2022) 中发布。视觉理解产品图像对于在电子商务平台上成功导航和完成任务至关重要，这使其成为 VisualWebArena 的自然选择。


Reddit The Reddit site also follows the same environment from WebArena, and represents a social forum platform. The site contains 31,464 posts containing a diverse set of images across different subreddits and forums, such as natural images, memes, consumer electronics, and charts.
Reddit Reddit 网站也遵循 WebArena 的相同环境，代表一个社交论坛平台。该站点包含 31,464 个帖子，涵盖不同子版块和论坛中的各种图像，如自然图像、模因、消费电子产品和图表。


### 4.2 Tasks
### 4.2 任务


Task Creation We introduce a set of 910 new tasks, split across the three sites detailed earlier. We focus on curating realistic visually grounded tasks, following a similar process as task creation in WebArena. We start by having 6 graduate students (co-authors of this paper) write intent templates (e.g., "Find me the \{\{attribute\}\}\{\{item\}\}. It should be between \{\{range\}\}."), which can be manually expanded by the annotator to form multiple tasks (e.g., "Find me the cheapest red Toyota. It should be between \$3000 to \$6000."). We encouraged the annotators to be creative, and make use of the visual layouts of the websites, input images, and cross-site functionalities to develop creative and realistic tasks. When tasks include input images, these were sourced from royalty-free, attribution-free sources and MS-COCO (Lin et al., 2014). Annotators also wrote the reward functions using the primitives described in Sec. 3.3. We collected a total of 314 unique templates (average of 2.9 tasks per template). While the majority of tasks can be solved, we also included a small subset (46 tasks, or 5.1%) which are unachievable. This subset tests the ability of agents to terminate early in the event where a task cannot be solved, which is essential in many real world scenarios. For unachievable tasks, we require agents to output a reason why the task is unachievable, which is evaluated using the fuzzy_match function (Sec. 3.3).
任务创建 我们引入了 910 个新任务，分布在前面提到的三个网站中。我们模仿 WebArena 的任务创建过程，专注于策划真实的视觉落地任务。首先，由 6 名研究生（本文共同作者）编写意图模板（例如，“帮我找一件{{attribute}}{{item}}。价格应在{{range}}之间。”），标注者可以手动扩展这些模板以形成多个任务（例如，“帮我找一辆最便宜的红色丰田车。价格应在 3000 到 6000 美元之间。”）。我们鼓励标注者发挥创意，利用网站的视觉布局、输入图像和跨站功能来开发兼具创意与真实性的任务。当任务包含输入图像时，这些图像来源于免版税、免署名的资源及 MS-COCO {{Lin et al., 2014}}。标注者还使用第 3.3 节中描述的基元编写了奖励函数。我们共收集了 314 个唯一模板（平均每个模板产生 2.9 个任务）。虽然大多数任务是可以完成的，但我们也包含了一小部分（46 个，占 5.1%）无法完成的任务。这一子集旨在测试智能体在任务无法解决时提前终止的能力，这在许多现实场景中至关重要。对于无法完成的任务，我们要求智能体输出无法完成的原因，并使用 fuzzy_match 函数进行评估{{Sec. 3.3}}。


Visually Grounded Tasks A key aspect of VisualWebArena is the inherent visual grounding of all tasks. Each task demands visual understanding, requiring agents to process and interpret visual information rather than relying solely on textual or HTML-based cues. This aligns closely with modern human-computer interfaces, where visual information (e.g., icons, colors) is often critical.
视觉落地任务 VisualWebArena 的一个关键特征是所有任务固有的视觉落地属性。每个任务都对视觉理解有要求，需要智能体处理并解释视觉信息，而不是仅仅依赖文本或基于 HTML 的提示。这与现代人机界面高度一致，因为在这些界面中，视觉信息（如图标、颜色）通常至关重要。


<table><tr><td>Webpage / Input Image(s)</td><td>Example Intent</td><td>Reward Function $r\left( {s,a}\right)$ Implementation</td></tr><tr><td><img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/02/2026_02_18__00_19_37_b7bf5c.jpg"/></td><td>Buy the least expensive red blanket from the "Blankets & Throws" category.</td><td>url="func:shopping_get_latest_order_url" <br> must_include( $\widehat{a}$ ,\{ "B0983XCYK6","Red" \})</td></tr><tr><td><img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/02/2026_02_18__00_19_37_584099.jpg"/></td><td>Add something like what the man is wearing to my wish list.</td><td>url="/wishlist" <br> locator(".wishlist .product-image-photo") <br> eval_vqa(s, "Is this a polo shirt? (yes/no)", "yes") <br> eval_vqa(s, "Is this shirt green? (yes/no)", "yes")</td></tr><tr><td><img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/02/2026_02_18__00_19_37_90f264.jpg"/></td><td>Create a post for each of these images in the most related forums.</td><td>eval_fuzzy_image_match $\left( {s,{a}^{ * }}\right)$</td></tr><tr><td><img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/02/2026_02_18__00_19_37_eacbb5.jpg"/></td><td>Navigate to my listing of the white car and change the price to \$25000. Update the price in the description as well.</td><td>url="/index.php?page=item&id=84144" <br> must_include( $\widehat{a}$ ,"\$25000 |OR| \$25,000") <br> must_exclude( $\widehat{a}$ ,"\$30000 |OR| \$30,000")</td></tr></table>
<table><tbody><tr><td>网页 / 输入图像</td><td>示例意图</td><td>奖励函数 $r\left( {s,a}\right)$ 实现</td></tr><tr><td><img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/02/2026_02_18__00_19_37_b7bf5c.jpg"/></td><td>从“毯子和披肩”类别中购买最便宜的红色毯子。</td><td>url="func:shopping_get_latest_order_url" <br/> must_include( $\widehat{a}$ ,\{ "B0983XCYK6","Red" \})</td></tr><tr><td><img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/02/2026_02_18__00_19_37_584099.jpg"/></td><td>将类似那个男人穿的衣服添加到我的愿望清单中。</td><td>url="/wishlist" <br/> locator(".wishlist .product-image-photo") <br/> eval_vqa(s, "Is this a polo shirt? (yes/no)", "yes") <br/> eval_vqa(s, "Is this shirt green? (yes/no)", "yes")</td></tr><tr><td><img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/02/2026_02_18__00_19_37_90f264.jpg"/></td><td>在最相关的论坛中为这些图像分别发布一个帖子。</td><td>eval_fuzzy_image_match $\left( {s,{a}^{ * }}\right)$</td></tr><tr><td><img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/02/2026_02_18__00_19_37_eacbb5.jpg"/></td><td>导航到我的白色汽车列表，并将价格更改为 \$25000。同时更新描述中的价格。</td><td>url="/index.php?page=item&id=84144" <br/> must_include( $\widehat{a}$ ,"\$25000 |OR| \$25,000") <br/> must_exclude( $\widehat{a}$ ,"\$30000 |OR| \$30,000")</td></tr></tbody></table>


Table 2: Various evaluation metrics to assign reward $r\left( {s,a}\right)  \in  R : S \times  A \rightarrow  \{ 0,1\}$ . Our execution-based reward primitives allow us to benchmark many diverse, realistic, and open-ended tasks.
表 2：用于分配奖励的各种评估指标 $r\left( {s,a}\right)  \in  R : S \times  A \rightarrow  \{ 0,1\}$。我们基于执行的奖励原语使我们能够基准测试许多多样、现实且开放式的任务。


For instance, a typical task might involve selecting a visually specific item, such as a "green polo shirt" where the color is visually discernible but not explicitly mentioned in text.
例如，一个典型的任务可能涉及选择一个视觉上特定的物品，比如一件“绿色 Polo 衫”，其颜色在视觉上可辨别，但未在文本中明确提及。


Task Complexity We classify each task into three difficulty levels: easy, medium, and hard. This classification is particularly useful for assessing performance across a spectrum of agents, ranging from smaller models to state-of-the-art LLMs and VLMs. We find in our analysis (Sec. 5) that many open-source models (e.g., LLaMA-2-70B, IDEFICS-80B) achieve a success rate of close to 0 on medium or hard tasks, but non-zero performance on easy tasks. This suggests that running open-source models on the easy subset would provide useful signal during development as well as faster iteration cycles (assuming performance between weaker and stronger agents are correlated). We also provide more detailed analysis in Appendix. C.2.
任务复杂度 我们将每个任务分为三个难度级别：简单、中等和困难。这种分类对于评估从较小模型到最先进 LLM 和 VLM 的一系列智能体的性能特别有用。我们在分析（第 5 节）中发现，许多开源模型（如 LLaMA-2-70B、IDEFICS-80B）在中等或困难任务上的成功率接近 0，但在简单任务上表现非零。这表明，在开发过程中在简单子集上运行开源模型将提供有用的信号以及更快的迭代周期（假设弱智能体和强智能体之间的性能是相关的）。我们还在附录 C.2 中提供了更详细的分析。


We annotate both the action and visual difficulty of each task. The action difficulty is determined by the estimated number of actions that a human would need to complete the task. Easy tasks are defined as those that require three or fewer actions, medium tasks involve four to nine actions, and hard tasks demand ten or more. Visual difficulty is similarly segmented: Easy tasks involve basic visual identification of colors, shapes, and high-level object detection (e.g., recognizing the presence of a cat). Medium tasks require discerning patterns, semantic understanding, or OCR on large text of shorter lengths. Hard tasks involve multiple images, OCR on small or lengthy text, or fine details. Finally, the overall difficulty level is determined by averaging the visual and reasoning complexities. However, human judgment may lead to deviations in this assessment, as certain tasks might inherently skew more towards primarily testing visual or reasoning challenges.
我们对每个任务的操作难度和视觉难度都进行了标注。操作难度由人类完成任务所需的估计操作次数决定。简单任务定义为需要三次或更少操作的任务，中等任务涉及四到九次操作，困难任务则需要十次或更多。视觉难度也进行了类似的划分：简单任务涉及颜色、形状的基本视觉识别以及高层物体检测（例如，识别猫的存在）。中等任务需要辨别模式、语义理解或对较短的大段文本进行 OCR。困难任务涉及多张图像、对微小或冗长文本的 OCR 或精细细节。最后，总体难度级别通过平均视觉和推理复杂度来确定。然而，人类判断可能会导致此评估出现偏差，因为某些任务可能天生更倾向于主要测试视觉或推理挑战。


### 4.3 Human Performance
### 4.3 人类表现


We measure the success rate of 7 college students on VisualWebArena tasks. Several of these students also assisted with task creation, and to avoid data leakage, we ensured that they were not assigned to the same tasks that they initially created. We sample one task per template, collecting a representative set of 230 tasks. We find that humans do well at these tasks, achieving an overall success rate of 88.7% (Tab. 3). The mistakes made in the remaining tasks are usually minor, such as not reading the task correctly or missing a part of the objective. For example, one task asked to add a particular item to the wishlist, but the human added it to the shopping cart instead. Another common failure mode was for tasks that required exhaustive search (e.g., "Find and navigate to the comments of this exact image."). Users were often unable to find the appropriate post after searching for 5-10 minutes and gave up, assuming that the task was unachievable. In many shopping tasks, humans also did not look through all possible candidate pages to identify the cheapest or most highly reviewed product. We found these failure modes interesting, as they represent issues that strong agents would be well poised to handle, potentially achieving above human performance and speed.
我们测量了 7 名大学生在 VisualWebArena 任务上的成功率。其中几名学生还协助了任务创建，为了避免数据泄露，我们确保他们没有被分配到自己最初创建的任务中。我们每个模板抽取一个任务，收集了一组具有代表性的 230 个任务。我们发现人类在这些任务中表现良好，实现了 88.7% 的总体成功率（表 3）。在其余任务中犯的错误通常很细微，例如没有正确阅读任务或遗漏了部分目标。例如，一个任务要求将特定商品添加到愿望单，但人类却将其添加到了购物车。另一种常见的失败模式是需要详尽搜索的任务（例如，“查找并导航到这张特定图片的评论”）。用户在搜索 5-10 分钟后往往无法找到合适的帖子并放弃，认为该任务无法完成。在许多购物任务中，人类也没有浏览所有可能的候选页面来识别最便宜或评价最高的商品。我们发现这些失败模式很有趣，因为它们代表了强大智能体能够很好处理的问题，有可能实现超越人类的表现和速度。


## 5 Baselines
## 5 基准模型


We run several baselines to benchmark the performance of state-of-the-art LLM and VLM agents. All models are prompt-based and provided with 3 in-context examples (one from each environment), which share no overlap with the benchmark tasks. The prompts we use are provided in the appendix. We summarize the results in Tab. 3 and describe the baselines in detail in the following sections.
我们运行了多个基准模型来评估最先进的 LLM 和 VLM 智能体的性能。所有模型均基于提示词，并提供 3 个上下文示例（每个环境一个），这些示例与基准任务没有重叠。我们使用的提示词在附录中提供。我们在表 3 中总结了结果，并在以下章节中详细描述了基准模型。


### 5.1 Text-based LLM Agents
### 5.1 基于文本的 LLM 智能体


Several prior works developed autonomous agents by prompting text-based LLMs (Zhou et al., 2024; Kim et al., 2023; Liu et al., 2023d). We benchmark several text-based LLM agents with Chain-of-Thought prompting (Wei et al., 2022) over the accessibility tree representations of the websites as input, and leave more advanced prompting strategies for future work. We test API-based LLMs, including GPT-4 Turbo (gpt-4-1106-preview), GPT- 3.5 Turbo (gpt-3.5-turbo-1106), and Gemini-Pro, as well as open sourced LLMs (LLaMA-2- 70B, Mixtral-8x7B).
此前的一些工作通过提示基于文本的 LLM 开发了自主智能体 (Zhou et al., 2024; Kim et al., 2023; Liu et al., 2023d)。我们基准测试了几个基于文本的 LLM 智能体，使用思维链提示 (Wei et al., 2022)，以网站的可访问性树表示作为输入，并将更先进的提示策略留给未来的工作。我们测试了基于 API 的 LLM，包括 GPT-4 Turbo (gpt-4-1106-preview)、GPT-3.5 Turbo (gpt-3.5-turbo-1106) 和 Gemini-Pro，以及开源 LLM (LLaMA-2-70B, Mixtral-8x7B)。


### 5.2 Image Caption Augmented LLM Agents
### 5.2 图像字幕增强的 LLM 智能体


VisualWebArena is a visually grounded benchmark, and we expect that leveraging complementary visual information would improve performance. Hence, we run pretrained image captioning models on every img element on the HTML page, and augment the accessibility tree with this information as the image alt-text before passing this as input to the LLM agents. If a task contains input images, we also caption them and include the captions as part of the prompt. We run experiments on GPT-3.5 with two recent image captioning models, BLIP-2-T5XL (Li et al., 2023) and LLaVA-v1.5- 7B (Liu et al., 2023a). Our results with GPT-3.5 as the LLM backbone ("Caption-augmented" section of Tab. 3) suggest that the LLaVA and BLIP-2 captioning models achieve comparable performance. Since BLIP-2 achieves a slightly higher success rate, is a smaller model, and requires less GPU VRAM, we use it as the captioning backbone for the remaining experiments.
VisualWebArena 是一个视觉驱动的基准测试，我们预期利用补充视觉信息能提升性能。因此，我们在 HTML 页面的每个 img 元素上运行预训练图像标注模型，并将该信息作为图像 alt 文本增强可访问性树，再将其作为输入传递给 LLM 智能体。如果任务包含输入图像，我们也会对其进行标注并将标注作为提示词的一部分。我们在 GPT-3.5 上使用两种近期的图像标注模型 BLIP-2-T5XL (Li et al., 2023) 和 LLaVA-v1.5-7B (Liu et al., 2023a) 进行了实验。以 GPT-3.5 作为 LLM 主干的结果（表 3 的“Caption-augmented”部分）表明，LLaVA 和 BLIP-2 标注模型取得了相当的性能。由于 BLIP-2 的成功率略高、模型更小且对 GPU 显存需求更低，我们将其作为剩余实验的标注主干。


### 5.3 Multimodal Agents
### 5.3 多模态智能体


Finally, we benchmark strong API-based and open-source VLMs as agents. We evaluate several models capable of processing multiple interleaved image-and-text inputs: GPT-4V (OpenAI, 2023), Gemini-Pro (Google, 2023), IDEFICS-80B-Instruct (a reimplementation of Flamingo (Alayrac et al., 2022)), and CogVLM (Wang et al., 2023b). We experiment with two settings:
最后，我们将强大的基于 API 和开源的 VLM 作为智能体进行基准测试。我们评估了多个能够处理多张交错图文输入的模型：GPT-4V (OpenAI, 2023)、Gemini-Pro (Google, 2023)、IDEFICS-80B-Instruct（Flamingo (Alayrac et al., 2022) 的重新实现）以及 CogVLM (Wang et al., 2023b)。我们尝试了两种设置：


Image Screenshot + Captions + Accessibility Tree: This approach provides the accessibility tree representation augmented with image captions as accessibility tree alt-text from BLIP-2-T5XL (similar to the caption-augmented agent), as well as the screenshot of the current webpage as inputs. This provides the model with both the structural information and the visual context of the website.
图像截图 + 标注 + 可访问性树：该方法提供由 BLIP-2-T5XL 增强了图像标注（作为可访问性树 alt 文本）的可访问性树表示（类似于标注增强型智能体），以及当前网页的截图作为输入。这为模型提供了网站的结构信息和视觉上下文。


Image Screenshot + Captions + SoM: Inspired by Set-of-Marks prompting (Yang et al., 2023a), we perform an initial preprocessing step by using JavaScript to automatically annotate every inter-actable element on the webpage with a bounding box and a unique ID. The annotated screenshot containing bounding boxes and IDs, are provided as input to the multimodal model along with a text representation of the SoM (see Fig. 2). Similar to the baselines above, we also provide the captions from BLIP-2-T5XL for all img elements on the page. There have been several projects ${}^{3}$ that propose similar representations. Most have been proof-of-concept demos, and to the best of our knowledge, we are the first to systematically benchmark this on a realistic and interactive web environment.
图像截图 + 标注 + SoM：受标记集（SoM）提示词 (Yang et al., 2023a) 的启发，我们执行了一个初始预处理步骤，利用 JavaScript 自动为网页上的每个可交互元素标注边界框和唯一 ID。包含边界框和 ID 的标注截图与 SoM 的文本表示一起提供给多模态模型（见图 2）。与上述基准类似，我们也为页面上的所有 img 元素提供来自 BLIP-2-T5XL 的标注。已有数个项目 ${}^{3}$ 提出了类似的表示方式。大多数项目仅为概念验证演示，据我们所知，我们是首个在真实且交互式的网络环境中对此进行系统基准测试的。


## 6 Results and Analysis
## 6 结果与分析


Our main baseline results are summarized in Tab. 3. All existing models substantially underperform compared to humans, which indicate significant headroom in VisualWebArena for future work. We discuss some main findings below with the GPT-4V model, with further analysis in the appendix.
我们的主要基准测试结果总结在表 3 中。所有现有模型表现均大幅落后于人类，这表明 VisualWebArena 为未来工作留下了巨大的提升空间。我们在下文结合 GPT-4V 模型讨论一些主要发现，并在附录中进行进一步分析。


Text-based LLMs Perform Poorly State-of-the-art text-only LLMs generally achieve poor results, with the best model (GPT-4) achieving an overall success rate of ${7.25}\%$ . When we augment the LLMs with captions, this considerably improves success rate (7.25% to 12.75% for GPT-4).
纯文本 LLM 表现不佳：最先进的纯文本 LLM 通常结果较差，表现最好的模型 (GPT-4) 取得的总体成功率为 ${7.25}\%$。当我们用标注增强 LLM 时，成功率得到了显著提升（GPT-4 从 7.25% 提升至 12.75%）。


Multimodality Helps Using multimodal agents significantly improves the success rate: GPT-4V (gpt-4-1106-vision-preview) achieves an overall success rate of ${15.05}\%$ ,substantially improving over the text-only agents. Gemini-Pro also experiences a significant uplift in success rate, from 3.85% (caption-augmented) to 6.04% (multimodal). Text-based agents may be limited in their ability to process complex images (e.g., those that require OCR or recognition of non-salient objects).
多模态有所帮助：使用多模态智能体显著提升了成功率：GPT-4V (gpt-4-1106-vision-preview) 取得了 ${15.05}\%$ 的总体成功率，较纯文本智能体有大幅提升。Gemini-Pro 的成功率也经历了显著增长，从 3.85%（标注增强）提升至 6.04%（多模态）。纯文本智能体在处理复杂图像（例如需要 OCR 或识别非显著对象的图像）时的能力可能有限。


---



${}^{3}$ GPT-4V-ACT and vimGPT propose similar interfaces.
${}^{3}$ GPT-4V-ACT 和 vimGPT 提出了类似的界面。


---



<table><tr><td rowspan="2">Model Type</td><td rowspan="2">LLM Backbone</td><td rowspan="2">Visual Backbone</td><td rowspan="2">Inputs</td><td colspan="4">Success Rate (↑)</td></tr><tr><td>Classifieds</td><td>Reddit</td><td>Shopping</td><td>Overall</td></tr><tr><td rowspan="5">Text-only</td><td rowspan="5">LLaMA-2-70B Mixtral-8x7B Gemini-Pro GPT-3.5 GPT-4</td><td rowspan="5">-</td><td rowspan="5">Acc. Tree</td><td>0.43%</td><td>1.43%</td><td>1.29%</td><td>1.10%</td></tr><tr><td>1.71%</td><td>2.86%</td><td>1.29%</td><td>1.76%</td></tr><tr><td>0.85%</td><td>0.95%</td><td>3.43%</td><td>2.20%</td></tr><tr><td>0.43%</td><td>0.95%</td><td>3.65%</td><td>2.20%</td></tr><tr><td>5.56%</td><td>4.76%</td><td>9.23%</td><td>7.25%</td></tr><tr><td rowspan="6">Caption-augmented</td><td>LLaMA-2-70B</td><td>BLIP-2-T5XL</td><td rowspan="6">Acc. Tree + Caps</td><td>0.00%</td><td>0.95%</td><td>0.86%</td><td>0.66%</td></tr><tr><td>Mixtral-8x7B</td><td>BLIP-2-T5XL</td><td>1.28%</td><td>0.48%</td><td>2.79%</td><td>1.87%</td></tr><tr><td>GPT-3.5</td><td>LLaVA-7B</td><td>1.28%</td><td>1.43%</td><td>4.08%</td><td>2.75%</td></tr><tr><td>GPT-3.5</td><td>BLIP-2-T5XL</td><td>0.85%</td><td>1.43%</td><td>4.72%</td><td>2.97%</td></tr><tr><td>Gemini-Pro</td><td>BLIP-2-T5XL</td><td>1.71%</td><td>1.43%</td><td>6.01%</td><td>3.85%</td></tr><tr><td>GPT-4</td><td>BLIP-2-T5XL</td><td>8.55%</td><td>8.57%</td><td>16.74%</td><td>12.75%</td></tr><tr><td rowspan="4">Multimodal</td><td colspan="2" rowspan="4">IDEFICS-80B-Instruct <br> CogVLM Gemini-Pro GPT-4V</td><td rowspan="4">Image + Caps + Acc. Tree</td><td>0.43%</td><td>0.95%</td><td>0.86%</td><td>0.77%</td></tr><tr><td>0.00%</td><td>0.48%</td><td>0.43%</td><td>0.33%</td></tr><tr><td>3.42%</td><td>4.29%</td><td>8.15%</td><td>6.04%</td></tr><tr><td>8.12%</td><td>12.38%</td><td>19.74%</td><td>15.05%</td></tr><tr><td rowspan="3">Multimodal (SoM)</td><td colspan="2" rowspan="3">IDEFICS-80B-Instruct <br> CogVLM <br> Gemini-Pro <br> GPT-4V</td><td rowspan="3">Image + Caps + SoM</td><td>0.85%</td><td>0.95%</td><td>1.07%</td><td>0.99%</td></tr><tr><td>0.00% <br> 3.42%</td><td>0.48% <br> 3.81%</td><td>0.43% <br> 7.73%</td><td>0.33% <br> 5.71%</td></tr><tr><td>9.83%</td><td>17.14%</td><td>19.31%</td><td>16.37%</td></tr><tr><td>Human Performance</td><td>-</td><td>-</td><td>Webpage</td><td>91.07%</td><td>87.10%</td><td>88.39%</td><td>88.70%</td></tr></table>
<table><tbody><tr><td rowspan="2">模型类型</td><td rowspan="2">LLM 骨干网络</td><td rowspan="2">视觉骨干网络</td><td rowspan="2">输入</td><td colspan="4">成功率 (↑)</td></tr><tr><td>分类广告</td><td>Reddit</td><td>购物</td><td>总计</td></tr><tr><td rowspan="5">纯文本</td><td rowspan="5">LLaMA-2-70B Mixtral-8x7B Gemini-Pro GPT-3.5 GPT-4</td><td rowspan="5">-</td><td rowspan="5">可访问性树</td><td>0.43%</td><td>1.43%</td><td>1.29%</td><td>1.10%</td></tr><tr><td>1.71%</td><td>2.86%</td><td>1.29%</td><td>1.76%</td></tr><tr><td>0.85%</td><td>0.95%</td><td>3.43%</td><td>2.20%</td></tr><tr><td>0.43%</td><td>0.95%</td><td>3.65%</td><td>2.20%</td></tr><tr><td>5.56%</td><td>4.76%</td><td>9.23%</td><td>7.25%</td></tr><tr><td rowspan="6">字幕增强</td><td>LLaMA-2-70B</td><td>BLIP-2-T5XL</td><td rowspan="6">可访问性树 + 字幕</td><td>0.00%</td><td>0.95%</td><td>0.86%</td><td>0.66%</td></tr><tr><td>Mixtral-8x7B</td><td>BLIP-2-T5XL</td><td>1.28%</td><td>0.48%</td><td>2.79%</td><td>1.87%</td></tr><tr><td>GPT-3.5</td><td>LLaVA-7B</td><td>1.28%</td><td>1.43%</td><td>4.08%</td><td>2.75%</td></tr><tr><td>GPT-3.5</td><td>BLIP-2-T5XL</td><td>0.85%</td><td>1.43%</td><td>4.72%</td><td>2.97%</td></tr><tr><td>Gemini-Pro</td><td>BLIP-2-T5XL</td><td>1.71%</td><td>1.43%</td><td>6.01%</td><td>3.85%</td></tr><tr><td>GPT-4</td><td>BLIP-2-T5XL</td><td>8.55%</td><td>8.57%</td><td>16.74%</td><td>12.75%</td></tr><tr><td rowspan="4">多模态</td><td colspan="2" rowspan="4">IDEFICS-80B-Instruct <br/> CogVLM Gemini-Pro GPT-4V</td><td rowspan="4">图像 + 字幕 + 可访问性树</td><td>0.43%</td><td>0.95%</td><td>0.86%</td><td>0.77%</td></tr><tr><td>0.00%</td><td>0.48%</td><td>0.43%</td><td>0.33%</td></tr><tr><td>3.42%</td><td>4.29%</td><td>8.15%</td><td>6.04%</td></tr><tr><td>8.12%</td><td>12.38%</td><td>19.74%</td><td>15.05%</td></tr><tr><td rowspan="3">多模态 (SoM)</td><td colspan="2" rowspan="3">IDEFICS-80B-Instruct <br/> CogVLM <br/> Gemini-Pro <br/> GPT-4V</td><td rowspan="3">图像 + 字幕 + SoM</td><td>0.85%</td><td>0.95%</td><td>1.07%</td><td>0.99%</td></tr><tr><td>0.00% <br/> 3.42%</td><td>0.48% <br/> 3.81%</td><td>0.43% <br/> 7.73%</td><td>0.33% <br/> 5.71%</td></tr><tr><td>9.83%</td><td>17.14%</td><td>19.31%</td><td>16.37%</td></tr><tr><td>人类表现</td><td>-</td><td>-</td><td>网页</td><td>91.07%</td><td>87.10%</td><td>88.39%</td><td>88.70%</td></tr></tbody></table>


Table 3: Success rates of baseline LLM and VLM agents on VisualWebArena.
表 3：基准 LLM 与 VLM 智能体在 VisualWebArena 上的成功率。


<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/02/2026_02_18__00_19_37_810fea.jpg"/>



Figure 3: Successful execution trajectory of the GPT-4V + SoM agent on the task for blocking a user that posted a certain picture. The text in red represents the actions output by the agent.
图 3：GPT-4V + SoM 智能体在屏蔽发布特定图片的用户的任务中的成功执行轨迹。红色文字代表智能体输出的动作。


SoM Improves Navigability We observe that the SoM representation (Sec. 5.3) further improves the performance of GPT-4V over using the accessibility tree, boosting overall success rate (15.05% → 16.37%). We observe particularly substantial improvements on Classifieds and Reddit, from 12.38% $\rightarrow  {17.14}\%$ and ${8.12}\%  \rightarrow  {9.83}\%$ respectively. We attribute this to the Classifieds and Reddit websites containing denser visual content. These websites often contain multiple smaller sized images that are arranged very closely (Fig. 2). In many of these pages, the accessibility tree does not provide sufficient information to disentangle elements that are spatially close. We hypothesize that the SoM representation is superior for strong VLM agents, which can more accurately disentangle and click on the desired elements. For the other VLMs, SoM does not significantly improve success, which we attribute to the finding from Yang et al. (2023a) that only GPT-4V demonstrates the SoM grounding ability (perhaps due to scale or training data).
SoM 提高导航能力 我们观察到，SoM 表示（第 5.3 节）相较于使用可访问性树进一步提升了 GPT-4V 的性能，使整体成功率从 15.05% 提升至 16.37%。在 Classifieds 和 Reddit 上，成功率分别从 12.38% $\rightarrow  {17.14}\%$ 和 ${8.12}\%  \rightarrow  {9.83}\%$ 提升，进步尤为显著。我们将其归因于 Classifieds 和 Reddit 网站包含更密集的视觉内容。这些网站通常包含多个排列紧密的小尺寸图像（图 2）。在许多此类页面中，可访问性树无法提供足够的信息来区分空间上接近的元素。我们假设 SoM 表示对于强大的 VLM 智能体更为优越，因为它们能更准确地解耦并点击目标元素。对于其他 VLM，SoM 并未显著提高成功率，我们将其归因于 Yang 等人 (2023a) 的发现，即只有 GPT-4V 展现出了 SoM 地标能力（可能源于模型规模或训练数据）。


One GPT-4V + SoM execution trajectory that we found particularly compelling was Reddit task #139, which requires exact image matching to find a post and block a user (Fig. 3). The model initially attempts to search for the correct forum, and when this fails it navigates to the list of forums. After navigating correctly to /f/memes, it identifies the offending image out of the many images on the page (Step 3 in Fig. 3) and blocks the author successfully without any unnecessary actions.
一个令我们印象深刻的 GPT-4V + SoM 执行轨迹是 Reddit 任务 #139，该任务需要精确的图像匹配来查找帖子并屏蔽用户（图 3）。模型最初尝试搜索正确的论坛，失败后导航至论坛列表。在正确导航至 /f/memes 后，它从页面上的众多图像中识别出违规图像（图 3 中的步骤 3），并在没有任何多余动作的情况下成功屏蔽了作者。


<table><tr><td>Task Subset</td><td>% of Total</td><td>SR (↑)</td></tr><tr><td>OCR required</td><td>17.1%</td><td>13.4%</td></tr><tr><td>No OCR required</td><td>82.9%</td><td>16.9%</td></tr><tr><td>Exact image match</td><td>8.7%</td><td>18.9%</td></tr><tr><td>No exact image match</td><td>91.3%</td><td>16.2%</td></tr><tr><td>Image inputs</td><td>25.2%</td><td>19.0%</td></tr><tr><td>No image inputs</td><td>74.8%</td><td>14.9%</td></tr></table>
<table><tbody><tr><td>任务子集</td><td>占总计 %</td><td>SR (↑)</td></tr><tr><td>需要 OCR</td><td>17.1%</td><td>13.4%</td></tr><tr><td>无需 OCR</td><td>82.9%</td><td>16.9%</td></tr><tr><td>图像完全匹配</td><td>8.7%</td><td>18.9%</td></tr><tr><td>无图像完全匹配</td><td>91.3%</td><td>16.2%</td></tr><tr><td>图像输入</td><td>25.2%</td><td>19.0%</td></tr><tr><td>无图像输入</td><td>74.8%</td><td>14.9%</td></tr></tbody></table>


Table 4: Success rate (SR) of GPT-4V (SoM) across different types of tasks.
表 4：GPT-4V (SoM) 在不同任务类型中的成功率 (SR)。


### 6.1 Performance by Task Type
### 6.1 按任务类型划分的性能


We analyze the success rate of the best VLM agent baseline (GPT-4V + SoM) across several additional subsets of tasks (Tab. 4). We include further analysis for other models in Appendix C.
我们分析了最佳 VLM 智能体基线 (GPT-4V + SoM) 在几个额外任务子集上的成功率 (表 4)。我们在附录 C 中包含了针对其他模型的进一步分析。


OCR Tasks 17.1% of VisualWebArena require optical character recognition (OCR), such as reading text from product images, or extracting text from an input image. We find that GPT-4V + SoM generally performs worse on tasks that require OCR (13.4%) compared to tasks which do not (16.9%), suggesting that OCR may be a bottleneck for current agents.
OCR 任务 VisualWebArena 中有 17.1% 的任务需要光学字符识别 (OCR)，例如从产品图像中读取文本，或从输入图像中提取文本。我们发现，与不需要 OCR 的任务 (16.9%) 相比，GPT-4V + SoM 在需要 OCR 的任务上的表现通常较差 (13.4%)，这表明 OCR 可能是当前智能体的瓶颈。


Exact Image Match 8.7% of tasks require exact image matching, which requires agents to identify precise visual matches. GPT-4V + SoM achieves a slightly higher success rate on this subset (18.9%) compared to other tasks (16.2%), suggesting that exact image matching is not a primary bottleneck.
图像精确匹配 8.7% 的任务需要图像精确匹配，这要求智能体识别精确的视觉匹配。与其它任务 (16.2%) 相比，GPT-4V + SoM 在该子集上取得了略高的成功率 (18.9%)，这表明图像精确匹配并非主要瓶颈。


Image Input Tasks 25.2% of VisualWebArena include one or more input images as part of the objective. These tasks generally appear more tractable for the GPT-4V + SoM agent, and it achieves a higher success rate (19.0%) compared to tasks without image inputs (14.9%).
图像输入任务 VisualWebArena 中有 25.2% 的任务在目标中包含一个或多个输入图像。对于 GPT-4V + SoM 智能体而言，这些任务通常看起来更容易处理，其成功率 (19.0%) 高于没有图像输入的任务 (14.9%)。


## 7 Conclusion
## 7 结论


In this work, we introduced VisualWebArena, a benchmark of realistic tasks designed to rigorously evaluate and advance the capabilities of autonomous multimodal web agents. VisualWebArena represents a significant step towards addressing the gap in the evaluation of multimodal agents on visually grounded tasks. We also introduce a visual agent inspired by Set-of-Marks prompting, and demonstrate the potential of this approach for simplifying action spaces and improving performance on visually complex websites. Our extensive evaluation of state-of-the-art LLM and VLM agents demonstrate that while VLMs show promise, there remains a considerable performance gap compared to humans, who achieve very high success rates on VisualWebArena. Our quantitative and qualitative analysis also highlights several common failure modes of existing LLM and VLM agents. We expect future work on improving the reasoning, visual understanding, and planning abilities of agents to be particularly exciting directions.
在这项工作中，我们推出了 VisualWebArena，这是一个旨在严格评估和提升自主多模态网络智能体能力的真实任务基准。VisualWebArena 代表了在解决多模态智能体视觉落地任务评估差距方面迈出的重要一步。我们还介绍了一种受 Set-of-Marks 提示启发的视觉智能体，并展示了该方法在简化动作空间和提高视觉复杂网站性能方面的潜力。我们对尖端 LLM 和 VLM 智能体的广泛评估表明，虽然 VLM 展现出前景，但与在 VisualWebArena 上获得极高成功率的人类相比，仍存在相当大的性能差距。我们的定量和定性分析还强调了现有 LLM 和 VLM 智能体的几种常见失败模式。我们期待未来在提高智能体的推理、视觉理解和规划能力方面开展极具吸引力的工作。


## 8 Ethical and Broader Impacts
## 8 伦理与更广泛的影响


Real World Impacts Advancing the capabilities of autonomous agents comes with many broader considerations and ethical implications. Strong autonomous agents have the potential to improve the accessibility of computer-based tasks, potentially aiding individuals with disabilities or those lacking technical skills. More broadly, agents have the potential to automate large portions of routine computer work. While the capabilities of existing autonomous agents are insufficient for even simple tasks (as shown in this paper), these impacts highlight the need to ensure that the broader economic and social implications on employment are carefully considered if/when autonomous agents are deployed in real world applications.
现实世界影响 提升自主智能体的能力伴随着许多更广泛的考量和伦理影响。强大的自主智能体具有提高基于计算机任务的可访问性的潜力，可能有助于残障人士或缺乏技术技能的人士。更广泛地说，智能体有潜力使大部分常规计算机工作实现自动化。虽然现有自主智能体的能力甚至不足以完成简单任务（如本文所示），但这些影响强调，如果/当自主智能体部署在现实世界应用中时，需要确保仔细考虑对就业产生的更广泛的经济和社会影响。


Bias and Safety When developing autonomous agents, it is also imperative to ensure that these agents do not inadvertently exclude or disadvantage any group. Further analysis is essential to ensure that deployed agents do not exhibit unintended biases. Agents also have the potential to cause more harm (than regular LLMs) in real world applications if careful safeguards are not in place. Further research is necessary to understand and mitigate possible harmful behaviors.
偏见与安全 在开发自主智能体时，还必须确保这些智能体不会无意中排除或歧视任何群体。进一步的分析对于确保部署的智能体不会表现出意外的偏见至关重要。如果没有到位的细致安全保障措施，智能体在现实世界应用中也可能产生（比常规 LLM）更多的危害。为了理解并减轻可能存在的有害行为，进一步的研究是必要的。


Intended Uses VisualWebArena is a research benchmark to measure and evaluate the progress of multimodal agents. It is primarily meant to act as a self-contained sandbox environment for safely building robust agents. The models we presented in this paper are research prototypes, and not intended for deployment in practical applications in their current state (especially in high risk domains).
预期用途 VisualWebArena 是一个用于衡量和评估多模态智能体进展的研究基准。它主要旨在作为一个自给自足的沙盒环境，用于安全地构建稳健的智能体。我们在本文中展示的模型是研究原型，不打算在当前状态下（特别是在高风险领域）部署于实际应用中。


## References
## 参考文献


Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katie Millican, Malcolm Reynolds, et al. 2022. Flamingo: a visual language model for few-shot learning. NeurIPS.
Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katie Millican, Malcolm Reynolds, et al. 2022. Flamingo: a visual language model for few-shot learning. NeurIPS.


Stanislaw Antol, Aishwarya Agrawal, Jiasen Lu, Margaret Mitchell, Dhruv Batra, C Lawrence Zitnick, and Devi Parikh. 2015. Vqa: Visual question answering. In ICCV.
Stanislaw Antol, Aishwarya Agrawal, Jiasen Lu, Margaret Mitchell, Dhruv Batra, C Lawrence Zitnick, and Devi Parikh. 2015. Vqa: Visual question answering. In ICCV.


Amanda Bertsch, Uri Alon, Graham Neubig, and Matthew R Gormley. 2023. Unlimiformer: Long-range transformers with unlimited length input. NeurIPS.
Amanda Bertsch, Uri Alon, Graham Neubig, and Matthew R Gormley. 2023. Unlimiformer: Long-range transformers with unlimited length input. NeurIPS.


Maciej Besta, Nils Blach, Ales Kubicek, Robert Ger-stenberger, Lukas Gianinazzi, Joanna Gajda, Tomasz Lehmann, Michal Podstawski, Hubert Niewiadomski, Piotr Nyczyk, et al. 2023. Graph of thoughts: Solving elaborate problems with large language models. arXiv preprint arXiv:2308.09687.
Maciej Besta, Nils Blach, Ales Kubicek, Robert Ger-stenberger, Lukas Gianinazzi, Joanna Gajda, Tomasz Lehmann, Michal Podstawski, Hubert Niewiadomski, Piotr Nyczyk, 等. 2023. 思维图：利用大语言模型解决复杂问题. arXiv 预印本 arXiv:2308.09687.


Greg Brockman, Vicki Cheung, Ludwig Pettersson, Jonas Schneider, John Schulman, Jie Tang, and Woj-ciech Zaremba. 2016. Openai gym.
Greg Brockman, Vicki Cheung, Ludwig Pettersson, Jonas Schneider, John Schulman, Jie Tang, 和 Woj-ciech Zaremba. 2016. OpenAI Gym 平台.


Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. 2020. Language models are few-shot learners. NeurIPS.
Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, 等. 2020. 语言模型即少样本学习者. NeurIPS.


Stephanie CY Chan, Ishita Dasgupta, Junkyung Kim, Dharshan Kumaran, Andrew K Lampinen, and Felix Hill. 2022. Transformers generalize differently from information stored in context vs in weights. NeurIPS MemARI Workshop.
Stephanie CY Chan, Ishita Dasgupta, Junkyung Kim, Dharshan Kumaran, Andrew K Lampinen, 和 Felix Hill. 2022. Transformer 在上下文存储信息与权重存储信息上的泛化差异. NeurIPS MemARI 工作坊.


Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, et al. 2023. Palm: Scaling language modeling with pathways. JMLR.
Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, 等. 2023. PaLM：利用 Pathways 扩展语言模型规模. JMLR.


Xiangxiang Chu, Limeng Qiao, Xinyang Lin, Shuang Xu, Yang Yang, Yiming Hu, Fei Wei, Xinyu Zhang, Bo Zhang, Xiaolin Wei, et al. 2023. Mo-bilevlm: A fast, reproducible and strong vision language assistant for mobile devices. arXiv preprint arXiv:2312.16886.
Xiangxiang Chu, Limeng Qiao, Xinyang Lin, Shuang Xu, Yang Yang, Yiming Hu, Fei Wei, Xinyu Zhang, Bo Zhang, Xiaolin Wei, 等. 2023. MobileVLM：一种适用于移动设备的快速、可复现且强大的视觉语言助手. arXiv 预印本 arXiv:2312.16886.


Hyung Won Chung, Le Hou, Shayne Longpre, Barret Zoph, Yi Tay, William Fedus, Yunxuan Li, Xuezhi Wang, Mostafa Dehghani, Siddhartha Brahma, et al. 2022. Scaling instruction-finetuned language models. arXiv preprint arXiv:2210.11416.
Hyung Won Chung, Le Hou, Shayne Longpre, Barret Zoph, Yi Tay, William Fedus, Yunxuan Li, Xuezhi Wang, Mostafa Dehghani, Siddhartha Brahma, 等. 2022. 扩展指令微调语言模型的规模. arXiv 预印本 arXiv:2210.11416.


Xiang Deng, Yu Gu, Boyuan Zheng, Shijie Chen, Samuel Stevens, Boshi Wang, Huan Sun, and Yu Su. 2023. Mind2web: Towards a generalist agent for the web. NeurIPS.
Xiang Deng, Yu Gu, Boyuan Zheng, Shijie Chen, Samuel Stevens, Boshi Wang, Huan Sun, 和 Yu Su. 2023. Mind2Web：迈向通用的网络智能体. NeurIPS.


Stan Franklin and Art Graesser. 1996. Is it an agent, or just a program?: A taxonomy for autonomous agents. In International workshop on agent theories, architectures, and languages, pages 21-35. Springer.
Stan Franklin 和 Art Graesser. 1996. 是智能体还是程序？：自治智能体分类法. 智能体理论、架构与语言国际工作坊, 第 21-35 页. Springer.


Gemini Team Google. 2023. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805.
Gemini Team Google. 2023. Gemini：一系列能力强大的多模态模型. arXiv 预印本 arXiv:2312.11805.


Izzeddin Gur, Hiroki Furuta, Austin Huang, Mustafa Safdari, Yutaka Matsuo, Douglas Eck, and Aleksan-dra Faust. 2023. A real-world webagent with planning, long context understanding, and program synthesis. arXiv preprint arXiv:2307.12856.
Izzeddin Gur, Hiroki Furuta, Austin Huang, Mustafa Safdari, Yutaka Matsuo, Douglas Eck, 和 Aleksan-dra Faust. 2023. 具备规划、长上下文理解与程序合成能力的真实世界网络智能体. arXiv 预印本 arXiv:2307.12856.


Ari Holtzman, Jan Buys, Li Du, Maxwell Forbes, and Yejin Choi. 2020. The curious case of neural text degeneration. ICLR.
Ari Holtzman, Jan Buys, Li Du, Maxwell Forbes, 和 Yejin Choi. 2020. 神经文本退化的奇特案例. ICLR.


Wenyi Hong, Weihan Wang, Qingsong Lv, Jiazheng Xu, Wenmeng Yu, Junhui Ji, Yan Wang, Zihan Wang, Yuxiao Dong, Ming Ding, et al. 2023. Cogagent: A visual language model for gui agents. arXiv preprint arXiv:2312.08914.
Wenyi Hong, Weihan Wang, Qingsong Lv, Jiazheng Xu, Wenmeng Yu, Junhui Ji, Yan Wang, Zihan Wang, Yuxiao Dong, Ming Ding, 等. 2023. CogAgent：一种用于 GUI 智能体的视觉语言模型. arXiv 预印本 arXiv:2312.08914.


Nicholas R Jennings, Katia Sycara, and Michael Wooldridge. 1998. A roadmap of agent research and development. Autonomous agents and multi-agent systems, 1:7-38.
Nicholas R Jennings, Katia Sycara, 和 Michael Wooldridge. 1998. 智能体研究与发展路线图. 自治智能体与多智能体系统, 1:7-38.


Albert Q Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, et al. 2023. Mistral 7b. arXiv preprint arXiv:2310.06825.
Albert Q Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, 等. 2023. Mistral 7B. arXiv 预印本 arXiv:2310.06825.


Albert Q Jiang, Alexandre Sablayrolles, Antoine Roux, Arthur Mensch, Blanche Savary, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Emma Bou Hanna, Florian Bressand, et al. 2024. Mixtral of experts. arXiv preprint arXiv:2401.04088.
Albert Q Jiang, Alexandre Sablayrolles, Antoine Roux, Arthur Mensch, Blanche Savary, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Emma Bou Hanna, Florian Bressand, 等. 2024. Mixtral 专家混合模型. arXiv 预印本 arXiv:2401.04088.


Geunwoo Kim, Pierre Baldi, and Stephen McAleer. 2023. Language models can solve computer tasks. NeurIPS.
Geunwoo Kim, Pierre Baldi, and Stephen McAleer. 2023. 语言模型可以解决计算机任务。NeurIPS。


Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. 2023. Blip-2: Bootstrapping language-image pretraining with frozen image encoders and large language models. ICML.
Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. 2023. Blip-2：通过冻结图像编码器和大型语言模型引导图文预训练。ICML。


Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C Lawrence Zitnick. 2014. Microsoft coco: Common objects in context. ECCV.
Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C Lawrence Zitnick. 2014. Microsoft coco：上下文中的常见物体。ECCV。


Evan Zheran Liu, Kelvin Guu, Panupong Pasupat, Tian-lin Shi, and Percy Liang. 2018. Reinforcement learning on web interfaces using workflow-guided exploration. ICLR.
Evan Zheran Liu, Kelvin Guu, Panupong Pasupat, Tian-lin Shi, and Percy Liang. 2018. 利用工作流引导探索在 Web 界面上进行强化学习。ICLR。


Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. 2023a. Improved baselines with visual instruction tuning. arXiv preprint arXiv:2310.03744.
Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. 2023a. 通过视觉指令微调改进基准线。arXiv 预印本 arXiv:2310.03744。


Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. 2023b. Visual instruction tuning. NeurIPS.
Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. 2023b. 视觉指令微调。NeurIPS。


Xiao Liu, Hao Yu, Hanchen Zhang, Yifan Xu, Xuanyu Lei, Hanyu Lai, Yu Gu, Hangliang Ding, Kaiwen Men, Kejuan Yang, Shudan Zhang, Xiang Deng, Ao-han Zeng, Zhengxiao Du, Chenhui Zhang, Sheng Shen, Tianjun Zhang, Yu Su, Huan Sun, Minlie Huang, Yuxiao Dong, and Jie Tang. 2023c. Agent-bench: Evaluating llms as agents.
Xiao Liu, Hao Yu, Hanchen Zhang, Yifan Xu, Xuanyu Lei, Hanyu Lai, Yu Gu, Hangliang Ding, Kaiwen Men, Kejuan Yang, Shudan Zhang, Xiang Deng, Ao-han Zeng, Zhengxiao Du, Chenhui Zhang, Sheng Shen, Tianjun Zhang, Yu Su, Huan Sun, Minlie Huang, Yuxiao Dong, and Jie Tang. 2023c. Agent-bench：评估作为智能体的 LLM。


Zhiwei Liu, Weiran Yao, Jianguo Zhang, Le Xue, Shelby Heinecke, Rithesh Murthy, Yihao Feng, Zeyuan Chen, Juan Carlos Niebles, Devansh Arpit, et al. 2023d. Bolaa: Benchmarking and orchestrating llm-augmented autonomous agents. arXiv preprint arXiv:2308.05960.
Zhiwei Liu, Weiran Yao, Jianguo Zhang, Le Xue, Shelby Heinecke, Rithesh Murthy, Yihao Feng, Zeyuan Chen, Juan Carlos Niebles, Devansh Arpit, et al. 2023d. Bolaa：基准测试与编排 LLM 增强的自主智能体。arXiv 预印本 arXiv:2308.05960。


Shayne Longpre, Le Hou, Tu Vu, Albert Webson, Hyung Won Chung, Yi Tay, Denny Zhou, Quoc V Le, Barret Zoph, Jason Wei, et al. 2023. The flan collection: Designing data and methods for effective instruction tuning. ICML.
Shayne Longpre, Le Hou, Tu Vu, Albert Webson, Hyung Won Chung, Yi Tay, Denny Zhou, Quoc V Le, Barret Zoph, Jason Wei, et al. 2023. Flan 集合：设计有效指令微调的数据与方法。ICML。


Grégoire Mialon, Clémentine Fourrier, Craig Swift, Thomas Wolf, Yann LeCun, and Thomas Scialom. 2023. Gaia: a benchmark for general ai assistants. arXiv preprint arXiv:2311.12983.
Grégoire Mialon, Clémentine Fourrier, Craig Swift, Thomas Wolf, Yann LeCun, and Thomas Scialom. 2023. Gaia：通用人工智能助手基准测试。arXiv 预印本 arXiv:2311.12983。


OpenAI. 2023. Gpt-4 technical report.
OpenAI. 2023. Gpt-4 技术报告。


Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. 2022. Training language models to follow instructions with human feedback. NeurIPS.
Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. 2022. 训练语言模型以通过人类反馈遵循指令。NeurIPS。


Jack W Rae, Sebastian Borgeaud, Trevor Cai, Katie Millican, Jordan Hoffmann, Francis Song, John Aslanides, Sarah Henderson, Roman Ring, Susannah Young, et al. 2021. Scaling language models: Methods, analysis & insights from training gopher. arXiv preprint arXiv:2112.11446.
Jack W Rae, Sebastian Borgeaud, Trevor Cai, Katie Millican, Jordan Hoffmann, Francis Song, John Aslanides, Sarah Henderson, Roman Ring, Susannah Young, et al. 2021. 扩展语言模型：训练 Gopher 的方法、分析与见解。arXiv 预印本 arXiv:2112.11446。


Tianlin Shi, Andrej Karpathy, Linxi Fan, Jonathan Hernandez, and Percy Liang. 2017. World of bits: An open-domain platform for web-based agents. In ICML.
Tianlin Shi, Andrej Karpathy, Linxi Fan, Jonathan Hernandez, and Percy Liang. 2017. Bits 世界：一个用于 Web 智能体的开放域平台。ICML。


Yi Tay, Mostafa Dehghani, Samira Abnar, Yikang Shen, Dara Bahri, Philip Pham, Jinfeng Rao, Liu Yang, Sebastian Ruder, and Donald Metzler. 2021. Long range arena: A benchmark for efficient transformers. ICLR.
Yi Tay, Mostafa Dehghani, Samira Abnar, Yikang Shen, Dara Bahri, Philip Pham, Jinfeng Rao, Liu Yang, Sebastian Ruder, and Donald Metzler. 2021. 长程竞技场：高效 Transformer 的基准测试。ICLR。


Shengbang Tong, Zhuang Liu, Yuexiang Zhai, Yi Ma, Yann LeCun, and Saining Xie. 2024. Eyes wide shut? exploring the visual shortcomings of multimodal llms. arXiv preprint arXiv:2401.06209.
Shengbang Tong, Zhuang Liu, Yuexiang Zhai, Yi Ma, Yann LeCun, and Saining Xie. 2024. 睁眼瞎？探索多模态 LLM 的视觉缺陷。arXiv 预印本 arXiv:2401.06209。


Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. 2023a. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971.
Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. 2023a. Llama: 开放且高效的基础语言模型. arXiv preprint arXiv:2302.13971.


Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. 2023b. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288.
Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. 2023b. Llama 2: 开放的基础及微调聊天模型. arXiv preprint arXiv:2307.09288.


Maria Tsimpoukelli, Jacob L Menick, Serkan Cabi, SM Eslami, Oriol Vinyals, and Felix Hill. 2021. Multimodal few-shot learning with frozen language models. NeurIPS.
Maria Tsimpoukelli, Jacob L Menick, Serkan Cabi, SM Eslami, Oriol Vinyals, and Felix Hill. 2021. 基于冻结语言模型的多模态少样本学习. NeurIPS.


Szymon Tworkowski, Konrad Staniszewski, Mikołaj Pacek, Yuhuai Wu, Henryk Michalewski, and Piotr Mitoś. 2023. Focused transformer: Contrastive training for context scaling. NeurIPS.
Szymon Tworkowski, Konrad Staniszewski, Mikołaj Pacek, Yuhuai Wu, Henryk Michalewski, and Piotr Mitoś. 2023. 聚焦 Transformer：用于上下文扩展的对比训练. NeurIPS.


Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. NeurIPS.
Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. NeurIPS.


Oriol Vinyals, Alexander Toshev, Samy Bengio, and Dumitru Erhan. 2015. Show and tell: A neural image caption generator. In CVPR.
Oriol Vinyals, Alexander Toshev, Samy Bengio, and Dumitru Erhan. 2015. Show and tell: 神经图像标题生成器. In CVPR.


Lei Wang, Chen Ma, Xueyang Feng, Zeyu Zhang, Hao Yang, Jingsen Zhang, Zhiyuan Chen, Jiakai Tang, Xu Chen, Yankai Lin, et al. 2023a. A survey on large language model based autonomous agents. arXiv preprint arXiv:2308.11432.
Lei Wang, Chen Ma, Xueyang Feng, Zeyu Zhang, Hao Yang, Jingsen Zhang, Zhiyuan Chen, Jiakai Tang, Xu Chen, Yankai Lin, et al. 2023a. 基于大语言模型的自主代理综述. arXiv preprint arXiv:2308.11432.


Weihan Wang, Qingsong Lv, Wenmeng Yu, Wenyi Hong, Ji Qi, Yan Wang, Junhui Ji, Zhuoyi Yang, Lei Zhao, Xixuan Song, et al. 2023b. Cogvlm: Visual expert for pretrained language models. arXiv preprint arXiv:2311.03079.
Weihan Wang, Qingsong Lv, Wenmeng Yu, Wenyi Hong, Ji Qi, Yan Wang, Junhui Ji, Zhuoyi Yang, Lei Zhao, Xixuan Song, et al. 2023b. CogVLM: 预训练语言模型的视觉专家. arXiv preprint arXiv:2311.03079.


Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le, Ed Chi, Sharan Narang, Aakanksha Chowdh-ery, and Denny Zhou. 2023c. Self-consistency improves chain of thought reasoning in language models. ICLR.
Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le, Ed Chi, Sharan Narang, Aakanksha Chowdh-ery, and Denny Zhou. 2023c. 自一致性提升语言模型中的思维链推理能力. ICLR.


Zhou Wang, Alan C Bovik, Hamid R Sheikh, and Eero P Simoncelli. 2004. Image quality assessment: from error visibility to structural similarity. IEEE transactions on image processing, 13(4):600-612.
Zhou Wang, Alan C Bovik, Hamid R Sheikh, and Eero P Simoncelli. 2004. 图像质量评估：从误差可见性到结构相似性. IEEE transactions on image processing, 13(4):600-612.


Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. 2022. Chain-of-thought prompting elicits reasoning in large language models. NeurIPS.
Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. 2022. 思维链提示激发大语言模型的推理能力. NeurIPS.


Zhiheng Xi, Wenxiang Chen, Xin Guo, Wei He, Yiwen Ding, Boyang Hong, Ming Zhang, Junzhe Wang, Senjie Jin, Enyu Zhou, et al. 2023. The rise and potential of large language model based agents: A survey. arXiv preprint arXiv:2309.07864.
Zhiheng Xi, Wenxiang Chen, Xin Guo, Wei He, Yiwen Ding, Boyang Hong, Ming Zhang, Junzhe Wang, Senjie Jin, Enyu Zhou, et al. 2023. 基于大语言模型的代理之崛起与潜力：综述. arXiv preprint arXiv:2309.07864.


An Yan, Zhengyuan Yang, Wanrong Zhu, Kevin Lin, Linjie Li, Jianfeng Wang, Jianwei Yang, Yiwu Zhong, Julian McAuley, Jianfeng Gao, et al. 2023. Gpt- 4v in wonderland: Large multimodal models for zero-shot smartphone gui navigation. arXiv preprint arXiv:2311.07562.
An Yan, Zhengyuan Yang, Wanrong Zhu, Kevin Lin, Linjie Li, Jianfeng Wang, Jianwei Yang, Yiwu Zhong, Julian McAuley, Jianfeng Gao, et al. 2023. GPT-4V 漫游仙境：用于零样本智能手机 GUI 导航的大型多模态模型. arXiv preprint arXiv:2311.07562.


Jianwei Yang, Hao Zhang, Feng Li, Xueyan Zou, Chun-yuan Li, and Jianfeng Gao. 2023a. Set-of-mark prompting unleashes extraordinary visual grounding in gpt-4v. arXiv preprint arXiv:2310.11441.
Jianwei Yang, Hao Zhang, Feng Li, Xueyan Zou, Chun-yuan Li, and Jianfeng Gao. 2023a. 标记集提示释放 GPT-4V 非凡的视觉定位能力. arXiv preprint arXiv:2310.11441.


Zhao Yang, Jiaxuan Liu, Yucheng Han, Xin Chen, Ze-biao Huang, Bin Fu, and Gang Yu. 2023b. Appa-gent: Multimodal agents as smartphone users. arXiv preprint arXiv:2312.13771.
Zhao Yang, Jiaxuan Liu, Yucheng Han, Xin Chen, Ze-biao Huang, Bin Fu, and Gang Yu. 2023b. AppAgent: 作为智能手机用户的多模态代理. arXiv preprint arXiv:2312.13771.


Shunyu Yao, Howard Chen, John Yang, and Karthik Narasimhan. 2022. Webshop: Towards scalable real-world web interaction with grounded language agents. NeurIPS.
Shunyu Yao, Howard Chen, John Yang, and Karthik Narasimhan. 2022. Webshop: Towards scalable real-world web interaction with grounded language agents. NeurIPS.


Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Thomas L Griffiths, Yuan Cao, and Karthik Narasimhan. 2023. Tree of thoughts: Deliberate problem solving with large language models. NeurIPS.
Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Thomas L Griffiths, Yuan Cao, and Karthik Narasimhan. 2023. Tree of thoughts: Deliberate problem solving with large language models. NeurIPS.


Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, et al. 2023. Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. arXiv preprint arXiv:2311.16502.
Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, et al. 2023. Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. arXiv preprint arXiv:2311.16502.


Aohan Zeng, Mingdao Liu, Rui Lu, Bowen Wang, Xiao Liu, Yuxiao Dong, and Jie Tang. 2023. Agenttuning: Enabling generalized agent abilities for llms. arXiv preprint arXiv:2310.12823.
Aohan Zeng, Mingdao Liu, Rui Lu, Bowen Wang, Xiao Liu, Yuxiao Dong, and Jie Tang. 2023. Agenttuning: Enabling generalized agent abilities for llms. arXiv preprint arXiv:2310.12823.


Zhuosheng Zhan and Aston Zhang. 2023. You only look at screens: Multimodal chain-of-action agents. arXiv preprint arXiv:2309.11436.
Zhuosheng Zhan and Aston Zhang. 2023. You only look at screens: Multimodal chain-of-action agents. arXiv preprint arXiv:2309.11436.


Susan Zhang, Stephen Roller, Naman Goyal, Mikel Artetxe, Moya Chen, Shuohui Chen, Christopher Dewan, Mona Diab, Xian Li, Xi Victoria Lin, et al. 2022. Opt: Open pre-trained transformer language models. arXiv preprint arXiv:2205.01068.
Susan Zhang, Stephen Roller, Naman Goyal, Mikel Artetxe, Moya Chen, Shuohui Chen, Christopher Dewan, Mona Diab, Xian Li, Xi Victoria Lin, et al. 2022. Opt: Open pre-trained transformer language models. arXiv preprint arXiv:2205.01068.


Boyuan Zheng, Boyu Gou, Jihyung Kil, Huan Sun, and Yu Su. 2024. Gpt-4v (ision) is a generalist web agent, if grounded. arXiv preprint arXiv:2401.01614.
Boyuan Zheng, Boyu Gou, Jihyung Kil, Huan Sun, and Yu Su. 2024. Gpt-4v (ision) is a generalist web agent, if grounded. arXiv preprint arXiv:2401.01614.


Shuyan Zhou, Frank F Xu, Hao Zhu, Xuhui Zhou, Robert Lo, Abishek Sridhar, Xianyi Cheng, Yonatan Bisk, Daniel Fried, Uri Alon, et al. 2024. Webarena: A realistic web environment for building autonomous agents. ICLR.
Shuyan Zhou, Frank F Xu, Hao Zhu, Xuhui Zhou, Robert Lo, Abishek Sridhar, Xianyi Cheng, Yonatan Bisk, Daniel Fried, Uri Alon, et al. 2024. Webarena: A realistic web environment for building autonomous agents. ICLR.


## Appendix
## 附录


We provide further information on the collected tasks (Sec A), analysis on model failure modes for Gemini and GPT-4 (Sec. C), more details on the new Classifieds environment (Sec. D), and on the task collection process (Sec. E).
我们提供了关于所收集任务的进一步信息（A 节）、Gemini 与 GPT-4 模型失败模式的分析（C 节）、新 Classifieds 环境的更多细节（D 节）以及任务收集过程的详细说明（E 节）。


## A Tasks Breakdown
## A 任务细分


Distribution of Tasks Across Sites
各网站任务分布


<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/02/2026_02_18__00_19_37_74bbd4.jpg"/>



Figure 4: Tasks proportion by sites.
图 4：按网站划分的任务比例。


<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/02/2026_02_18__00_19_37_acb964.jpg"/>



Figure 5: Tasks proportion by difficulty.
图 5：按难度划分的任务比例。


As described in Sec. 4.2, we collected a total of 910 tasks across the Classifieds, Reddit, and Shopping sites, with several multi-site tasks that involve more than one site. Several of these tasks also reference Wikipedia as a knowledge base. The breakdown across various sites is summarized in Fig. 4.
如 4.2 节所述，我们在 Classifieds、Reddit 和 Shopping 网站共收集了 910 个任务，其中包含多个涉及多个网站的任务。其中部分任务还将维基百科作为知识库。图 4 总结了各网站的任务分布情况。


The difficulty level of each task (for both visual difficulty and action difficulty) is summarized in Fig. 5, according to the specifications detailed in Sec. 4.2. VisualWebArena tasks span a variety of difficulty levels. In Sec. C.2 below, we also discuss the success rate of the agents across difficulty levels, and find that these are roughly correlated, with success rate decreasing as difficulty increases.
根据 4.2 节中的详细规范，图 5 总结了每个任务的难度级别（包括视觉难度和动作难度）。VisualWebArena 任务涵盖了多种难度级别。在下文 C.2 节中，我们还讨论了智能体在不同难度级别下的成功率，发现二者大致相关，成功率随难度增加而下降。


<table><tr><td rowspan="2">Agent Backbone</td><td rowspan="2">Model Type</td><td colspan="4">Success Rate (↑)</td></tr><tr><td>Classifieds</td><td>Reddit</td><td>Shopping</td><td>Overall</td></tr><tr><td>Llama-3-70B-Instruct</td><td>Caption-augmented</td><td>7.69%</td><td>5.24%</td><td>12.88%</td><td>9.78%</td></tr><tr><td>Gemini-Flash-1.5</td><td>Image + Caps + SoM</td><td>3.85%</td><td>4.76%</td><td>8.80%</td><td>6.59%</td></tr><tr><td>Gemini-Pro-1.5</td><td>Image + Caps + SoM</td><td>5.98%</td><td>12.86%</td><td>14.59%</td><td>11.98%</td></tr><tr><td>GPT-40</td><td>Image + Caps + SoM</td><td>20.51%</td><td>16.67%</td><td>20.82%</td><td>19.78%</td></tr></table>
<table><tbody><tr><td rowspan="2">智能体骨干网络</td><td rowspan="2">模型类型</td><td colspan="4">成功率 (↑)</td></tr><tr><td>分类广告</td><td>Reddit</td><td>购物</td><td>总计</td></tr><tr><td>Llama-3-70B-Instruct</td><td>标题增强</td><td>7.69%</td><td>5.24%</td><td>12.88%</td><td>9.78%</td></tr><tr><td>Gemini-Flash-1.5</td><td>图像 + 标题 + SoM</td><td>3.85%</td><td>4.76%</td><td>8.80%</td><td>6.59%</td></tr><tr><td>Gemini-Pro-1.5</td><td>图像 + 标题 + SoM</td><td>5.98%</td><td>12.86%</td><td>14.59%</td><td>11.98%</td></tr><tr><td>GPT-4o</td><td>图像 + 标题 + SoM</td><td>20.51%</td><td>16.67%</td><td>20.82%</td><td>19.78%</td></tr></tbody></table>


Table 5: Success rates of recent LLM and VLM agents on VisualWebArena.
表 5：近期 LLM 与 VLM 智能体在 VisualWebArena 上的成功率。


## B Additional Results
## B 额外结果


After the ACL submission deadline, we also ran the SoM agent with other recently released frontier VLMs: GPT-40 ${}^{4}$ , Gemini-Pro 1.5 (gemini-1.5-pro-preview-0514), and Gemini-Flash 1.5 (gemini-1.5-flash-preview-0514). We note that these recent models are natively multimodal, which may allow them to achieve stronger performance on multimodal tasks such as Visual-WebArena. We also run the Llama-3-70B-Instruct text-only LLM, augmented with captions from BLIP-2. The results are summarized in Tab. 5. GPT-40 achieves a success rate of 19.78%, and outperforms GPT-4V (16.37%).
在 ACL 投稿截止日期后，我们还结合其他近期发布的尖端 VLM 运行了 SoM 智能体，包括：GPT-4o ${}^{4}$、Gemini-Pro 1.5 (gemini-1.5-pro-preview-0514) 以及 Gemini-Flash 1.5 (gemini-1.5-flash-preview-0514)。我们注意到，这些近期模型是原生多模态的，这可能使它们在 VisualWebArena 等多模态任务上获得更强的性能。我们还运行了通过 BLIP-2 生成的字幕进行增强的 Llama-3-70B-Instruct 纯文本 LLM。结果汇总于表 5。GPT-4o 实现了 19.78% 的成功率，优于 GPT-4V (16.37%)。


Interestingly, we observe that Llama-3-70B-Instruct performs substantially better its Llama- 2-70B predecessor, achieving an overall success rate of ${9.78}\%$ ,which is only slightly below the success rate of the caption-augmented GPT-4 agent (12.75%), and substantially better than the caption augmented GPT-3.5 (2.97%) and Llama-2-70B (0.66%) agents.
有趣的是，我们观察到 Llama-3-70B-Instruct 的表现显著优于其前身 Llama-2-70B，实现了 ${9.78}\%$ 的总成功率，仅略低于字幕增强型 GPT-4 智能体 (12.75%)，且显著优于字幕增强型 GPT-3.5 (2.97%) 和 Llama-2-70B (0.66%) 智能体。


## C Further Analysis
## C 进一步分析


### C.1 Few-shot Prompting
### C.1 少样本提示 (Few-shot Prompting)


<table><tr><td rowspan="2">#Examples</td><td colspan="4">Success Rate (↑)</td></tr><tr><td>Classifieds</td><td>Reddit</td><td>Shopping</td><td>Overall</td></tr><tr><td>0</td><td>4.29%</td><td>2.38%</td><td>0.43%</td><td>2.86%</td></tr><tr><td>1</td><td>5.36%</td><td>1.43%</td><td>2.14%</td><td>3.63%</td></tr><tr><td>3</td><td>8.15%</td><td>4.29%</td><td>3.42%</td><td>6.04%</td></tr></table>
<table><tbody><tr><td rowspan="2">#示例</td><td colspan="4">成功率 (↑)</td></tr><tr><td>分类广告</td><td>Reddit</td><td>购物</td><td>总计</td></tr><tr><td>0</td><td>4.29%</td><td>2.38%</td><td>0.43%</td><td>2.86%</td></tr><tr><td>1</td><td>5.36%</td><td>1.43%</td><td>2.14%</td><td>3.63%</td></tr><tr><td>3</td><td>8.15%</td><td>4.29%</td><td>3.42%</td><td>6.04%</td></tr></tbody></table>


Table 6: Performance with different number of in-context examples.
表 6：不同上下文示例数量下的性能。


In most of our main experimental results, we prompt the model with 3 in-context examples. We perform an analysis of the success rate against the number of in-context examples provided (Tab. 6). For 1-shot experiments, we provide the model with the single in-context example from its corresponding environment. All experiments are run with the multimodal Gemini-Pro model (as GPT-4V is prohibitively expensive) with the Image + Caption + Acc. Tree as the observation space.
在大多数主要实验结果中，我们为模型提供 3 个上下文示例。我们分析了成功率与提供的上下文示例数量之间的关系（表 6）。对于单次学习（1-shot）实验，我们为模型提供来自其对应环境的单个上下文示例。所有实验均使用多模态 Gemini-Pro 模型运行（因为 GPT-4V 过于昂贵），并将图像 + 标题 + 辅助功能树作为观测空间。


We observe that overall success rate tends to increase with the number of examples provided, with a significant jump from 1 to 3 in-context examples. The improved results with a greater number of examples suggest that the performance of the VLM agents may improve significantly if we fine-tune the models on web trajectories, which will be an exciting direction for future work.
我们观察到，总体成功率往往随着提供的示例数量增加而提高，从 1 个到 3 个上下文示例有显著飞跃。更多示例带来的结果提升表明，如果我们针对网页轨迹微调模型，VLM 智能体的性能可能会显著提高，这将是未来工作的一个令人兴奋的方向。


### C.2 Performance by Task Difficulty
### C.2 按任务难度划分的性能


We conduct an analysis of the GPT-4 models across different action and visual difficulty levels (Fig. 6). We observe that success rate generally decreases as action/vision difficulty increases, which makes intuitive sense based on the difficulty taxonomy described in Sec. 4.2. The findings also show that multimodal models perform better especially on hard visual tasks. On this subset, GPT-4V + SoM achieves an average success rate of ${12.4}\%$ ,which is significantly higher than that of the caption-augmented (8.0%) and the text-only agents (4.8%). In addition to success rates, the GPT-4V trajectory lengths also increased with action difficulty, with harder tasks requiring more steps.
我们对 GPT-4 模型在不同操作和视觉难度级别下的表现进行了分析（图 6）。我们观察到，成功率通常随着操作/视觉难度的增加而下降，这符合第 4.2 节中描述的难度分类。研究结果还表明，多模态模型在困难视觉任务上的表现尤为出色。在此子集上，GPT-4V + SoM 实现了 ${12.4}\%$ 的平均成功率，显著高于标题增强型（8.0%）和纯文本智能体（4.8%）。除了成功率，GPT-4V 的轨迹长度也随着操作难度的增加而增加，更困难的任务需要更多的步骤。


### C.3 Task Subset Analysis
### C.3 任务子集分析


In this section, we provide more fine-grained analysis across different task subsets, similar to the one in Sec. 6.1 of the main paper. We examine both the GPT-4 text and multimodal agents, as well as the Gemini-Pro agents. This analysis may provide useful insights towards capabilities that future VLM models should have to perform well on web navigation tasks (specifically, OCR, exact image matching, and handling multiple interleaved image and text inputs).
在本节中，我们对不同任务子集进行了更细致的分析，类似于正文第 6.1 节。我们考察了 GPT-4 文本和多模态智能体，以及 Gemini-Pro 智能体。该分析可能会为未来 VLM 模型在网页导航任务中表现出色应具备的能力（具体而言是 OCR、精确图像匹配以及处理多个交错的图像和文本输入）提供有用的见解。


---



${}^{4}$ https://openai.com/index/hello-gpt-4o/
${}^{4}$ https://openai.com/index/hello-gpt-4o/


---



Visual Difficulty (v)
视觉难度 (v)


<table><tr><td>a \\v</td><td>Easy</td><td>Medium</td><td>Hard</td><td>Overall</td></tr><tr><td>Easy</td><td>30.1%</td><td>20.5%</td><td>26.3%</td><td>25.8%</td></tr><tr><td>Medium</td><td>15.2%</td><td>11.3%</td><td>11.7%</td><td>12.9%</td></tr><tr><td>hard</td><td>14.1%</td><td>10.4%</td><td>8.9%</td><td>10.5%</td></tr><tr><td>Overall</td><td>21.4%</td><td>14.3%</td><td>12.4%</td><td>16.4%</td></tr></table>
<table><tbody><tr><td>a \\v</td><td>简单</td><td>普通</td><td>困难</td><td>总计</td></tr><tr><td>简单</td><td>30.1%</td><td>20.5%</td><td>26.3%</td><td>25.8%</td></tr><tr><td>普通</td><td>15.2%</td><td>11.3%</td><td>11.7%</td><td>12.9%</td></tr><tr><td>困难</td><td>14.1%</td><td>10.4%</td><td>8.9%</td><td>10.5%</td></tr><tr><td>总计</td><td>21.4%</td><td>14.3%</td><td>12.4%</td><td>16.4%</td></tr></tbody></table>


a \\v Easy Medium Hard Overall
a \\v 简单 中等 困难 总体


Easy 18.9% 11.1% 10.5% 14.8%
简单 18.9% 11.1% 10.5% 14.8%


Medium 1.6% 6.1% 7.8% 4.7%
中等 1.6% 6.1% 7.8% 4.7%


Action Difficulty (a) hard 1.6% 4.2% 1.5% 2.4%
动作难度 (a) 困难 1.6% 4.2% 1.5% 2.4%


Overall 9.0% 7.3% 4.8% 7.3%
总体 9.0% 7.3% 4.8% 7.3%


(a) Success rate of GPT-4 Text-only (c) Success rate of GPT-4V + SoM
(a) GPT-4 纯文本成功率 (c) GPT-4V + SoM 成功率


a \\v Easy Medium Hard Overall a \\v Easy Medium Hard Overall
a \\v 简单 中等 困难 总体 a \\v 简单 中等 困难 总体


Easy 23.1% 18.8% 13.2% 20.1% Easy 6.0 7.7 6.1 6.9
简单 23.1% 18.8% 13.2% 20.1% 简单 6.0 7.7 6.1 6.9


Medium 14.4% 9.6% 5.2% 10.4% Medium 10.4 10.6 7.2 10.0
中等 14.4% 9.6% 5.2% 10.4% 中等 10.4 10.6 7.2 10.0


Hard 7.8% 7.3% 8.1% 7.8% Hard 14.1 9.2 12.5 12.1
困难 7.8% 7.3% 8.1% 7.8% 困难 14.1 9.2 12.5 12.1


Overall 16.9% 12.2% 8.0% 12.7% Overall 9.5 9.4 10.2 9.6
总体 16.9% 12.2% 8.0% 12.7% 总体 9.5 9.4 10.2 9.6


(b) Success rate of GPT-4 + Captions (d) Trajectory length of GPT-4V + SoM
(b) GPT-4 + 标题成功率 (d) GPT-4V + SoM 轨迹长度


Figure 6: Success rates $\left( {\mathrm{a},\mathrm{b},\mathrm{c}}\right)$ and trajectory lengths $\left( \mathrm{d}\right)$ across different difficulty levels.
图 6：不同难度级别的成功率 $\left( {\mathrm{a},\mathrm{b},\mathrm{c}}\right)$ 和轨迹长度 $\left( \mathrm{d}\right)$。


OCR Tasks On OCR tasks, which take up 17.1% of the benchmark, we observe that the GPT-4 family of models achieve a lower success rate on tasks that require OCR compared to tasks that do not (Fig. 7). This is consistent with the findings for GPT-4V + SoM reported in Sec. 6.1 of the main paper. We also observe that introducing multimodality (over just captions) substantially improves performance on OCR tasks (from 6.4% to 12.2%), showcasing the importance of having multimodal models for text recognition capabilities, as captioning models generally do not capture such fine-grained information.
OCR 任务 在占基准测试 17.1% 的 OCR 任务中，我们观察到 GPT-4 系列模型在需要 OCR 的任务上的成功率低于不需要 OCR 的任务（图 7）。这与正文第 6.1 节报告的 GPT-4V + SoM 的结果一致。我们还观察到，引入多模态（相较于仅使用标题）显著提升了 OCR 任务的性能（从 6.4% 提升至 12.2%），展示了多模态模型对于文本识别能力的重要性，因为标题生成模型通常无法捕获此类细粒度信息。


For Gemini-Pro agents, we also observe similar trends, with the multimodal and SoM models achieving a higher than proportionate gain on the OCR subset (compared to the non-OCR subset). Interestingly, the multimodal Gemini-Pro agents achieve a higher success rate on tasks that require OCR compared to tasks that do not. These results may suggest that it has strong inherent OCR capabilities, which we believe will be useful to explore in future work (especially on the stronger Gemini-Ultra model once it is generally available).
对于 Gemini-Pro 智能体，我们也观察到了类似的趋势，多模态和 SoM 模型在 OCR 子集上获得了不成比例的高增益（与非 OCR 子集相比）。有趣的是，多模态 Gemini-Pro 智能体在需要 OCR 的任务上的成功率高于不需要 OCR 的任务。这些结果可能表明它具有强大的原生 OCR 能力，我们认为这在未来的工作中值得探索（特别是在更强大的 Gemini-Ultra 模型普遍可用之后）。


Exact Image Match Of the tasks in VisualWebArena, 8.7% require exact image matching, which tests the ability of agents to identify images that have the exact same content (in contrast to those that are just semantically similar). From Fig. 8, we observe that the GPT-4V SoM model achieves a higher succeess rate on tasks that expect exact image match, while the other GPT-4 agents achieve a relatively lower success rate on the exact match subset. This suggests that the SoM representation may be more optimal for exact image match, due to its visual-centric observation and action space.
图像精确匹配 在 VisualWebArena 的任务中，有 8.7% 要求图像精确匹配，这测试了智能体识别内容完全相同（而非仅仅语义相似）图像的能力。从图 8 中我们可以看出，GPT-4V SoM 模型在需要精确图像匹配的任务上获得了更高的成功率，而其他 GPT-4 智能体在精确匹配子集上的成功率相对较低。这表明 SoM 表示法可能更适合精确图像匹配，因为它采用了以视觉为中心的观测和动作空间。


For the Gemini models, we observe that success rates on exact match tasks are substantially lower than success rates on non-exact match tasks. Interestingly, we also observe a similar trend as the GPT-4 agents, where introducing multimodality improves success rates on exact match tasks, which is further bolstered with the SoM representation.
对于 Gemini 模型，我们观察到精确匹配任务的成功率明显低于非精确匹配任务。有趣的是，我们还观察到了与 GPT-4 智能体类似的趋势，即引入多模态提高了精确匹配任务的成功率，而 SoM 表示进一步增强了这一趋势。


Image Input Tasks 25.2% (229 tasks) in VisualWebArena are specified with image inputs (e.g., the task in Fig. 3, and the first and third tasks in Fig. 1). The results of the Gemini-Pro and GPT-4 agents are summarized in Fig. 9.
图像输入任务：VisualWebArena 中 25.2%（229 个任务）指定了图像输入（例如图 3 中的任务，以及图 1 中的第一和第三个任务）。Gemini-Pro 和 GPT-4 智能体的结果总结在图 9 中。


We observe that for the GPT-4 agent, success rates are generally higher on tasks that involve image inputs, with the exception of the text-only agent. This aligns with intuition, as agents that do not have access to visual information would not be able to understand the task correctly, and would perform worse at successfully accomplishing it. For the captioning, multimodal, and SoM GPT-4 agents, success rates are higher on the tasks involving image input, which we attribute to these tasks being more tractable once the visual content is correctly understood.
我们观察到，对于 GPT-4 智能体，涉及图像输入的任务成功率通常更高，纯文本智能体除外。这符合直觉，因为无法获取视觉信息的智能体将无法正确理解任务，并且在成功完成任务方面的表现会更差。对于说明生成、多模态和 SoM GPT-4 智能体，涉及图像输入的任务成功率更高，我们将其归因于一旦视觉内容被正确理解，这些任务就会变得更易处理。


Interestingly, we see a contrast with the Gemini-Pro agents, where success rate is generally lower on tasks that involve input images. This may imply that the model may not be able to process multiple interleaved image-text inputs as well. This may be useful to revisit in the future with the stronger Gemini-Ultra model once it is released, or with stronger open sourced VLMs. We believe that being able to handle interleaved multimodal inputs will be a core requirement for strong web agents, and more comprehensive error analysis with stronger models may yield useful insights.
有趣的是，我们看到 Gemini-Pro 智能体的情况与之相反，其在涉及输入图像的任务上成功率通常较低。这可能意味着该模型可能无法很好地处理多个交错的图文输入。一旦更强大的 Gemini-Ultra 模型发布，或者有了更强大的开源 VLM，未来重新审视这一点可能会很有用。我们相信，能够处理交错的多模态输入将是强大的网络智能体的核心要求，而对更强大模型进行更全面的错误分析可能会产生有用的见解。


<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/02/2026_02_18__00_19_37_7f974f.jpg"/>



Figure 7: Success rate of GPT-4 and Gemini agents on tasks that do not require OCR vs. tasks that do.
图 7：GPT-4 和 Gemini 智能体在不需要 OCR 的任务与需要 OCR 的任务上的成功率。


Trajectory Lengths vs. Success Rates Hard reasoning tasks, on average, require more steps to be successfully solved. We plot the trajectory length of the GPT-4V + SoM model in Fig. 10. The findings suggest that the model assumes a significant portion of tasks can be completed in a few steps, as it terminates a majority of tasks after less than 10 steps. However, this assumption doesn't imply that the model successfully solves the majority of tasks: the error rate remains relatively uniform across longer trajectory lengths.
轨迹长度与成功率：平均而言，高难度推理任务需要更多步骤才能成功解决。我们在图 10 中绘制了 GPT-4V + SoM 模型的轨迹长度。研究结果表明，该模型假设很大一部分任务可以在几步内完成，因为它在不到 10 步后就终止了大部分任务。然而，这种假设并不意味着模型成功解决了大部分任务：在较长的轨迹长度上，错误率保持相对均匀。


### C.4 Failure Modes
### C.4 失败模式


In this section, we describe other common issues we observed with our baseline agent models.
在本节中，我们描述了在基准智能体模型中观察到的其他常见问题。


Failure Over Longer Horizons We observed that in several examples, the agents would correctly perform a task but undo it, leading to failure. The GPT-4 captioning-only model on shopping task 54 ("Add the one [poster] with waves to my wish list.") made an assumption that the product image with a caption about a lighthouse was the correct one, and added it to the wishlist. However, after going to the wish list page the agent removes the poster because "there is no explicit mention of waves in the current items listed on the Wish List page." This issue is not unique to the text input agents; even the GPT-4 SoM agent faced a similar problem in shopping task 397 ("Buy the item on the page with a banana theme."). The agent initially added the correct item to the shopping cart and proceeded to check out, but stopped in the middle stating in the reasoning trace output that it does not think the item fit the criteria (despite having added it to the cart just a few steps ago).
长时程失败：我们在几个示例中观察到，智能体会正确执行任务但随后又撤销，导致失败。GPT-4 纯说明模型在购物任务 54（“将带有海浪的那张 [海报] 添加到我的愿望清单。”）中，假设带有灯塔说明的产品图像是正确的，并将其添加到了愿望清单。然而，在进入愿望清单页面后，智能体移除了海报，理由是“当前愿望清单页面列出的物品中没有明确提到海浪”。这个问题并非文本输入智能体所独有；即使是 GPT-4 SoM 智能体在购物任务 397（“购买页面上带有香蕉主题的商品。”）中也遇到了类似问题。智能体最初将正确的商品添加到购物车并继续结账，但中途停止，在推理轨迹输出中表示它认为该商品不符合标准（尽管几步前才将其添加到购物车中）。


Failures on Easy Tasks We observed surprisingly poor performance on many tasks with easy action and easy visual difficulty levels, such as in shopping task 46, which tasks the agent to add the red product in the second row to the cart (starting on the page shown in Fig. 11). The multimodal and SoM GPT-4V agents clicked on a blue tablecloth in the first row and gave up when they couldn't find an option to order it in red. Despite appearing to be a simple task (the correct product is the red cloth in the second row), none of the agents we benchmarked were able to successfully complete it.
简单任务失败：我们观察到在许多动作和视觉难度级别较低的任务中，表现出奇地差，例如购物任务 46，该任务要求智能体将第二行中的红色产品添加到购物车（从图 11 所示的页面开始）。多模态和 SoM GPT-4V 智能体点击了第一行的一块蓝色桌布，并在找不到将其订购为红色的选项时放弃了。尽管这看起来是一个简单的任务（正确的产品是第二行的红色布料），但我们测试的所有智能体都未能成功完成它。


Giving Up Too Early Another frequent issue we observed that occurred across all the agents was giving up too early. For example, GPT-4V + SoM fails on shopping task 248 ("Order a 6 pack of the green chocolate bars. If the shipping is more than 7% of the total price, leave a 3 star review mentioning it, otherwise 5."). This tasks involves several steps which the model is able to correctly plan out, but the very first action needed is to slightly scroll down so the "add to cart" button is visible. However, even after identifying the correct product the model gives up on the first step instead of scrolling, because it does not immediately see the button. There are other instances of this occurring, such as in shopping task 175, where an agent will use the search bar to search for something, and then immediately give up because it does not see the target product instead of trying new ways to find the product.
过早放弃 我们观察到的另一个在所有智能体中普遍存在的问题是过早放弃。例如，GPT-4V + SoM 在购物任务 248（“订购 6 盒绿色巧克力棒。如果运费超过总价的 7%，请留下 3 星评价并注明，否则评价 5 星。”）中失败。该任务包含多个步骤，模型能够正确规划，但所需的第一步操作是稍微向下滚动以便看到“加入购物车”按钮。然而，即使在识别出正确的商品后，模型因为没有立即看到按钮而在第一步就放弃了，而不是进行滚动。类似情况还出现在其他案例中，如购物任务 175，智能体使用搜索栏搜索某物，然后因为没看到目标商品就立即放弃，而不是尝试新的寻找方式。


<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/02/2026_02_18__00_19_37_629ea4.jpg"/>



Figure 8: Success rates of agents on tasks that require exact image match vs. those that do not.
图 8：智能体在需要精确图像匹配的任务与不需要的任务上的成功率。


Getting Stuck in Loops Another issue we observed was oscillating or looping between pages, where the agent would look something up or navigate to a page, unsuccessfully attempt to perform the next action (such as adding it to the cart), and on failure it goes back and repeats from the beginning. An example of this is in classifieds task 205 where the model is tasked to compare two makeup palettes in two tabs, and the GPT-4V agent spends most of the time switching between the tabs. We believe that these issues will likely be alleviated by introducing more sophisticated tracking of past states and execution history, which is a promising direction for future work.
陷入循环 我们观察到的另一个问题是在页面之间震荡或循环，智能体查找到某些内容或导航到某个页面，尝试执行下一步操作（如加入购物车）失败，失败后又返回并从头开始。分类广告任务 205 是一个例子，模型被要求在两个标签页中对比两个化妆盘，而 GPT-4V 智能体大部分时间都在切换标签页。我们认为，通过引入更先进的过去状态和执行历史跟踪，这些问题可能会得到缓解，这是未来工作的一个有前景的方向。


Failure Example: Changing User Phone Number Shopping task #345, a multi-site task that also involves the Wikipedia site, demonstrated several interesting points of failure that we saw throughout the execution traces for many other tasks. Fig. 12 contains the execution trace of the GPT-4V multimodal agent for the task "Prepend the country code of South Korea to the phone number of my account profile." There are three major mistakes made by the agent in this execution trace:
失败案例：更改用户电话号码 购物任务 #345 是一个涉及维基百科网站的多站点任务，它展示了我们在许多其他任务的执行轨迹中看到的几个有趣的失败点。图 12 包含 GPT-4V 多模态智能体执行任务“将韩国国家代码添加到我账户资料的电话号码前”的执行轨迹。智能体在此执行轨迹中犯了三个主要错误：


- Useless actions: In step 3 of the trajectory, the agent creates a new blank tab and does not interact with it for the rest of the trajectory. While this does not impact the correctness of the final task, it does show that the agents sometimes take unnecessary steps.
- 无用操作：在轨迹的第 3 步，智能体创建了一个新的空白标签页，并在轨迹的其余部分未与其交互。虽然这不影响最终任务的正确性，但确实表明智能体有时会采取不必要的步骤。


- Appending text instead of replacing: Many agents added text to input fields without deleting the previous text, which would often result in long, repeating search queries or addresses. An example of this occurs in step 7 of Fig. 12.
- 追加文本而非替换：许多智能体在不删除原有文本的情况下向输入框添加文本，这通常导致生成冗长、重复的搜索查询或地址。图 12 的第 7 步就出现了这样的例子。


<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/02/2026_02_18__00_19_37_2708dc.jpg"/>



Figure 9: Success rates of agents on tasks that include input images as part of the specification vs. tasks that are specified with just written text.
图 9：智能体在包含输入图像作为说明的任务与仅通过文本说明的任务上的成功率。


- Repeating actions: Another frequent issue we saw across agents was repeating actions, like how the agent kept jumping between step 6 and step 7 of Fig. 12 until it hit the maximum trajectory length. In this case, we believe this looping effect stems from the issue mentioned above and each time the agent tries to correct the phone number, it keeps appending the correct number instead of replacing the incorrect number with the correct number.
- 重复操作：我们在不同智能体中看到的另一个频繁问题是重复操作，例如智能体在图 12 的第 6 步和第 7 步之间不断跳跃，直到达到最大轨迹长度。在这种情况下，我们认为这种循环效应源于上述问题，智能体每次尝试纠正电话号码时，都会不断追加正确的号码，而不是用正确的号码替换错误的号码。


### C.5 Comparison Between Agents
### C.5 智能体之间的比较


In this section, we describe some qualitative differences we observed between the different agents on various tasks in VisualWebArena.
在本节中，我们描述了在 VisualWebArena 的各种任务中观察到的不同智能体之间的一些定性差异。


Text-only vs. Caption-augmented Agents For GPT-4, the text model unsurprisingly performs much worse than even the captioning model, failing to even do the most basic tasks. For example, Red-dit task #101 is the relatively simple task to "Navigate to the comments section of a post that contains a picture of a keyboard." Out of all of the GPT-4 baseline agents, the text-only agent is the only one to fail this task, as it's unable to identify the appropriate post from just the title. Interestingly, it still manages to make an educated guess and navigate to the hottest post on /f/MechanicalKeyboards (which unfortunately, did not include a keyboard in its image).
纯文本与带描述增强的智能体 对于 GPT-4，不出所料，纯文本模型的表现远逊于带描述的模型，甚至无法完成最基本的任务。例如，Reddit 任务 #101 是一个相对简单的任务：“导航到包含键盘图片的帖子的评论区。”在所有 GPT-4 基准智能体中，纯文本智能体是唯一失败的，因为它无法仅根据标题识别出合适的帖子。有趣的是，它仍然尝试进行合理猜测，并导航到 /f/MechanicalKeyboards 上最热门的帖子（遗憾的是，该帖子的图片中不包含键盘）。


Caption-augmented vs. SoM Agents We observed in many examples that the GPT-4V SoM and multimodal agents outperformed the caption-augmented baselines in terms of navigation capabilities and visual understanding. The multimodal models were generally better at understanding visual information on webpages, as relevant information in many images is lost when they are translated into captions. One pertinent example is Reddit task #40, where a picture of the skyline of Pittsburgh is provided, and the task is "I'd like to find the subreddit for the city this photo was taken in. Can you navigate to it?". The GPT-4V + SoM agent correctly identifies the location of the photo, with the first line of its reasoning output as "The photo shows a city skyline with prominent buildings labeled with logos for UPMC and PNC, which suggests that the photo was taken in Pittsburgh, Pennsylvania.". Using this information, the agent is able to successfully navigate to the appropriate subreddit, /f/pittsburgh. In contrast, the captioning agent labels the image as "city skyline with many tall buildings" (as this is the output from the BLIP-
标题增强与 SoM 智能体对比 我们在多个示例中观察到，GPT-4V SoM 和多模态智能体在导航能力和视觉理解方面优于标题增强基线。多模态模型通常能更好地理解网页上的视觉信息，因为许多图像中的相关信息在转化为标题时会丢失。一个典型的例子是 Reddit 任务 #40，其中提供了一张匹兹堡天际线的照片，任务是“我想找到这张照片拍摄城市的子版块。你能导航到那里吗？”。GPT-4V + SoM 智能体正确识别了照片的地点，其推理输出的第一行是“照片显示了城市天际线，显著建筑上标有 UPMC 和 PNC 的标志，这表明照片是在宾夕法尼亚州匹兹堡拍摄的。”。利用这一信息，智能体能够成功导航到相应的子版块 /f/pittsburgh。相比之下，标题化智能体将图像标记为“拥有许多高楼的城市天际线”（因为这是来自 BLIP-


<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/02/2026_02_18__00_19_37_7b30a3.jpg"/>



Figure 10: Performance of the GPT-4V + SoM agent across different trajectory lengths.
图 10：GPT-4V + SoM 智能体在不同轨迹长度下的性能。


2 model), which prevents it from identifying the appropriate subreddit. This highlights a fundamental issue with captioning models: they frequently highlight salient visual information, which hinders success on many tasks that require fine-grained visual information not typically captured by captions.
2 模型的内容），这使其无法识别合适的子版块。这凸显了标题化模型的一个根本问题：它们经常强调显著的视觉信息，这阻碍了在许多需要标题通常无法捕获的细粒度视觉信息的任务上取得成功。


Multimodal vs. SoM Agents The SoM representation generally performs better on tasks that require more navigation steps, due to its simplified observation and action space. One example is clas-sifieds task #31, "Find the latest listing of a white Google Pixel phone and post a comment offering \$10 less than their asking price." (Fig. 13). While the multimodal model was unable to search for the correct terms, the SoM model was able to leverage the simplified action space to traverse more efficiently throughout the environment. It succeeded at this task by filtering for cell phones after the initial search for more relevant results, and managed to fill out the necessary comment form fields. From our observations, the SoM representation is generally more efficient compared to the multimodal representation (which only has access to the page screenshot and accessibility tree). With a strong VLM capable of SoM, the agent does not have to implicitly perform visual co-referencing to match elements from the accessibility tree to the visual buttons and inputs that it wants to interact with.
多模态与 SoM 智能体对比 由于其简化的观察和动作空间，SoM 表征在需要更多导航步骤的任务上通常表现更好。一个例子是分类广告任务 #31，“查找白色 Google Pixel 手机的最新列表，并发布评论，出价比其要价低 10 美元。”（图 13）。虽然多模态模型无法搜索正确的术语，但 SoM 模型能够利用简化的动作空间在整个环境中更高效地穿梭。它在初步搜索后通过筛选手机以获得更相关的结果，并成功填写了必要的评论表单字段。根据我们的观察，与多模态表征（仅能访问页面截图和辅助功能树）相比，SoM 表征通常更高效。凭借具备 SoM 能力的强大 VLM，智能体不必隐式执行视觉共指来将辅助功能树中的元素与其想要交互的视觉按钮和输入框相匹配。


<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/02/2026_02_18__00_19_37_55410d.jpg"/>



Figure 11: The starting page for the task "Add the red one in the second row of this page to my shopping cart."
图 11：任务“将此页面第二行的红色那个加入我的购物车”的起始页面。


<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/02/2026_02_18__00_19_37_f44b3a.jpg"/>



Figure 12: Unsuccessful execution trajectory of the GPT-4V multimodal agent on the task for adding the a country code to the user's phone number. The text in red represents the commands output by the agent.
图 12：GPT-4V 多模态智能体在为用户手机号添加国家代码任务中不成功的执行轨迹。红色文字代表智能体输出的命令。


## D The Classifieds Environment
## D 分类广告环境


The Classifieds environment contains 65,955 listings, each with a title, text description, and a product image of the item being sold. To populate the site with realistic content, we scraped data across a variety of categories on Craigslist over 3 weeks, focusing on the Northeastern States of the US (similar to the geographic region in the Reddit site). This approach ensured a diverse and rich dataset, representative of real-world classifieds posts. We utilized the scrubadub Python package for redacting Personally Identifiable Information (PII), including addresses, phone numbers, and emails. We use generated placeholders for names (e.g., "Bill Smith"), emails with fictitious addresses (e.g., bill_smith@example.com), and phone numbers with the fictional 555-prefix numbers.
分类广告环境包含 65,955 条列表，每条列表包含标题、文本描述以及所售物品的产品图像。为了给网站填充真实内容，我们在 3 周内爬取了 Craigslist 上各个类别的数据，重点关注美国东北部各州（类似于 Reddit 网站中的地理区域）。这种方法确保了数据集的多样性和丰富性，代表了真实的分类广告帖子。我们使用了 scrubadub Python 包来脱敏个人身份信息 (PII)，包括地址、电话号码和电子邮件。我们对姓名使用生成的占位符（例如“Bill Smith”），对电子邮件使用虚拟地址（例如 bill_smith@example.com），对电话号码使用虚拟的 555 前缀号码。


Fig. 14 and 15 show two pages within the Clas-sifieds site, the homepage and the detail page of a particular listing. Users can also use the search function, or filter posts by category or location to find items.
图 14 和 15 展示了分类广告网站中的两个页面：首页和特定列表的详情页。用户还可以使用搜索功能，或按类别或位置筛选帖子来查找物品。


## E Task Collection Process
## E 任务收集过程


Our main task collection process is described in Sec. 4.2. We collected the set of 910 tasks by recruiting 6 computer science graduate students (co-authors of this paper), who were all familiar with commercial versions of the Classifieds, Shopping, and Reddit sites, and have used them in their personal lives.
我们的主要任务收集过程在第 4.2 节中描述。我们通过招募 6 名计算机科学研究生（本文的共同作者）收集了 910 个任务，他们都熟悉分类广告、购物和 Reddit 网站的商业版本，并在个人生活中使用过它们。


Annotators were first instructed to spend some time exploring the VisualWebArena websites, to familiarize themselves with their functionality and content (as this may differ slightly from real world implementations). During task creation, we encouraged annotators to be creative, and make use of the visual layouts of the websites, input images, and cross-site functionalities to develop creative and realistic tasks. We ensured that there were no repeated tasks, and that there were not too many tasks of the same type (by first producing templates, followed by instantiating them with different arguments to create multiple tasks, as described in Sec. 4.2).
标注员首先被指示花一些时间探索 VisualWebArena 网站，以熟悉其功能和内容（因为这可能与现实世界的实现略有不同）。在任务创建过程中，我们鼓励标注员发挥创意，利用网站的视觉布局、输入图像和跨站点功能来开发创意且现实的任务。我们确保没有重复任务，并且同一类型的任务不会过多（通过先生成模板，然后用不同的参数实例化它们来创建多个任务，如第 4.2 节所述）。


## F Baseline Agent Settings
## F 基线智能体设置


For all baseline agents we report in the paper, we use a webpage viewport size of ${1280} \times  {2048}$ , and truncate text observations to 3840 tokens (or 15360 characters for Gemini). For models with shorter context windows (e.g., LLaMA, IDEFICS, CogVLM), we instead use a viewport size of ${1280} \times  {720}$ and truncate text observations to 640 tokens. For GPT-3.5 and GPT-4 models, we follow (Zhou et al., 2024) in using a temperature of 1.0 and a top-p of 0.9 . For Gemini models we use the suggested default temperature of 0.9 and top-p of 1.0. For the remaining models, we find that they benefit from sampling from lower temperatures, and use a temperature of 0.6 and top-p of 0.95 . Nucleus sampling (Holtzman et al., 2020) is used in all experiments.
对于文中报告的所有基准智能体，我们使用的网页视口大小为 ${1280} \times  {2048}$，并将文本观测截断为 3840 个 token（对于 Gemini 为 15360 个字符）。对于上下文窗口较短的模型（如 LLaMA、IDEFICS、CogVLM），我们改用 ${1280} \times  {720}$ 的视口大小，并将文本观测截断为 640 个 token。对于 GPT-3.5 和 GPT-4 模型，我们遵循 (Zhou et al., 2024) 使用 1.0 的温度和 0.9 的 top-p。对于 Gemini 模型，我们使用建议的默认温度 0.9 和 top-p 1.0。对于其余模型，我们发现它们受益于较低温度的采样，因此使用 0.6 的温度和 0.95 的 top-p。所有实验均采用核采样 (Holtzman et al., 2020)。


The system message and the prompt with in-context examples for the baseline SoM agents are shown in Fig. 16 and Fig. 17 respectively. We prompt the model with 3 in-context examples for all baselines. For multimodal and SoM models, we include the screenshot of each in-context example as well as the screenshot of the current page. For text-only and caption augmented models, the examples consist of just the text and captions.
基准 SoM 智能体的系统消息和带有上下文示例的提示分别如图 16 和图 17 所示。对于所有基准，我们使用 3 个上下文示例提示模型。对于多模态和 SoM 模型，我们包含每个上下文示例的截图以及当前页面的截图。对于纯文本和增强说明模型，示例仅包含文本和说明。


<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/02/2026_02_18__00_19_37_7c787d.jpg"/>



Figure 13: GPT-4V SoM agent on Classifieds task #31, "Find the latest listing of a white Google Pixel phone and post a comment offering \$10 less than their asking price.". It succeeds at the task by leveraging the more efficient navigation space.
图 13：GPT-4V SoM 智能体在 Classifieds 任务 #31 中的表现，“查找最新发布的白色 Google Pixel 手机，并发表评论，出价比其要价低 10 美元。”它通过利用更高效的导航空间成功完成了任务。


<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/02/2026_02_18__00_19_37_fcf639.jpg"/>



Figure 14: Homepage of the Classifieds site. Users can search for keywords, filter by category, or post location.
图 14：Classifieds 网站主页。用户可以搜索关键词，按类别或发布地点进行筛选。


<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/02/2026_02_18__00_19_37_7c68e2.jpg"/>



Figure 15: Example post in the Classifieds website. Users can add comments and reviews to individual listings.
图 15：Classifieds 网站中的示例帖子。用户可以对单个列表添加评论和回复。


---



You are an autonomous intelligent agent tasked with navigating a web browser. You will be given web-based tasks. These tasks
你是一个负责操作网页浏览器的自主智能体。你将接收基于网页的任务。这些任务


will be accomplished through the use of specific actions you can issue.
将通过你发出的特定动作来完成。


Here's the information you'll have:
以下是你将获得的信息：


The user's objective: This is the task you're trying to complete.
用户目标：这是你试图完成的任务。


The current web page screenshot: This is a screenshot of the webpage, with each interactable element assigned a unique
当前网页截图：这是网页的截图，每个可交互元素都被分配了一个唯一的


numerical id. Each bounding box and its respective id shares the same color.
数字 ID。每个边界框及其对应的 ID 共享相同的颜色。


The observation, which lists the IDs of all interactable elements on the current web page with their text content if any, in the
观测结果：按 [id] [tagType] [text content] 的格式列出当前网页上所有可交互元素的 ID 及其文本内容（如果有）。


format [id] [tagType] [text content]. tagType is the type of the element, such as button, link, or textbox. text content is the text
tagType 是元素的类型，如 button、link 或 textbox。text content 是元素的文本


content of the element. For example, [1234] [button] ['Add to Cart'] means that there is a button with id 1234 and text content
内容。例如，[1234] [button] ['Add to Cart'] 表示当前网页上有一个 ID 为 1234 且文本内容为


'Add to Cart' on the current web page. [] [StaticText] [text] means that the element is of some text that is not interactable.
'Add to Cart' 的按钮。[] [StaticText] [text] 表示该元素是不可交互的某些文本。


The current web page's URL: This is the page you're currently navigating.
当前网页的 URL：这是你当前正在浏览的页面。


The open tabs: These are the tabs you have open.
已打开的标签页：这些是您当前打开的标签页。


The previous action: This is the action you just performed. It may be helpful to track your progress.
上一步操作：这是您刚刚执行的操作，有助于跟踪进度。


The actions you can perform fall into several categories:
您可以执行的操作分为以下几类：


Page Operation Actions:
页面操作类：


```click [id]```: This action clicks on an element with a specific id on the webpage.
```click [id]```：点击网页中具有特定 id 的元素。


```type [id] [content]```: Use this to type the content into the field with id. By default, the "Enter" key is pressed after typing
```type [id] [content]```：在 id 对应的字段中输入内容。默认在输入后按“回车”键，


unless press_enter_after is set to 0 , i.e., ```type [id] [content] [0]```.
除非将 press_enter_after 设置为 0，即 ```type [id] [content] [0]```。


```hover [id]```: Howev over an element with id.
```hover [id]```：悬停在具有 id 的元素上。


```press [key_comb]```: Simulates the pressing of a key combination on the keyboard (e.g., Ctrl+v).
```press [key_comb]```：模拟键盘组合键的操作（例如 Ctrl+v）。


```scroll [down]``` or ```scroll [up]```: Scroll the page up or down.
```scroll [down]``` 或 ```scroll [up]```：向上或向下滚动页面。


## Tab Management Actions:
## 标签页管理类：


```new_tab```: Open a new, empty browser tab.
```new_tab```：打开一个新的空白浏览器标签页。


```tab_focus [tab_index]```: Switch the browser's focus to a specific tab using its index.
```tab_focus [tab_index]```：通过索引将浏览器焦点切换到特定标签页。


```close_tab```: Close the currently active tab.
```close_tab```：关闭当前活动的标签页。


URL Navigation Actions:
URL 导航类：


```goto [url]```: Navigate to a specific URL.
```goto [url]```：导航至特定的 URL。


```go_back```: Navigate to the previously viewed page.
```go_back```: 返回上一页。


```go_forward```: Navigate to the next page (if a previous 'go_back' action was performed).
```go_forward```: 前进到下一页（如果之前执行过“后退”操作）。


Completion Action:
完成操作：


```stop [answer]```: Issue this action when you believe the task is complete. If the objective is to find a text-based answer,
```stop [answer]```: 当你认为任务已完成时执行此操作。如果目标是查找文本答案，


provide the answer in the bracket.
请在括号内提供答案。


## Homepage:
## 首页：


If you want to visit other websites, check out the homepage at http://homepage.com.It has a list of websites you can visit.
如果你想访问其他网站，请查看首页 http://homepage.com。它包含你可以访问的网站列表。


http://homepage.com/password.html lists all the account name and password for the websites. You can use them to log in to the
http://homepage.com/password.html 列出了所有网站的账户名和密码。你可以使用它们登录


websites.
网站。


To be successful, it is very important to follow the following rules:
为了成功完成任务，遵循以下规则非常重要：


1. You should only issue an action that is valid given the current observation
1. 你只能发布在当前观察下有效的操作


2. You should only issue one action at a time.
2. 你一次只能发布一个操作。


3. You should follow the examples to reason step by step and then issue the next action.
3. 你应该参考示例进行逐步推理，然后发布下一个操作。


4. Generate the action in the correct format. Start with a "In summary, the next action I will perform is" phrase, followed by
4. 以正确的格式生成操作。以“In summary, the next action I will perform is”短语开头，后跟


action inside ``` `. For example, "In summary, the next action I will perform is ```click [1234]```".
操作置于 ``` 内。例如，“总而言之，我将执行的下一个操作是 ```click [1234]```”。


5. Issue stop action when you think you have achieved the objective. Don't generate anything after stop.
5. 当你认为已达成目标时，发出停止（stop）操作。停止后不要生成任何内容。


&nbsp;&nbsp;&nbsp;&nbsp;Figure 16: System message used in our SoM agent.
&nbsp;&nbsp;&nbsp;&nbsp;图 16：我们的 SoM 智能体中使用的系统消息。


---



---



example_user:
示例用户：


IMAGES: (1) current page screenshot
图像：(1) 当前页面截图


OBSERVATION:
观察：


[31] [IMG] [Image, description: hp fx-7010dn fax machine, url: B08GKZ3ZKD.0.jpg]
[31] [IMG] [图像，描述：hp fx-7010dn 传真机，url: B08GKZ3ZKD.0.jpg]


[32] [A] [HP CB782A#ABA 640 Inkjet Fax Machine (Renewed)]
[32] [A] [HP CB782A#ABA 640 喷墨传真机（翻新）]


[ ] [StaticText] [\$279.49]
[ ] [静态文本] [\$279.49]


&nbsp;&nbsp;&nbsp;&nbsp;[33] [BUTTON] [Add to Cart]
&nbsp;&nbsp;&nbsp;&nbsp;[33] [按钮] [加入购物车]


&nbsp;&nbsp;&nbsp;&nbsp;[34] [A] [Add to Wish List]
&nbsp;&nbsp;&nbsp;&nbsp;[34] [A] [加入收藏夹]


&nbsp;&nbsp;&nbsp;&nbsp;[35] [A] [Add to Compare]
&nbsp;&nbsp;&nbsp;&nbsp;[35] [A] [加入对比]


&nbsp;&nbsp;&nbsp;&nbsp;URL: http://onestopmarket.com/office-products/office-electronics.html
&nbsp;&nbsp;&nbsp;&nbsp;URL: http://onestopmarket.com/office-products/office-electronics.html


OBJECTIVE: What is the price of HP Inkjet Fax Machine?
目标：HP 喷墨传真机的价格是多少？


PREVIOUS ACTION: None
前一次操作：无


example_assistant: Let's think step-by-step. This page list the information of HP Inkjet Fax Machine, which is the product
示例助手：让我们逐步思考。此页面列出了 HP 喷墨传真机的信息，即产品


identified in the objective. Its price is \$279.49. I think I have achieved the objective. I will issue the stop action with the answer.
已在目标中识别。其价格为$279.49。我认为我已经达成了目标。我将发布带有答案的停止操作。


In summary, the next action I will perform is ' ' stop [\$279.49]' '
综上所述，我将执行的下一个动作是“ stop [$279.49] ”


example_user:
example_user:


IMAGES: (1) current page screenshot
图像：(1) 当前页面截图


OBSERVATION:
观测：


[ ] [StaticText] [/f/food]
[ ] [静态文本] [/f/food]


[ ] [StaticText] [[homemade] Obligatory Halloween Pumpkin Loaf! Submitted by kneechalice t3_yid9lu 1 year ago]
[ ] [静态文本] [[自制] 万圣节必备南瓜面包！由 kneechalice 提交于 t3_yid9lu 1年前]


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[9] [IMG] [ ]
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[9] [图片] [ ]


[ ] [StaticText] [Submitted by kneechalice t3_yid9lu 1 year ago]
[ ] [静态文本] [由 kneechalice 提交于 t3_yid9lu 1年前]


[10] [A] [kneechalice]
[10] [链接] [kneechalice]


[11] [A] [45 comments]
[11] [链接] [45 条评论]


[ ] [StaticText] [[I ate] Maple Pecan Croissant Submitted by AccordingtoJP t3_y3hrpn 1 year ago]
[ ] [静态文本] [[我吃了] 枫糖胡桃可颂 由 AccordingtoJP 提交于 t3_y3hrpn 1年前]


&nbsp;&nbsp;&nbsp;&nbsp;[14] [IMG] []
&nbsp;&nbsp;&nbsp;&nbsp;[14] [图片] []


[ ] [StaticText] [Submitted by AccordingtoJP t3_y3hrpn 1 year ago]
[ ] [静态文本] [由 AccordingtoJP 提交于 t3_y3hrpn 1年前]


&nbsp;&nbsp;&nbsp;&nbsp;[15] [A] [AccordingtoJP]
&nbsp;&nbsp;&nbsp;&nbsp;[15] [链接] [AccordingtoJP]


[16] [A] [204 comments]
[16] [链接] [204 条评论]


&nbsp;&nbsp;&nbsp;&nbsp;URL: http://reddit.com
&nbsp;&nbsp;&nbsp;&nbsp;URL: http://reddit.com


OBJECTIVE: Tell me what the top comment on the croissant post says.
目标：告诉我关于羊角面包帖子中最高赞评论的内容。


PREVIOUS ACTION: None
前序操作：无


example_assistant: Let's think step-by-step. This page has a post titled '[I ate] Maple Pecan Croissant', which is the post
示例助手：让我们逐步思考。此页面有一个标题为“[我吃了] 枫糖核桃羊角面包”的帖子，即


mentioned in the objective. In order to find the top comment, I will navigate into the comments section of the post. In summary,
目标中提到的帖子。为了找到最高赞评论，我将进入该帖子的评论区。总之，


the next action I will perform is ' ' 'click [11]' '
我将执行的下一步操作是'''点击 [11]'''


example_user:
示例用户：


IMAGES: (1) current page screenshot
图像：(1) 当前页面截图


OBSERVATION:
观察：


&nbsp;&nbsp;&nbsp;&nbsp;[ ] [StaticText] [What are you looking for today?]
&nbsp;&nbsp;&nbsp;&nbsp;[ ] [静态文本] [你今天在寻找什么？]


&nbsp;&nbsp;&nbsp;&nbsp;[5] [INPUT] []
&nbsp;&nbsp;&nbsp;&nbsp;[5] [输入框] []


&nbsp;&nbsp;&nbsp;&nbsp;[6] [SELECT] [Select a category]
&nbsp;&nbsp;&nbsp;&nbsp;[6] [选择框] [选择类别]


&nbsp;&nbsp;&nbsp;&nbsp;[7] [BUTTON] [Search]
&nbsp;&nbsp;&nbsp;&nbsp;[7] [按钮] [搜索]


&nbsp;&nbsp;&nbsp;&nbsp;[ ] [StaticText] [Latest Listings]
&nbsp;&nbsp;&nbsp;&nbsp;[ ] [静态文本] [最新房源/信息]


[ ] [StaticText] [Atlas Powered Audio System w/ Tripod 150.00 \$ Music instruments Borough of Red Lion (Pennsylvania)
[ ] [静态文本] [Atlas 动力音频系统（带三脚架）150.00 \$ 乐器，红狮镇（宾夕法尼亚州）]


2023/11/16]



[8] [IMG] [Atlas Powered Audio System w/ Tripod]
[8] [图像] [Atlas 动力音频系统（带三脚架）]


[9] [A] [Atlas Powered Audio System w/ Tripod]
[9] [A] [Atlas 有源音响系统（含三脚架）]


[ ] [StaticText] [150.00 \$]
[ ] [StaticText] [150.00 \$]


[ ] [StaticText] [Neptune Gaming Console 350.00 \$ Video gaming Pennwyn (Pennsylvania) 2023/11/16]
[ ] [StaticText] [Neptune 游戏机 350.00 \$ 视频游戏 宾夕法尼亚州 2023/11/16]


[10] [IMG] [Neptune Gaming Console]
[10] [IMG] [Neptune 游戏机]


[11] [A] [Neptune Gaming Console]
[11] [A] [Neptune 游戏机]


&nbsp;&nbsp;&nbsp;&nbsp;[ ] [StaticText] [350.00 \$]
&nbsp;&nbsp;&nbsp;&nbsp;[ ] [StaticText] [350.00 \$]


URL: http://classifieds.com
URL: http://classifieds.com


OBJECTIVE: Help me find the cheapest dark colored guitar.
目标：帮我找到最便宜的深色吉他。


PREVIOUS ACTION: None
上一步操作：无


example_assistant: Let's think step-by-step. The objective is to find the cheapest dark colored guitar on the site. The site has a
示例助教：让我们逐步思考。目标是在网站上找到最便宜的深色吉他。该网站有一个


search box whose ID is [5]. I can search for guitars by entering "guitar". I can submit this by pressing the Enter afterwards. In
ID 为 [5] 的搜索框。我可以输入 "guitar" 来搜索吉他。之后可以按回车键提交。总


summary, the next action I will perform is ' ' 'type [5] [guitar] [1] ' '
结来说，我将执行的下一步操作是 ' ' 'type [5] [guitar] [1] ' '


---



Figure 17: In-context examples and prompt used for our SoM agents. The webpage screenshots with SoM are added to the end of each round of the example_user dialogue.
图 17：SoM 智能体使用的上下文示例和提示词。带有 SoM 的网页截图被添加在每一轮示例用户对话的末尾。