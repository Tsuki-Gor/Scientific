# Open-Ended Instructable Embodied Agents with Memory-Augmented Large Language Models
# 具备记忆增强大语言模型的开放式可操作具身代理


Gabriel Sarch Yue Wu Michael J. Tarr Katerina Fragkiadaki
Gabriel Sarch Yue Wu Michael J. Tarr Katerina Fragkiadaki


Carnegie Mellon University
卡内基梅隆大学


\{gsarch,ywu5,mt01\}@andrew.cmu.edu,katef@cs.cmu.edu
\{gsarch,ywu5,mt01\}@andrew.cmu.edu,katef@cs.cmu.edu


helper-agent-llm.github.io
helper-agent-llm.github.io


## Abstract
## 摘要


Pre-trained and frozen LLMs can effectively map simple scene re-arrangement instructions to programs over a robot's visuomotor functions through appropriate few-shot example prompting. To parse open-domain natural language and adapt to a user's idiosyncratic procedures, not known during prompt engineering time, fixed prompts fall short. In this paper, we introduce HELPER, an embodied agent equipped with as external memory of language-program pairs that parses free-form human-robot dialogue into action programs through retrieval-augmented LLM prompting: relevant memories are retrieved based on the current dialogue, instruction, correction or VLM description, and used as in-context prompt examples for LLM querying. The memory is expanded during deployment to include pairs of user's language and action plans, to assist future inferences and personalize them to the user's language and routines. HELPER sets a new state-of-the-art in the TEACh benchmark in both Execution from Dialog History (EDH) and Trajectory from Dialogue (TfD), with 1.7x improvement over the previous SOTA for TfD. Our models, code and video results can be found in our project's website: helper-agent-llm.github.io.
预训练且冻结的 LLM 通过恰当的少样本示例提示，能够有效地将简单的场景重排指令映射为机器人视觉运动功能上的程序。要解析开放领域的自然语言，并适配用户在提示工程阶段未知的个人化流程，固定提示就力有未逮。本文提出 HELPER，这是一种具备外部语言-程序对记忆的具身代理：它通过检索增强的 LLM 提示，将自由形式的人机对话解析为动作程序。系统会基于当前对话、指令、纠正或 VLM 描述检索相关记忆，并将其作为上下文提示示例供 LLM 查询。部署过程中，记忆会持续扩展，以纳入用户语言与行动计划的配对，从而帮助未来推断，并将其个性化到用户的语言与日常习惯。HEPER 在 TEACh 基准的对话执行（EDH）和对话轨迹生成（TfD）两项上均达到新的最先进水平：相较此前 TfD 的 SOTA，提升 1.7 倍。我们的模型、代码和视频结果可在项目网站查看：helper-agent-llm.github.io。


## 1 Introduction
## 1 引言


Parsing free-form human instructions and human-robot dialogue into task plans that a robot can execute is challenging due to the open-endedness of environments and procedures to accomplish, and to the diversity and complexity of language humans use to communicate their desires. Human language often contains long-term references, questions, errors, omissions, or references and descriptions of routines specific to a particular user (Tellex et al., 2011; Liang, 2016; Klein and Manning, 2003). Instructions need to be interpreted in the environmental context in which they are issued, and plans need to adapt in a closed-loop to execution failures.
将自由形式的人类指令和人机对话解析为机器人可执行的任务计划，具有挑战性，因为环境和完成这些任务的流程具有开放性，而人类用于表达意图的语言也多样而复杂。自然语言中常包含长期指代、问题、错误、遗漏，或针对特定用户的惯例性指代和描述（Tellex et al., 2011; Liang, 2016; Klein and Manning, 2003）。指令需要结合其给出的环境上下文来理解，计划也需要在执行失败后以闭环方式自适应调整。


Large Language Models (LLMs) trained on Internet-scale text can parse language instructions to task plans with appropriate plan-like or code-like prompts, without any finetuning of the language model, as shown in recent works (Ahn et al., 2022; Liang et al., 2022; Zeng et al., 2022; Huang et al., 2022b; Singh et al., 2022a). The state of the environment is provided as a list of objects and their spatial coordinates, or as a free-form text description from a vision-language model (Liang et al., 2022; Liu et al., 2023b; Wu et al., 2023a; Ahn et al., 2022). Using LLMs for task planning requires engineering a prompt that includes a description of the task for the LLM to perform, a robot API with function documentation and expressive function names, environment and task instruction inputs, and a set of in-context examples for inputs and outputs for the task (Liang et al., 2022). These methods are not trained in the domain of interest; rather they are prompt-engineered having in mind the domain at hand.
近期工作表明，大语言模型（LLMs）在互联网规模文本上训练后，无需对语言模型进行任何微调，只需使用合适的类计划或类代码提示，就能将语言指令解析为任务计划（Ahn et al., 2022; Liang et al., 2022; Zeng et al., 2022; Huang et al., 2022b; Singh et al., 2022a）。环境状态以物体及其空间坐标的列表形式提供，或由视觉语言模型给出的自由形式文本描述提供（Liang et al., 2022; Liu et al., 2023b; Wu et al., 2023a; Ahn et al., 2022）。将LLMs用于任务规划，需要设计一个提示，其中包括供LLM执行的任务描述、带函数文档和具表现力的函数名的机器人API、环境与任务指令输入，以及用于任务输入和输出的一组上下文示例（Liang et al., 2022）。这些方法并非在目标领域上训练；相反，它们是在提示工程中针对当前领域量身设计的。


How can we extend LLM-prompting for semantic parsing and task planning to open-domain, freeform instructions, corrections, human-robot dialogue, and users' idiosyncratic routines, not known at prompt engineering time? The prompts used for the domain of tabletop rearrangement are already approaching the maximum context window of widely used LLMs (Singh et al., 2022a; Liang et al., 2022). Even as context window size grows, more prompt examples result in larger attention operations and cause an increase in both inference time and resource usage.
我们如何将用于语义解析和任务规划的LLM提示扩展到开放领域、自由形式指令、更正、人机对话，以及在提示工程时并未知晓的用户个性化惯例？用于桌面重排领域的提示已接近广泛使用的LLMs的最大上下文窗口（Singh et al., 2022a; Liang et al., 2022）。即使上下文窗口继续增大，更多的提示示例也会带来更大的注意力计算，并导致推理时间和资源消耗增加。


To this end, we introduce HELPER (Human-instructable Embodied Language Parsing via Evolving Routines), a model that uses retrieval-augmented situated prompting of LLMs to parse free-form dialogue, instructions and corrections from humans and vision-language models to programs over a set of parameterized visuomotor routines. HELPER is equipped with an external nonparametric key-value memory of language - program pairs. HELPER uses its memory to retrieve relevant in-context language and action program examples, and generates prompts tailored to the current language input. HELPER expands its memory with successful executions of user specific procedures; it then recalls them and adapts them in future interactions with the user. HELPER uses pre-trained vision-language models (VLMs) to diagnose plan failures in language format, and uses these to retrieve similar failure cases with solutions from its memory to seed the prompt. To execute a program predicted by the LLM, HELPER combines successful practices of previous home embodied agents, such as semantic and occupancy map building (Chaplot et al., 2020a; Blukis et al., 2022; Min et al., 2021), LLM-based common sense object search (Inoue and Ohashi, 2022), object detection and tracking with off-the-shelf detectors (Chap-lot et al., 2020b), object attribute detection with VLMs (Zhang et al., 2022), and verification of action preconditions during execution.
为此，我们提出HELPER（Human-instructable Embodied Language Parsing via Evolving Routines），一种利用检索增强的情境提示来解析人类和视觉语言模型给出的自由形式对话、指令与更正，并将其转换为一组参数化视听运动例程上的程序的模型。HELPER配备了一个外部的非参数化语言-程序对键值记忆。HELPER利用其记忆检索相关的上下文内语言和动作程序示例，并生成针对当前语言输入的提示。HELPER通过成功执行用户特定流程来扩展其记忆；随后在未来与该用户的交互中回忆并调整这些流程。HELPER使用预训练的视觉语言模型（VLMs）以语言形式诊断计划失败，并据此检索记忆中带有解决方案的相似失败案例来为提示提供种子。为了执行LLM预测的程序，HELPER结合了以往家庭具身智能体的成功做法，例如语义和占用地图构建（Chaplot et al., 2020a; Blukis et al., 2022; Min et al., 2021）、基于LLM的常识性物体搜索（Inoue and Ohashi, 2022）、使用现成检测器进行物体检测与跟踪（Chap-lot et al., 2020b）、使用VLMs进行物体属性检测（Zhang et al., 2022），以及执行过程中动作前提条件的验证。


<img src="https://cdn.noedgeai.com/bo_d8j3q42lb0pc73bdldrg_1.jpg?x=182&y=186&w=1285&h=639&r=0"/>



Figure 1: Open-ended instructable agents with retrieval-augmented LLMs. We equip LLMs with an external memory of language and program pairs to retrieve in-context examples for prompts during LLM querying for task plans. Our model takes as input instructions, dialogue segments, corrections and VLM environment descriptions, retrieves relevant memories to use as in-context examples, and prompts LLMs to predict task plans and plan adjustments. Our agent executes the predicted plans from visual input using occupancy and semantic map building, 3D object detection and state tracking, and active exploration using guidance from LLMs' common sense to locate objects not present in the maps. Successful programs are added to the memory paired with their language context, allowing for personalized subsequent interactions.
图1：结合检索增强LLMs的开放式可指令智能体。我们为LLMs配备了一个语言和程序对的外部记忆，以便在LLM查询任务计划时检索上下文示例作为提示。我们的模型将指令、对话片段、更正以及VLM环境描述作为输入，检索相关记忆以用作上下文示例，并提示LLMs预测任务计划及其调整。我们的智能体利用视觉输入，通过构建占用图和语义图、进行3D物体检测和状态跟踪，以及借助LLMs常识指导进行主动探索来定位地图中不存在的物体，从而执行预测的计划。成功的程序会与其语言上下文一起加入记忆，从而支持后续的个性化交互。


We test HELPER on the TEACh benchmark (Pad-makumar et al., 2021), which evaluates agents in their ability to complete a variety of long-horizon household tasks from RGB input given natural language dialogue between a commander (the instruction-giving user) and a follower (the instruction-seeking user). We achieve a new state-of-the-art in the TEACh Execution from Dialog History and Trajectory-from-Dialogue settings, improving task success by ${1.7}\mathrm{x}$ and goal-condition success by 2.1x compared to prior work in TfD. By further soliciting and incorporating user feedback, HELPER attains an additional 1.3x boost in task success. Our work is inspired by works in the language domain (Perez et al., 2021; Schick and Schütze, 2020; Gao et al., 2020; Liu et al., 2021) that retrieve in-context prompt examples based on the input language query for NLP tasks. HELPER extends this capability to the domain of instructable embodied agents, and demonstrates the potential of memory-augmented LLMs for semantic parsing of open-ended free-form instructive language into an expandable library of programs.
我们在 TEACh 基准（Pad-makumar 等，2021）上测试 HELPER，该基准评估智能体在仅凭 RGB 输入、并在指挥者（给出指令的用户）与跟随者（寻求指令的用户）之间的自然语言对话条件下，完成各种长时域家务任务的能力。相较于 TfD 领域的既有工作，我们在 TEACh 的“基于对话历史的执行（Execution from Dialog History）”与“基于对话轨迹的生成（Trajectory-from-Dialogue）”设置中取得了新的最先进结果：任务成功提升 ${1.7}\mathrm{x}$，目标条件达成成功提升 2.1 倍。通过进一步征询并整合用户反馈，HELPER 在任务成功上又获得了额外 1.3 倍的增益。我们的工作受到语言领域研究的启发（Perez 等，2021；Schick 和 Schütze，2020；Gao 等，2020；Liu 等，2021）：这些研究会根据输入语言查询，为 NLP 任务检索上下文提示示例。HELPER 将这一能力扩展到可指令的具身智能体领域，并展示了记忆增强的大语言模型用于将开放式、自由形式的指令性语言进行语义解析，从而映射到可扩展的程序库的潜力。


## 2 Related Work
## 2 相关工作


Instructable Embodied Agents Significant strides have been made by training large neural networks to jointly map instructions and their sensory contexts to agent actions or macro-actions using imitation learning (Anderson et al., 2018b; Ku et al., 2020; Anderson et al., 2018a; Savva et al., 2019; Gervet et al., 2022; Shridhar et al., 2020; Cao et al.; Suglia et al., 2021; Fan et al., 2018; Yu et al., 2020; Brohan et al., 2022; Stone et al., 2023; Yu et al., 2023). Existing approaches differ—among others-in the way the state of the environment is communicated to the model. Many methods map RGB image tokens and language inputs directly to actions or macro-actions (Pashevich et al., 2021; Wijmans et al., 2020; Suglia et al., 2021; Krantz et al., 2020). Other methods map language instructions and linguistic descriptions of the environment's state in terms of object lists or objects spatial coordinates to macro-actions, foregoing visual feature description of the scene, in a attempt to generalize better (Liang et al., 2022; Singh et al., 2022a; Chaplot et al., 2020a; Min et al., 2021; Liu et al., 2022a; Murray and Cakmak, 2022; Liu et al., 2022b; Inoue and Ohashi, 2022; Song et al., 2022; Zheng et al., 2022; Zhang et al., 2022; Huang et al., 2022b, 2023; Ahn et al., 2022; Zeng et al., 2022; Huang et al., 2022a). Some of these methods fine-tune language models to map language input to macro-actions, while others prompt frozen LLMs to predict action programs, relying on the emergent in-context learning property of LLMs to emulate novel tasks at test time. Some methods use natural language as the output format of the LLM (Wu et al., 2023a; Song et al., 2022; Blukis et al., 2022; Huang et al., 2022b), and others use code format (Singh et al., 2022a; Liang et al., 2022; Huang et al., 2023). HELPER prompts frozen LLMs to predict Python programs over visuo-motor functions for parsing dialogue, instructions and corrective human feedback.
可指令具身代理 通过训练大型神经网络，使其能够借助模仿学习，将指令与感知语境共同映射到智能体动作或宏动作（Anderson et al., 2018b; Ku et al., 2020; Anderson et al., 2018a; Savva et al., 2019; Gervet et al., 2022; Shridhar et al., 2020; Cao et al.; Suglia et al., 2021; Fan et al., 2018; Yu et al., 2020; Brohan et al., 2022; Stone et al., 2023; Yu et al., 2023），取得了显著进展。现有方法的差异在于——例如——环境状态如何被传达给模型。许多方法将 RGB 图像特征标记与语言输入直接映射为动作或宏动作（Pashevich et al., 2021; Wijmans et al., 2020; Suglia et al., 2021; Krantz et al., 2020）。还有一些方法将语言指令以及环境状态的语言描述（用物体列表或物体的空间坐标来表述）映射为宏动作，而不对场景进行视觉特征描述，以便在泛化上做得更好（Liang et al., 2022; Singh et al., 2022a; Chaplot et al., 2020a; Min et al., 2021; Liu et al., 2022a; Murray and Cakmak, 2022; Liu et al., 2022b; Inoue and Ohashi, 2022; Song et al., 2022; Zheng et al., 2022; Zhang et al., 2022; Huang et al., 2022b, 2023; Ahn et al., 2022; Zeng et al., 2022; Huang et al., 2022a）。其中一些方法微调语言模型，将语言输入映射到宏动作；而另一些方法通过提示冻结的 LLM 来预测动作程序，利用 LLM 的涌现式上下文学习能力，在测试时模拟新任务。一些方法使用自然语言作为 LLM 的输出格式（Wu et al., 2023a; Song et al., 2022; Blukis et al., 2022; Huang et al., 2022b），另一些则使用代码格式（Singh et al., 2022a; Liang et al., 2022; Huang et al., 2023）。HELPER 用提示词驱动冻结的 LLM，在视觉-运动功能上预测 Python 程序，用于解析对话、指令以及纠正性的人工反馈。


The work closest to HELPER is LLM Planner (Song et al., 2022) which uses memory-augmented prompting of pretrained LLMs for instruction following. However, it differs from HELPER in several areas such as plan memory expansion, VLM-guided correction, and usage of LLMs for object search. Furthermore, while Singh et al. (2022b) frequently seeks human feedback, HELPER requests feedback only post full task execution and employs Visual-Language Models (VLMs) for error feedback, reducing user interruptions.
与 HELPER 最接近的工作是 LLM Planner（Song et al., 2022），它通过带记忆增强的提示来引导预训练 LLM 完成指令跟随。然而，HEL PER 在多个方面与之不同，例如计划记忆的扩展、VLM 引导的纠错，以及使用 LLM 进行物体搜索。此外，尽管 Singh et al.（2022b）经常寻求人类反馈，HELPER 只在完整任务执行之后才请求反馈，并使用视觉-语言模型（VLM）进行错误反馈，从而减少用户中断。


Numerous simulation environments exist for evaluating home assistant frameworks, including Habitat (Savva et al., 2019), GibsonWorld (Shen et al., 2021), ThreeDWorld (Gan et al., 2022), and AI2THOR (Kolve et al., 2017). ALFRED (Shrid-har et al., 2020) and TEACh (Padmakumar et al., 2021) are benchmarks in the AI2THOR environment (Kolve et al., 2017), measuring agents' competence in household tasks through natural language. Our research focuses on the 'Trajectory from Dialogue' (TfD) evaluation in TEACh, mirroring ALFRED but with greater task and input complexity.
用于评估家庭助理框架的模拟环境有很多，包括 Habitat（Savva et al., 2019）、GibsonWorld（Shen et al., 2021）、ThreeDWorld（Gan et al., 2022）和 AI2THOR（Kolve et al., 2017）。ALFRED（Shrid-har et al., 2020）和 TEACh（Padmakumar et al., 2021）是在 AI2THOR 环境中的基准（Kolve et al., 2017），通过自然语言来衡量智能体在家庭任务中的能力。我们的研究聚焦于 TEACh 中的“对话生成轨迹”（Trajectory from Dialogue, TfD）评测：它与 ALFRED 类似，但任务与输入复杂度更高。


<img src="https://cdn.noedgeai.com/bo_d8j3q42lb0pc73bdldrg_2.jpg?x=836&y=185&w=628&h=399&r=0"/>



Figure 2: HELPER's architecture. The model uses memory-augmented LLM prompting for task planning from instructions, corrections and human-robot dialogue and for re-planning during failures given feedback from a VLM model. The generated program is executed the EXECUTOR module. The Executor builds semantic, occupancy and 3D object maps, tracks object states, verifies action preconditions, and queries LLMs for search locations for objects missing from the maps, using the LOCATOR module.
图 2：HELPER 的架构。该模型使用带记忆增强的 LLM 提示，根据指令、纠错和人机对话进行任务规划；在失败发生并获得来自 VLM 模型的反馈时，还会进行重新规划。生成的程序由 EXECUTOR 模块执行。Executor 构建语义、占据信息和 3D 物体地图，跟踪物体状态，验证动作前提，并在地图中缺失物体时通过 LOCATOR 模块向 LLM 查询物体的搜索位置。


Prompting LLMs for action prediction and visual reasoning Since the introduction of few-shot prompting by (Brown et al., 2020), several approaches have improved the prompting ability of LLMs by automatically learning prompts (Lester et al., 2021), chain of thought prompting (Nye et al.; Gao et al., 2022; Wei et al., 2022; Wang et al., 2022; Chen et al., 2022; Yao et al., 2023) and retrieval-augmented LLM prompting (Nakano et al., 2021; Shi et al., 2023; Jiang et al., 2023) for language modeling, question answering, and long-form, multi-hop text generation. HELPER uses memory-augmented prompting by retrieving and integrating similar task plans into the prompt to facilitate language parsing to programs.
用于动作预测与视觉推理的LLM提示自从（Brown et al., 2020）提出少样本提示以来，已有多种方法通过自动学习提示（Lester et al., 2021）、链式思维提示（Nye et al.; Gao et al., 2022; Wei et al., 2022; Wang et al., 2022; Chen et al., 2022; Yao et al., 2023）以及检索增强的LLM提示（Nakano et al., 2021; Shi et al., 2023; Jiang et al., 2023）来提升语言建模、问答和长篇、多跳文本生成的提示能力。HELPER通过检索并将相似的任务计划整合到提示中来进行记忆增强提示，从而促进将语言解析为程序。


LLMs have been used as policies in Minecraft to predict actions (Wang et al., 2023b,a), error correction (Liu et al., 2023b), and for understanding instruction manuals for game play in some Atari games (Wu et al., 2023b). They have also significantly improved text-based agents in text-based simulated worlds (Yao et al., 2022; Shinn et al., 2023; Wu et al., 2023c; Richards, 2023). ViperGPT (Surís et al., 2023), and CodeVQA (Sub-ramanian et al., 2023) use LLM prompting to decompose referential expressions and questions to programs over simpler visual routines. Our work uses LLMs for planning from free-form dialogue and user corrective feedback for home task completion, a domain not addressed in previous works.
LLM已在Minecraft中被用作策略，用于预测动作（Wang et al., 2023b,a）、进行错误纠正（Liu et al., 2023b），并在部分Atari游戏中通过理解游戏操作的说明手册来辅助游戏（Wu et al., 2023b）。它们也显著提升了基于文本的模拟世界中的文本代理（Yao et al., 2022; Shinn et al., 2023; Wu et al., 2023c; Richards, 2023）。ViperGPT（Surís et al., 2023）和CodeVQA（Sub-ramanian et al., 2023）使用LLM提示，将指称表达和问题分解为更简单视觉例程之上的程序。我们的工作则使用LLM基于自由形式对话和用户纠错反馈进行规划，以完成家庭任务——这是先前研究未涉及的领域。


## 3 Method
## 3 方法


HELPER is an embodied agent designed to map human-robot dialogue, corrections and VLM descriptions to actions programs over a fixed API of parameterized navigation and manipulation primitives. Its architecture is outlined in Figure 2. At its heart, it generates plans and plan adjustments by querying LLMs using retrieval of relevant language-program pairs to include as in-context examples in the LLM prompt. The generated programs are then sent to the EXECUTOR module, which translates each program step into specific navigation and manipulation action. Before executing each step in the program, the EXECUTOR verifies if the necessary preconditions for an action, such as the robot already holding an object, are met. If not, the plan is adjusted according to the current environmental and agent state. Should a step involve an undetected object, the EXECUTOR calls on the LOCATOR module to efficiently search for the required object by utilizing previous user instructions and LLMs' common sense knowledge. If any action fails during execution, a VLM predicts the reason for this failure from pixel input and feeds this into the PLANNER for generating plan adjustments.
HELPER 是一种具身智能体，旨在将人机对话、纠正信息以及 VLM 描述映射为基于固定 API 的动作程序，API 由参数化的导航与操作原语构成。其架构如图 2 所示。核心在于：通过查询 LLM，并检索相关的语言-程序对，将其作为情境示例纳入 LLM 提示，从而生成计划及对计划的调整。随后，生成的程序会被发送至执行器（EXECUTOR）模块，该模块将程序的每一步转换为具体的导航与操作动作。在执行程序中的每一步之前，EXECUTOR 会检查执行某项动作所需的前提条件是否满足，例如机器人是否已持有某个物体。若不满足，则会根据当前环境与智能体状态对计划进行调整。如果某一步涉及未被检测到的物体，EXECUTOR 会调用定位器（LOCATOR）模块，以利用先前的用户指令及 LLM 的常识知识高效搜索所需物体。若执行过程中任何动作失败，VLM 将根据像素输入预测失败原因，并将其反馈给规划器（PLANNER），以生成计划调整。


### 3.1 PLANNER: Retrieval-Augmented LLM Planning
### 3.1 PLANNER：检索增强的LLM规划


Given an input $I$ consisting of a dialogue segment, instruction, or correction, HELPER uses memory-augmented prompting of frozen LLMs to map the input into an executable Python program over a parametrized set of manipulation and navigation primitives $G \in  \left\{  {{G}_{\text{ manipulation }} \cup  {G}_{\text{ navigation }}}\right\}$ that the EXECUTOR can perform (e.g., goto(X), pickup(X), slice(X), ...). Our action API can be found in Section D of the Appendix.
给定一个由对话片段、指令或修正组成的输入$I$，HELPER通过对冻结LLM进行记忆增强提示，将输入映射为一个可执行的Python程序，该程序基于一组参数化的操作与导航原语$G \in  \left\{  {{G}_{\text{ manipulation }} \cup  {G}_{\text{ navigation }}}\right\}$，由EXECUTOR执行（例如，goto(X)、pickup(X)、slice(X)等）。我们的动作API见附录D节。


HELPER maintains a key-value memory of language - program pairs, as shown in Figure 3A. Each language key is mapped to a 1D vector using an LLM's frozen language encoder. Given current context $I$ ,the model retrieves the top- $K$ keys,i.e., the keys with the smallest ${L}_{2}$ distance with the embedding of the input context $I$ ,and adds the corresponding language - program pairs to the LLM prompt as in-context examples for parsing the current input $I$ .
HELPER维护一个语言-程序对的键值记忆，如图3A所示。每个语言键都通过LLM冻结的语言编码器映射为一个1D向量。给定当前上下文$I$，模型检索前$K$个键，即与输入上下文$I$的嵌入距离最小的键${L}_{2}$，并将相应的语言-程序对作为上下文示例加入LLM提示中，以解析当前输入$I$。


Figure 3B illustrates the prompt format for the PLANNER. It includes the API specifying the primitives $G$ parameterized as Python functions, the retrieved examples,and the language input $I$ . The LLM is tasked to generate a Python program over parameterized primitives $G$ . Examples of our prompts and LLM responses can be found in Section F of the Appendix.
图3B展示了PLANNER的提示格式。它包括指定原语$G$的API，这些原语以Python函数形式参数化、检索到的示例，以及语言输入$I$。LLM的任务是在参数化原语之上生成一个Python程序$G$。我们的提示和LLM响应示例见附录F节。


#### 3.1.1 Memory Expansion
#### 3.1.1 记忆扩展


The key-value memory of HELPER can be continually expanded with successful executions of instructions to adapt to a user's specific routines, as shown in Figure 1. An additional key-value pair is added with the language instruction paired with the execution plan if the user indicates the task was successful. Then, HELPER can recall this plan and adapt it in subsequent interactions with the user. For example, if a user instructs HELPER one day to "Perform the Mary cleaning. This involves cleaning two plates and two cups in the sink", the user need only say "Do the Mary cleaning" in future interactions, and HELPER will retrieve the previous plan, include it in the examples section of the prompt, and query the LLM to adapt it accordingly. The personalization capabilities of HELPER are evaluated in Section 4.4.
HELPER 的键值记忆可以通过成功执行指令而不断扩展，以适应用户的具体习惯，如图 1 所示。若用户表明该任务成功，则会把“语言指令-执行计划”这一组键值对新增到记忆中。随后，HELPER 能够回忆该计划，并在与用户后续互动中进行相应调整。例如，如果用户有一天指示 HELPER“进行玛丽清洁。这包括清洗水槽里的两个盘子和两个杯子”，那么在未来的互动中，用户只需说“做玛丽清洁”，HELPER 就会检索先前的计划，把它纳入提示词的示例部分，并向 LLM 询问据此进行调整。HELPER 的个性化能力在第 4.4 节进行评估。


#### 3.1.2 Incorporating user feedback
#### 3.1.2 融入用户反馈


A user's feedback can improve a robot's performance, but requesting feedback frequently can deteriorate the overall user experience. Thus, we enable HELPER to elicit user feedback only when it has completed execution of the program. Specifically, it asks "Is the task completed to your satisfaction? Did I miss anything?" once it believes it has completed the task. The user responds either that the task has been completed (at which point HELPER stops acting) or points out problems and corrections in free-form natural language, such as, "You failed to cook a slice of potato. The potato slice needs to be cooked.". HELPER uses the language feedback to re-plan using the PLANNER. We evaluate HELPER in its ability to seek and utilize user feedback in Section 4.3.
用户的反馈可以提升机器人的性能，但频繁索取反馈会削弱整体的用户体验。因此，我们让 HELPER 仅在其完成程序执行后才征求用户反馈。具体来说，当它认为任务已完成时，会问：“任务是否已按你的满意度完成？我还有遗漏的吗？”。用户要么回复任务已完成（此时 HELPER 停止行动），要么以自由形式的自然语言指出问题并提出纠正，例如：“你没有把一片土豆做熟。土豆片需要被煮熟。” HELPER 会利用这种语言反馈，通过 PLANNER 重新规划。我们在第 4.3 节评估 HELPER 获取并利用用户反馈的能力。


<img src="https://cdn.noedgeai.com/bo_d8j3q42lb0pc73bdldrg_4.jpg?x=186&y=182&w=1287&h=494&r=0"/>



Figure 3: HELPER parses dialogue segments, instructions, and corrections into visuomotor programs using retrieval-augmented LLM prompting. A. Illustration of the encoding and memory retrieval process. B. Prompt format and output of the Planner.
图 3：HElPER 使用检索增强的 LLM 提示，将对话片段、指令和纠正解析为视觉运动程序。A. 编码与记忆检索过程示意。B. 规划器的提示格式与输出。


#### 3.1.3 Visually-Grounded Plan Correction using Vision-Language Models
#### 3.1.3 使用视觉-语言模型的视觉引导式计划修正


Generated programs may fail for various reasons, such as when a step in the plan is missed or an object-of-interest is occluded. When the program fails, HELPER uses a vision-language model (VLM) pre-trained on web-scale data, specifically the ALIGN model (Jia et al., 2021), to match the current visual observation with a pre-defined list of textual failure cases, such as an object is blocking you from interacting with the selected object, as illustrated in Figure 4. The best match is taken to be the failure feedback $F$ . The PLANNER module then retrieves the top- $K$ most relevant error correction examples, each containing input dialogue, failure feedback, and the corresponding corrective program, from memory based on encodings of input $I$ and failure feedback $F$ from the VLM. The LLM is prompted with the the failed program step, the predicted failure description $F$ from the VLM, the in-context examples, and the original dialogue segment $I$ . The LLM outputs a self-reflection as to why the failure occurred, and generates a program over manipulation and navigation primitives $G$ ,and an additional set of corrective primitives ${G}_{\text{ corrective }}$ (e.g.,step-back(),move-to-an-alternate-viewpoint()), ...). This program is sent to the ExECUTOR for execution.
生成的程序可能因多种原因失败，例如计划中的某一步被遗漏，或关注的目标被遮挡。当程序失败时，HELPER 使用一个在大规模网页数据上预训练的视觉-语言模型（VLM），具体为 ALIGN 模型（Jia et al., 2021），将当前的视觉观测与一份预定义的文本故障案例列表进行匹配，例如：某个物体阻挡了你与所选目标进行交互，如图 4 所示。最佳匹配对应的故障反馈为 $F$ 。随后，PLANNER 模块基于来自 VLM 的对输入 $I$ 与故障反馈 $F$ 的编码，从记忆中检索前 $K$ 个最相关的错误修正示例；每个示例都包含输入对话、故障反馈以及相应的纠正程序。LLM 的提示中包括失败程序的步骤、VLM 预测的故障描述 $F$、上下文示例以及原始对话片段 $I$ 。LLM 会输出关于失败为何发生的自我反思，并生成一个针对操作与导航原语的程序 $G$ ，以及一组额外的纠正原语 ${G}_{\text{ corrective }}$（例如，step-back(),move-to-an-alternate-viewpoint()), ...）。该程序随后被发送给 EXECUTOR 进行执行。


<img src="https://cdn.noedgeai.com/bo_d8j3q42lb0pc73bdldrg_4.jpg?x=843&y=839&w=620&h=180&r=0"/>



Figure 4: Inference of a failure feedback description by matching potential failure language descriptions with the current image using a vision-language model (VLM).
图 4：使用视觉-语言模型（VLM），将潜在的故障语言描述与当前图像进行匹配，从而推断故障反馈描述。


### 3.2 Executor: Scene Perception, Pre-Condition Checks, Object Search and Action Execution
### 3.2 执行器：场景感知、前置条件检查、物体搜索与动作执行


The Executor module executes the predicted Python programs in the environment, converting the code into low-level manipulation and navigation actions, as shown in Figure 2. At each time step, the EXECUTOR receives an RGB image and obtains an estimated depth map via monocular depth estimation (Bhat et al., 2023) and object masks via an off-the-shelf object detector (Dong et al., 2021).
执行器模块在环境中执行预测的 Python 程序，将代码转换为低层级的操作与导航动作，如图 2 所示。每个时间步，执行器接收一幅 RGB 图像，并通过单目深度估计（Bhat et al., 2023）获取估计深度图，通过现成的目标检测器（Dong et al., 2021）获取物体掩码。


#### 3.2.1 Scene and object state perception
#### 3.2.1 场景与目标状态感知


Using the depth maps, object masks, and approximate egomotion of the agent at each time step, the EXECUTOR maintains a 3D occupancy map and object memory of the home environment to navigate around obstacles and keep track of previously seen objects, similar to previous works (Sarch et al., 2022). Objects are detected in every frame and are merged into object instances based on closeness of the predicted 3D centroids. Each object instance is initialized with a set of object state attributes (cooked, sliced, dirty, ...) by matching the object crop against each attribute with the pre-trained ALIGN model (Jia et al., 2021). Object attribute states are updated when an object is acted upon via a manipulation action.
通过使用深度图、目标掩膜以及每个时间步智能体的近似自运动估计，执行器（EXECUTOR）维护家庭环境的三维占据地图和目标记忆，以便绕开障碍并追踪先前看到的物体，方式类似于以往工作（Sarch 等，2022）。每一帧都会检测目标，并根据预测的三维质心之间的接近程度将其合并为具体的目标实例。每个目标实例都会借助预训练的 ALIGN 模型（Jia 等，2021）将目标裁剪区域与各个属性进行匹配，从而初始化一组目标状态属性（已烹饪、已切片、脏污……）。当物体在操控动作作用下发生交互时，会更新其属性状态。


#### 3.2.2 Manipulation and navigation pre-condition checks
#### 3.2.2 操作与导航前置条件检查


The EXECUTOR module verifies the pre-conditions of an action before the action is taken to ensure the action is likely to succeed. In our case, these constraints are predefined for each action (for example, the agent must first be holding a knife to slice an object). If any pre-conditions are not satisfied, the EXECUTOR adjusts the plan accordingly. In more open-ended action interfaces, an LLM's common sense knowledge can be used to infer the preconditions for an action, rather than pre-defining them.
执行器（EXECUTOR）会在执行某个动作之前先验证该动作的前置条件，以确保动作更可能成功。在我们的设置中，这些约束为每个动作预先定义（例如，代理在切割物体前必须先拿着刀）。如果任何前置条件未满足，EXECUTOR 会相应调整计划。在更开放的动作接口中，也可以利用 LLM 的常识知识来推断动作的前置条件，而无需事先预定义。


#### 3.2.3 Locator: LLM-based common sense object search
#### 3.2.3 定位器：基于LLM的常识性物体搜索


When HELPER needs to find an object that has not been detected before, it calls on the LOCATOR module. The Locator prompts an LLM to suggest potential object search location for the EXECUTOR to search nearby, e.g. "search near the sink" or "search in the cupboard". The LOCATOR prompt takes in the language $I$ (which may reference the object location, e.g., "take the mug from the cupboard" ) and queries the LLM to generate proposed locations by essentially parsing the instruction as well as using its common sense. Based on these predictions, HELPER will go to the suggested locations if they exist in the semantic map (e.g., to the sink) and search for the object-of-interest. The LOCATOR's prompt can be found in Section D of the Appendix.
当 HELPER 需要寻找以前未被检测到的物体时，它会调用定位器（LOCATOR）模块。定位器会提示一个 LLM，给出可能的物体搜索位置，以便执行器（EXECUTOR）在附近进行搜索，例如“在水槽附近寻找”或“在橱柜里寻找”。定位器的提示会输入语言 $I$（可能引用物体位置，例如“从橱柜里拿起马克杯”），并向 LLM 查询，通过本质上解析指令，同时结合其常识来生成建议位置。基于这些预测，如果建议位置在语义地图中存在（例如水槽），HElPER 就会前往相应位置搜索目标物体。定位器的提示可在附录 D 中找到。


Implementation details. We use OpenAI's gpt-4-0613 (gpt, 2023) API, except when mentioned otherwise. We resort to the text-embedding-ada-002 (ada, 2022) API to obtain text embeddings. Furthermore, we use the SOLQ object detector (Dong et al., 2021), which is pretrained on MSCOCO (Lin et al., 2014) and fine-tuned on the training rooms of TEACh. For monocular depth estimation, we use the ZoeDepth network (Bhat et al., 2023), pretrained on the NYU indoor dataset (Nathan Silberman and Fergus, 2012) and subsequently fine-tuned on the training rooms of TEACh. In the TEACh evaluations,we use $K = 3$ for retrieval.
实现细节。除非另有说明，我们使用 OpenAI 的 gpt-4-0613（gpt, 2023）API。我们还会调用 text-embedding-ada-002（ada, 2022）API 来获取文本嵌入。此外，我们使用 SOLQ 物体检测器（Dong et al., 2021），它在 MSCOCO（Lin et al., 2014）上进行了预训练，并在 TEACh 的训练房间上进行了微调。对于单目深度估计，我们使用 ZoeDepth 网络（Bhat et al., 2023），该网络在 NYU 室内数据集（Nathan Silberman and Fergus, 2012）上预训练，并随后在 TEACh 的训练房间上进行了微调。在 TEACh 的评测中，我们使用 $K = 3$ 进行检索。


## 4 Experiments
## 4 实验


We test HELPER in the TEACh benchmark (Pad-makumar et al., 2021). Our experiments aim to answer the following questions:
我们在 TEACh 基准（Pad-makumar et al., 2021）上测试 HELPER。我们的实验旨在回答以下问题：


1. How does HELPER compare to the SOTA on task planning and execution from free-form dialogue?
1. HELPER 在基于自由形式对话的任务规划与执行方面，相比最先进方法表现如何？


2. How much do different components of HELPER contribute to performance?
2. HELPER 的不同组件对性能贡献有多大？


3. How much does eliciting human feedback help task completion?
3. 促使人类反馈对任务完成有多大帮助？


4. How effectively does HELPER adapt to a user's specific procedures?
4. HELPER 对用户特定流程的适应效果有多好？


### 4.1 Evaluation on the TEACh dataset
### 4.1 在 TEACh 数据集上的评估


Dataset The dataset comprises over 3,000 human-human, interactive dialogues, geared towards completing household tasks within the AI2-THOR simulation environment (Kolve et al., 2017). We evaluate on the Trajectory from Dialogue (TfD) evaluation variant, where the agent is given a dialogue segment at the start of the episode. The model is then tasked to infer the sequence of actions to execute based on the user's intents in the dialogue segment, ranging from MAKE COFFEE to PREPARE BREAKFAST. We show examples of such dialogues in Figures 1 & 3. We also test on the Execution from Dialogue History (EDH) task in TEACh, where the TfD episodes are partitioned into "sessions". The agent is spawned at the start of one of the sessions and must predict the actions to reach the next session given the dialogue and action history of the previous sessions. The dataset is split into training and validation sets. The validation set is divided into 'seen' and 'unseen' rooms based on their presence in the training set. Validation 'seen' has the same room instances but different object locations and initial states than any episodes in the training set. At each time step, the agent obtains an egocentric RGB image and must choose an action from a specified set to transition to the next step, such as pickup(X), turn left(), etc. Please see Appendix Section G for more details on the simulation environment.
数据集 该数据集包含超过 3,000 段人-人交互对话，旨在在 AI2-THOR 模拟环境中完成家居任务（Kolve et al., 2017）。我们采用“来自对话的轨迹”（TfD）评测变体：在每个回合开始时，智能体会被提供一段对话片段。随后，模型需要根据该对话片段中用户的意图推断要执行的动作序列，从制作咖啡（MAKE COFFEE）到准备早餐（PREPARE BREAKFAST）。这类对话示例见图 1 和图 3。我们还在 TEACh 上测试“来自对话历史的执行”（EDH）任务：将 TfD 回合划分为若干“会话”。智能体在某个会话开始时被生成，并需在已知前述会话的对话与动作历史的情况下，预测动作以到达下一个会话。数据集被划分为训练集和验证集。验证集会根据训练集中是否出现过进行“已见（seen）”与“未见（unseen）”房间划分。已见验证的房间实例与训练集相同，但对象位置和初始状态与训练集中的任一回合不同。在每个时间步，智能体会获得一个以自身视角的 RGB 图像，并必须从给定动作集合中选择一种动作，以切换到下一步，例如 pickup(X)、turn left() 等。有关模拟环境的更多细节，请参见附录 G。


Evaluation metrics Following evaluation practises for the TEACh benchmark, we use the following two metrics: 1. Task success rate (SR), which refers to the fraction of task sessions in which the agent successfully fulfills all goal conditions. 2. Goal condition success rate (GC), which quantifies the proportion of achieved goal conditions across all sessions. Both of these metrics have corresponding path length weighted (PLW) variants. In these versions, the agent incurs penalties for executing a sequence of actions that surpasses the length of the reference path annotated by human experts.
评测指标 按照 TEACh 基准的评测惯例，我们使用以下两个指标：1. 任务成功率（SR），指智能体成功满足所有目标条件的任务会话所占比例。2. 目标条件成功率（GC），用于衡量在全部会话中实现的目标条件的比例。上述两项指标均对应带路径长度加权（PLW）的变体。在这些版本中，智能体若执行的动作序列超过人类专家标注的参考路径长度，将受到惩罚。


<table><tr><td></td><td>No In-Domain LLM</td><td>Memory-Augmented LLM</td><td>User Personalization</td><td>Accepts User Feedback</td><td>VLM-Guided correction</td><td>LLM-Guided Search</td><td>PreCondition Check</td></tr><tr><td>E.T. ${}^{\left( \text{ Pashevich et al.,2021 }\right) }$</td><td>X</td><td>X</td><td>X</td><td>X</td><td>X</td><td>X</td><td>X</td></tr><tr><td>JARVIS(Zheng et al., 2022)</td><td>X</td><td>X</td><td>X</td><td>X</td><td>X</td><td>X</td><td>X</td></tr><tr><td>FILM(Min et al., 2021, 2022)</td><td>X</td><td>X</td><td>X</td><td>X</td><td>X</td><td>X</td><td>X</td></tr><tr><td>DANLI ${}^{\text{ (Zhang et al., 2022) }}$</td><td>X</td><td>X</td><td>X</td><td>X</td><td>X</td><td>X</td><td>✓</td></tr><tr><td>LLM-Planner(Song et al., 2022)</td><td>✓</td><td>✓</td><td>X</td><td>X</td><td>X</td><td>X</td><td>X</td></tr><tr><td>Code as Policies(Liang et al., 2022)</td><td>✓</td><td>X</td><td>X</td><td>X</td><td>X</td><td>X</td><td>X</td></tr><tr><td>HELPER (ours)</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td></tr></table>
<table><tbody><tr><td></td><td>无域内大语言模型</td><td>记忆增强大语言模型</td><td>用户个性化</td><td>接收用户反馈</td><td>视觉语言模型引导的纠错</td><td>大语言模型引导的搜索</td><td>前置条件检查</td></tr><tr><td>E.T. ${}^{\left( \text{ Pashevich et al.,2021 }\right) }$</td><td>X</td><td>X</td><td>X</td><td>X</td><td>X</td><td>X</td><td>X</td></tr><tr><td>JARVIS（郑等，2022）</td><td>X</td><td>X</td><td>X</td><td>X</td><td>X</td><td>X</td><td>X</td></tr><tr><td>FILM（明等，2021，2022）</td><td>X</td><td>X</td><td>X</td><td>X</td><td>X</td><td>X</td><td>X</td></tr><tr><td>DANLI ${}^{\text{ (Zhang et al., 2022) }}$</td><td>X</td><td>X</td><td>X</td><td>X</td><td>X</td><td>X</td><td>✓</td></tr><tr><td>大语言模型规划器（宋等，2022）</td><td>✓</td><td>✓</td><td>X</td><td>X</td><td>X</td><td>X</td><td>X</td></tr><tr><td>将代码作为策略（梁等，2022）</td><td>✓</td><td>X</td><td>X</td><td>X</td><td>X</td><td>X</td><td>X</td></tr><tr><td>HELPER（我们）</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td></tr></tbody></table>


Table 1: Comparison of HELPER to previous work.
表1：HELPER与既有工作的对比。


Baselines We consider the following baselines: 1. Episodic Transformer (E.T.) (Pashevich et al., 2021) is an end-to-end multimodal transformer that encodes language inputs and a history of visual observations to predict actions, trained with imitation learning from human demonstrations.
基线 我们考虑以下基线：1. 记忆式Transformer（E.T.）（Pashevich等，2021）是一个端到端的多模态Transformer，它对语言输入以及视觉观测历史进行编码，以预测动作，并通过模仿学习（来自人类示范）进行训练。


2. Jarvis (Zheng et al., 2022) trains an LLM on the TEACh dialogue to generate high-level subgoals that mimic those performed by the human demonstrator. Jarvis uses a semantic map and the Episodic Transformer for object search.
2. Jarvis（Zheng等，2022）在TEACh对话上训练LLM，以生成与人类示范者执行方式相似的高层子目标。Jarvis使用语义地图以及Episodic Transformer用于对象搜索。


3. FILM (Min et al., 2021, 2022) fine-tunes an LLM to produce parametrized plan templates. Similar to Jarvis, FILM uses a semantic map for carrying out subgoals and a semantic policy for object search.
3. FILM（Min等，2021，2022）对LLM进行微调，以生成参数化的计划模板。与Jarvis类似，FILM使用语义地图来执行子目标，并使用语义策略进行对象搜索。


4. DANLI (Zhang et al., 2022) fine-tunes an LLM to predict high-level subgoals, and uses symbolic planning over an object state and spatial map to create an execution plan. DANLI uses an object search module and manually-defined error correction.
4. DANLI（Zhang等，2022）对LLM进行微调以预测高层子目标，并通过基于对象状态和空间地图的符号规划来生成执行计划。DANLI使用对象搜索模块以及手动定义的错误纠正。


HELPER differs from the baselines in its use of memory-augmented context-dependent prompting of pretrained LLMs and pretrained visual-language models for planning, failure diagnosis and recovery, and object search. We provide a more in-depth comparison of HELPER to previous work in Table 1.
HELPER与这些基线的不同在于：它通过为预训练LLM和预训练视觉-语言模型提供带记忆的、与上下文相关的提示，用于规划、故障诊断与恢复，以及对象搜索。我们在表1中给出了HELPER与既有工作的更深入比较。


Evaluation We show quantitative results for HELPER and the baselines on the TEACh Trajectory from Dialogue (TfD) and Execution from Dialogue History (EDH) validation split in Table 2. On the TfD validation unseen, HELPER achieves a 13.73% task success rate and 14.17% goal-condition success rate, a relative improvement of 1.7x and 2.1x, respectively, over DANLI, the prior SOTA in this setting. HELPER additionally sets a new SOTA in the EDH task, achieving a 17.40% task success rate and 25.86% goal-condition success rate on validation unseen.
评估 我们在表2中展示HELPER及其基线在TEACh对话轨迹（TfD）和基于对话历史的执行（EDH）的验证划分上的定量结果。在TfD验证集上（未见过的情况），HELPER的任务成功率为13.73%，目标条件成功率为14.17%，分别相对DANLI（该设定下的先前SOTA）提升了1.7倍和2.1倍。HELPER在EDH任务上也达到了新的SOTA：在验证集未见样本上，任务成功率为17.40%，目标条件成功率为25.86%。


### 4.2 Ablations
### 4.2 消融实验


We ablate components of HELPER in order to quantify what matters for performance in Table ${2Ab}$ - lations. We perform all ablations on the TEACh TfD validation unseen split. We draw the following conclusions:
我们对 HELPER 的各个组件进行消融实验，以量化表 ${2Ab}$-lations 中哪些因素对性能最关键。我们所有消融实验都在 TEACh 的 TfD 验证集、未见划分上进行。我们得出以下结论：


1. Retrieval-augmented prompting helps for planning, re-planning and failure recovery. Replacing the memory-augmented prompts with a fixed prompt (w/o Mem Aug; Table 2) led to a relative 18% reduction in success rate.
1. 检索增强式提示有助于规划、重新规划与故障恢复。将记忆增强式提示替换为固定提示（w/o Mem Aug；表 2）使成功率相对下降 18%。


2. VLM error correction helps the agent recover from failures. Removal of the visually-grounded plan correction (w/o Correction; Table 2) led to a relative 6% reduction in success rate.
2. VLM 错误校正确实有助于智能体从失败中恢复。移除基于视觉的计划校正（w/o Correction；表 2）使成功率相对下降 6%。


3. The pre-condition check and the LLM search help. Removal of the action pre-condition checks (w/o Pre Check; Table 2) led to a relative 16% reduction in success rate. Replacing the LOCATOR LLM-based search with a random search (w/o LoCATOR; Table 2) led to a relative ${12}\%$ reduction in success rate.
3. 先决条件检查与 LLM 搜索也有帮助。移除动作先决条件检查（w/o Pre Check；表 2）使成功率相对下降 16%。将基于 LOCATOR 的 LLM 搜索替换为随机搜索（w/o LoCATOR；表 2）使成功率相对下降 ${12}\%$%。


4. Larger LLMs perform better. Using GPT-3.5 (w GPT-3.5; Table 2) exhibits a relative 31% reduction in success rate compared to using GPT-4. Our findings on GPT-4's superior planning abilities align with similar findings from recent studies of Wu et al. (2023d); Bubeck et al. (2023); Liu et al. (2023a); Wang et al. (2023a).
4. 更大的 LLM 表现更好。使用 GPT-3.5（w GPT-3.5；表 2）相较于使用 GPT-4，成功率相对降低 31%。我们关于 GPT-4 具备更强规划能力的发现，与 Wu 等（2023d）；Bubeck 等（2023）；Liu 等（2023a）；Wang 等（2023a）近期研究中的类似结论一致。


5. Perception is a bottleneck. Using GT depth (w/ GT depth; Table 2) led to an improvement of 1.15x compared to using estimated depth from RGB. Notable is the 1.77x improvement in path-length weighted success when using GT depth. This change is due to lower accuracy for far depths in our depth estimation network lower, thereby causing the agent to spend more time mapping the environment and navigating noisy obstacle maps. Using lidar or better map estimation techniques could mitigate this issue.
5. 感知是瓶颈。使用 GT 深度（w/ GT depth；表 2）相较于从 RGB 估计深度，提升了 1.15x。值得注意的是，使用 GT 深度时，路径长度加权成功率提升了 1.77x。该变化源于我们深度估计网络在远距离深度上的准确率更低，从而导致智能体花更多时间对环境建图并在嘈杂的障碍物地图中导航。使用激光雷达或更好的地图估计技术可能有助于缓解这一问题。


Table 2: Trajectory from Dialogue (TfD) and Execution from Dialog History (EDH) evaluation on the TEACh validation set. Trajectory length weighted metrics are included in ( parentheses ). SR = success rate. GC = goal condition success rate.
表 2：在 TEACh 验证集上，于对话轨迹驱动（TfD）与基于对话历史的执行（EDH）的评估。轨迹长度加权指标包含在（括号）中。SR = 成功率。GC = 目标条件成功率。


<table><tr><td rowspan="3"></td><td colspan="4">TfD</td><td colspan="4">EDH</td></tr><tr><td colspan="2">Unseen</td><td colspan="2">Seen</td><td colspan="2">Unseen</td><td colspan="2">Seen</td></tr><tr><td>SR</td><td>GC</td><td>SR</td><td>GC</td><td>SR</td><td>GC</td><td>SR</td><td>GC</td></tr><tr><td>E.T.</td><td>0.48 (0.12)</td><td>0.35 (0.59)</td><td>1.02 (0.17)</td><td>1.42 (4.82)</td><td>7.8 (0.9)</td><td>9.1 (1.7)</td><td>10.2 (1.7)</td><td>15.7 (4.1)</td></tr><tr><td>JARVIS</td><td>1.80 (0.30)</td><td>3.10 (1.60)</td><td>1.70 (0.20)</td><td>5.40 (4.50)</td><td>15.80 (2.60)</td><td>16.60 (8.20)</td><td>15.10 (3.30)</td><td>22.60 (8.70)</td></tr><tr><td>FILM</td><td>2.9 (1.0)</td><td>6.1 (2.5)</td><td>5.5 (2.6)</td><td>5.8 (11.6)</td><td>10.2 (1.0)</td><td>18.3 (2.7)</td><td>14.3 (2.1)</td><td>26.4 (5.6)</td></tr><tr><td>DANLI</td><td>7.98 (3.20)</td><td>6.79 (6.57)</td><td>4.97 (1.86)</td><td>10.50 (10.27)</td><td>16.98 (7.24)</td><td>23.44 (19.95)</td><td>17.76 (9.28)</td><td>24.93 (22.20)</td></tr><tr><td>HELPER (ours)</td><td>13.73 (1.61)</td><td>14.17 (4.56)</td><td>12.15 (1.79)</td><td>18.62 (9.28)</td><td>17.40 (2.91)</td><td>25.86 (7.90)</td><td>18.59 (4.00)</td><td>32.09 (9.81)</td></tr></table>
<table><tbody><tr><td rowspan="3"></td><td colspan="4">TfD</td><td colspan="4">EDH</td></tr><tr><td colspan="2">未见过</td><td colspan="2">见过</td><td colspan="2">未见过</td><td colspan="2">见过</td></tr><tr><td>SR</td><td>GC</td><td>SR</td><td>GC</td><td>SR</td><td>GC</td><td>SR</td><td>GC</td></tr><tr><td>E.T.</td><td>0.48 (0.12)</td><td>0.35 (0.59)</td><td>1.02 (0.17)</td><td>1.42 (4.82)</td><td>7.8 (0.9)</td><td>9.1 (1.7)</td><td>10.2 (1.7)</td><td>15.7 (4.1)</td></tr><tr><td>JARVIS</td><td>1.80 (0.30)</td><td>3.10 (1.60)</td><td>1.70 (0.20)</td><td>5.40 (4.50)</td><td>15.80 (2.60)</td><td>16.60 (8.20)</td><td>15.10 (3.30)</td><td>22.60 (8.70)</td></tr><tr><td>FILM</td><td>2.9 (1.0)</td><td>6.1 (2.5)</td><td>5.5 (2.6)</td><td>5.8 (11.6)</td><td>10.2 (1.0)</td><td>18.3 (2.7)</td><td>14.3 (2.1)</td><td>26.4 (5.6)</td></tr><tr><td>DANLI</td><td>7.98 (3.20)</td><td>6.79 (6.57)</td><td>4.97 (1.86)</td><td>10.50 (10.27)</td><td>16.98 (7.24)</td><td>23.44 (19.95)</td><td>17.76 (9.28)</td><td>24.93 (22.20)</td></tr><tr><td>HELPER（我们的）</td><td>13.73 (1.61)</td><td>14.17 (4.56)</td><td>12.15 (1.79)</td><td>18.62 (9.28)</td><td>17.40 (2.91)</td><td>25.86 (7.90)</td><td>18.59 (4.00)</td><td>32.09 (9.81)</td></tr></tbody></table>


<table><tr><td colspan="3">Ablations</td></tr><tr><td>w/o Mem Aug</td><td>11.27 (1.39)</td><td>11.09 (4.00)</td></tr><tr><td>w/o Pre Check</td><td>11.6 (1.36)</td><td>11.32 (4.15)</td></tr><tr><td>w/o Correction</td><td>12.9 (1.53)</td><td>12.45 (4.91)</td></tr><tr><td>w/o Locator</td><td>12.09 (1.29)</td><td>10.89 (3.83)</td></tr><tr><td>w/ GPT-3.5</td><td>9.48 (1.21)</td><td>10.05 (3.68)</td></tr><tr><td>w/ GT depth</td><td>15.85 (2.85)</td><td>14.49 (6.89)</td></tr><tr><td>w/ GT depth,seg</td><td>22.55 (6.39)</td><td>30.00 (14.56)</td></tr><tr><td>w/ GT percept</td><td>30.23 (9.12)</td><td>50.46 (20.24)</td></tr><tr><td colspan="3">User Feedback</td></tr><tr><td>w/ Feedback 1</td><td>16.34 (1.67)</td><td>14.70 (4.69)</td></tr><tr><td>w/ Feedback 2</td><td>17.48 (1.97)</td><td>14.93 (4.74)</td></tr><tr><td>w/ GT percept, <br> Feedback 2</td><td>37.75 (10.96)</td><td>56.77 (19.80)</td></tr></table>
<table><tbody><tr><td colspan="3">消融实验</td></tr><tr><td>不加记忆增强</td><td>11.27 (1.39)</td><td>11.09 (4.00)</td></tr><tr><td>不做预检查</td><td>11.6 (1.36)</td><td>11.32 (4.15)</td></tr><tr><td>不做纠错</td><td>12.9 (1.53)</td><td>12.45 (4.91)</td></tr><tr><td>不使用定位器</td><td>12.09 (1.29)</td><td>10.89 (3.83)</td></tr><tr><td>使用 GPT-3.5</td><td>9.48 (1.21)</td><td>10.05 (3.68)</td></tr><tr><td>使用真实深度</td><td>15.85 (2.85)</td><td>14.49 (6.89)</td></tr><tr><td>使用真实深度与分割</td><td>22.55 (6.39)</td><td>30.00 (14.56)</td></tr><tr><td>使用真实感知</td><td>30.23 (9.12)</td><td>50.46 (20.24)</td></tr><tr><td colspan="3">用户反馈</td></tr><tr><td>使用反馈 1</td><td>16.34 (1.67)</td><td>14.70 (4.69)</td></tr><tr><td>使用反馈 2</td><td>17.48 (1.97)</td><td>14.93 (4.74)</td></tr><tr><td>使用真实感知， <br/> 反馈 2</td><td>37.75 (10.96)</td><td>56.77 (19.80)</td></tr></tbody></table>


Using ground truth segmentation masks and depth (w/ GT depth, seg; Table 2) improves task success and goal-conditioned task success by 1.64x and 2.11x, respectively. This shows the limitations of frame-based object detection and late fusion of detection responses over time. 3D scene representations that fuse features earlier across views may significantly improve 3D object detection. Using GT perception (w/ GT percept; Table 2), which includes depth, segmentation, action success, oracle failure feedback, and increased API failure limit (50), led to 2.20x and 3.56x improvement.
使用真实分割掩码和深度（含GT深度、分割；表2）可将任务成功率和目标条件下的任务成功率分别提升1.64倍和2.11倍。这表明基于帧的目标检测以及随时间滞后的检测响应融合存在局限。将不同视角更早期融合特征以形成3D场景表征，可能显著提升3D目标检测。使用GT感知（含GT感知；表2），其中包括深度、分割、行动成功、来自oracle的失败反馈，并将API失败上限提高到50，最终带来了2.20倍和3.56倍的提升。


### 4.3 Eliciting Users' Feedback
### 4.3 征询用户反馈


We enable HELPER to elicit sparse user feedback by asking "Is the task completed to your satisfaction? Did I miss anything?" once it believes it has completed the task, as explained in Section 3.1.2. The user will then respond with steps missed by HELPER, and HELPER will re-plan based on this feedback. As shown in in Table 2 User Feedback, asking for a user's feedback twice improves performance by ${1.27}\mathrm{x}$ . Previous works do not explore this opportunity of eliciting human feedback partly due to the difficulty of interpreting it-being free-form language-which our work addresses.
我们让 HELPER 通过在其认为任务已完成后询问“任务是否已按您的满意程度完成？我遗漏了什么吗？”来征询稀疏的用户反馈，如第 3.1.2 节所述。随后，用户会给出 HELPER 遗漏的步骤，HEPPER 会根据该反馈重新规划。正如表 2 用户反馈所示，向用户再次征询一次反馈可使性能提升 ${1.27}\mathrm{x}$。以往工作未能探索这一征询人类反馈的机会，部分原因在于要理解这种自由形式语言的反馈较为困难，而我们的工作解决了这一问题。


### 4.4 Personalization
### 4.4 个性化


We evaluate HELPER's ability to retrieve user-specific routines, as well as on their ability to modify the retrieved routines, with one, two, or three modifications, as discussed in 3.1.1. For example, for three modifications we might instruct HELPER: "Make me a Dax sandwich with 1 slice of tomato, 2 lettuce leaves, and add a slice of bread".
我们评估 HELPER 检索用户特定例程的能力，以及在检索到的例程基础上进行修改的能力；修改次数为一次、两次或三次，如 3.1.1 所述。例如，对于三次修改，我们可能会让 HELPER：“给我做一份 Dax 三明治：番茄 1 片、生菜 2 片，并加一片面包”。


Dataset The evaluation tests 10 user-specific plans for each modification category in five distinct tasks: MAKE A SANDWICH; PREPARE BREAKFAST; MAKE A SALAD; PLACE X ON Y; and CLEAN X. The evaluation contains 40 user requests. The complete list of user-specific plans and modification requests can be found in the Appendix, Section C.
数据集 评估在五个不同任务中，每个修改类别测试 10 份面向用户的计划：做三明治；准备早餐；做沙拉；把 X 放到 Y 上；清洁 X。评估共包含 40 条用户请求。用户特定计划及修改请求的完整列表见附录 C 节。


Evaluation We report the success rate in Table 3. HELPER generates the correct personalized plan for all but three instances, out of 40 evaluation requests. This showcases the ability of HELPER to acquire, retrieve and adapt plans based on context and previous user interactions.
评估 我们在表 3 中报告成功率。对于 40 条评估请求，除三个实例外，HElPER 都能生成正确的个性化计划。这展示了 HELPER 能够基于上下文与既往用户交互获取、检索并调整计划的能力。


## 5 Limitations
## 5 局限性


Our model in its current form has the following limitations:
我们当前形式的模型存在以下局限：


1. Simplified failure detection. The AI2-THOR simulator much simplifies action failure detection which our work and previous works exploit (Min et al., 2021; Inoue and Ohashi, 2022). In a more general setting, continuous progress monitoring from pixels would be required for failure detection, which model VLMs can deliver and we will address in future work.
1. 简化的失败检测。AI2-THOR 模拟器对动作失败检测做了很大简化，而我们的工作以及以往研究都利用了这一点（Min 等，2021；Inoue 和 Ohashi，2022）。在更一般的场景下，失败检测需要从像素进行连续进展监测，这正是 VLM 能够提供的能力，我们将在未来工作中解决。


Table 3: Evaluation of HELPER for user personalization. Reported is success of generating the correct plan for 10 personalized plans for a request of the original plan without modifications, and one, two, or three modifications to the original plan. These experiments use the text-davinci-003 model as the prompted LLM.
表 3：用于用户个性化的 HELPER 评估。报告了：在不修改原始计划的请求下，为 10 个个性化计划生成正确计划的成功情况，以及对原始计划进行一次、两次或三次修改时的成功情况。这些实验使用 text-davinci-003 模型作为被提示的 LLM。


<table><tr><td></td><td>Success</td></tr><tr><td>Original Plan</td><td>100%</td></tr><tr><td>One Change</td><td>100%</td></tr><tr><td>Two Changes</td><td>80%</td></tr><tr><td>Three Changes</td><td>90%</td></tr></table>
<table><tbody><tr><td></td><td>成功</td></tr><tr><td>原始计划</td><td>100%</td></tr><tr><td>一次变更</td><td>100%</td></tr><tr><td>两次变更</td><td>80%</td></tr><tr><td>三次变更</td><td>90%</td></tr></tbody></table>


2. 3D perception bottleneck. HELPER relies on 2D object detectors and depth 3D lifting for 3D object localization. We observe a 2X boost in TEACh success rate from using ground truth segmentation in HELPER. In future work we plan to integrate early 2D features into persistent 3D scene feature representations for more accurate 3D object detection.
2. 3D 感知瓶颈。HELPER 依赖 2D 目标检测器以及用于 3D 目标定位的深度 3D 反投影。我们观察到，在 HELPER 中使用真实分割，TEACh 成功率提升了 2 倍。未来工作中，我们计划将早期的 2D 特征集成到持久的 3D 场景特征表示中，以实现更准确的 3D 目标检测。


4. Cost from LLM querying. GPT-4 API is the most accurate LLM used in HELPER and incurs a significant cost. NLP research in model compression may help decreasing these costs, or finetuning smaller models with enough input-output pairs.
4. 来自 LLM 查询的成本。GPT-4 API 是 HELPER 中使用的最准确的 LLM，但代价很高。模型压缩方面的 NLP 研究可能有助于降低这些成本，或在足够的输入输出样本下对更小的模型进行微调。


3. Multimodal (vision and language) memory retrieval. Currently, we use a text bottleneck in our environment state descriptions. Exciting future directions include exploration of visual state incorporation to the language model and partial adaptation of its parameters. A multi-modal approach to the memory and plan generation would help contextualize the planning more with the visual state.
3. 多模态（视觉与语言）记忆检索。目前，我们在环境状态描述中使用的是文本瓶颈。令人期待的未来方向包括探索将视觉状态融入语言模型，以及对其参数进行部分适配。面向记忆与计划生成的多模态方法将有助于让规划更充分地结合视觉状态。


Last, to follow human instructions outside of simulation environments our model would need to interface with robot closed-loop policies instead of abstract manipulation primitives, following previous work (Liang et al., 2022).
最后，为了在仿真环境之外执行对人的指令，我们的模型需要与机器人闭环策略进行接口对接，而不是使用抽象的操作原语，遵循既有工作（Liang et al., 2022）。


## 6 Conclusion
## 6 结论


We presented HELPER, an instructable embodied agent that uses memory-augmented prompting of pre-trained LLMs to parse dialogue segments, instructions and corrections to programs over action primitives, that it executes in household environments from visual input. HELPER updates its memory with user-instructed action programs after successful execution, allowing personalized interactions by recalling and adapting them. It sets a new state-of-the-art in the TEACh benchmark. Future research directions include extending the model to include a visual modality by encoding visual context during memory retrieval or as direct input to the LLM. We believe our work contributes towards exciting new capabilities for instructable and conversable systems, for assisting users and personalizing human-robot communication.
我们提出 HELPER，这是一种可操作指令的具身代理，它通过对预训练 LLM 进行记忆增强式提示，将对话片段、指令与对程序的修正解析为对动作原语的可执行计划，并从视觉输入在家庭环境中执行。HEPPER 在成功执行后，会使用用户指令的动作程序更新其记忆，从而通过回忆与适配实现个性化交互。它在 TEACh 基准上达到了新的最先进水平。未来研究方向包括：扩展模型以引入视觉模态，例如在记忆检索时对视觉上下文进行编码，或将其作为 LLM 的直接输入。我们相信，我们的工作为具备可操作指令与可对话的系统带来令人振奋的新能力，帮助用户并实现人机沟通的个性化。


## 7 Acknowledgements
## 7 致谢


This material is based upon work supported by National Science Foundation grants GRF DGE1745016 & DGE2140739 (GS), a DARPA Young Investigator Award, a NSF CAREER award, an AFOSR Young Investigator Award, and DARPA Machine Common Sense, and an ONR award AWD00002287 . Any opinions, findings and conclusions or recommendations expressed in this material are those of the authors and do not necessarily reflect the views of the United States Army, the National Science Foundation, or the United States Air Force.
本材料基于国家科学基金会 GRF DGE1745016 和 DGE2140739（GS）资助、DARPA 青年研究者奖、NSF CAREER 奖、AFOSR 青年研究者奖、DARPA Machine Common Sense 项目以及 ONR 奖项 AWD00002287 所支持的工作。本文所表达的任何意见、发现、结论或建议均为作者观点，不一定反映美国陆军、国家科学基金会或美国空军的立场。


This research project has benefitted from the Mi-crosoft Accelerate Foundation Models Research (AFMR) grant program through which leading foundation models hosted by Microsoft Azure along with access to Azure credits were provided to conduct the research.
本研究项目受益于 Microsoft Accelerate Foundation Models Research（AFMR）资助计划，通过该计划提供了托管于 Microsoft Azure 上的领先基础模型，以及用于开展研究的 Azure 额度。


## Ethics Statement
## 伦理声明


The objective of this research is to construct autonomous agents. Despite the absence of human experimentation, practitioners could potentially implement this technology in human-inclusive environments. Therefore, applications of our research should appropriately address privacy considerations.
本研究的目标是构建自主智能体。尽管未进行人体试验，实践者仍可能将这项技术应用到包含人的环境中。因此，我们的研究应用应恰当地处理隐私考量。


All the models developed in this study were trained using Ai2Thor (Kolve et al., 2017). Consequently, there might be an inherent bias towards North American homes. Additionally, we only consider English language inputs in this study.
本研究中开发的所有模型均使用 Ai2Thor 进行训练（Kolve 等，2017）。因此，可能存在对北美家庭的固有偏差。此外，本研究仅考虑英语输入。


## References
## 参考文献


2022. New and improved embedding model.
2022。新且改进的嵌入模型。


2023. Openai. gpt-4 technical report. arXiv preprint arxiv:2303.08774.
2023。Openai。《gpt-4 技术报告》。arXiv 预印本 arXiv:2303.08774。


Michael Ahn, Anthony Brohan, Noah Brown, Yev-gen Chebotar, Omar Cortes, Byron David, Chelsea Finn, Chuyuan Fu, Keerthana Gopalakrishnan, Karol Hausman, Alex Herzog, Daniel Ho, Jasmine Hsu, Julian Ibarz, Brian Ichter, Alex Irpan, Eric Jang, Rosario Jauregui Ruano, Kyle Jeffrey, Sally Jes-month, Nikhil Joshi, Ryan Julian, Dmitry Kalashnikov, Yuheng Kuang, Kuang-Huei Lee, Sergey Levine, Yao Lu, Linda Luu, Carolina Parada, Peter Pastor, Jornell Quiambao, Kanishka Rao, Jarek Rettinghouse, Diego Reyes, Pierre Sermanet, Nicolas Sievers, Clayton Tan, Alexander Toshev, Vincent Vanhoucke, Fei Xia, Ted Xiao, Peng Xu, Sichun Xu, Mengyuan Yan, and Andy Zeng. 2022. Do as i can and not as i say: Grounding language in robotic af-fordances. In arXiv preprint arXiv:2204.01691.
Michael Ahn, Anthony Brohan, Noah Brown, Yev-gen Chebotar, Omar Cortes, Byron David, Chelsea Finn, Chuyuan Fu, Keerthana Gopalakrishnan, Karol Hausman, Alex Herzog, Daniel Ho, Jasmine Hsu, Julian Ibarz, Brian Ichter, Alex Irpan, Eric Jang, Rosario Jauregui Ruano, Kyle Jeffrey, Sally Jes-month, Nikhil Joshi, Ryan Julian, Dmitry Kalashnikov, Yuheng Kuang, Kuang-Huei Lee, Sergey Levine, Yao Lu, Linda Luu, Carolina Parada, Peter Pastor, Jornell Quiambao, Kanishka Rao, Jarek Rettinghouse, Diego Reyes, Pierre Sermanet, Nicolas Sievers, Clayton Tan, Alexander Toshev, Vincent Vanhoucke, Fei Xia, Ted Xiao, Peng Xu, Sichun Xu, Mengyuan Yan, and Andy Zeng。2022。《按我能做的做，而不是按我说的做：将语言落地到机器人可供性》中。In arXiv 预印本 arXiv:2204.01691。


Peter Anderson, Angel Chang, Devendra Singh Chap-lot, Alexey Dosovitskiy, Saurabh Gupta, Vladlen Koltun, Jana Kosecka, Jitendra Malik, Roozbeh Mot-taghi, Manolis Savva, et al. 2018a. On evaluation of embodied navigation agents. arXiv preprint arXiv:1807.06757.
Peter Anderson, Angel Chang, Devendra Singh Chap-lot, Alexey Dosovitskiy, Saurabh Gupta, Vladlen Koltun, Jana Kosecka, Jitendra Malik, Roozbeh Mot-taghi, Manolis Savva, 等。2018a。具身导航智能体评估。arXiv 预印本 arXiv:1807.06757。


Peter Anderson, Qi Wu, Damien Teney, Jake Bruce, Mark Johnson, Niko Sünderhauf, Ian Reid, Stephen Gould, and Anton Van Den Hengel. 2018b. Vision-and-language navigation: Interpreting visually-grounded navigation instructions in real environments. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 3674- 3683.
Peter Anderson, Qi Wu, Damien Teney, Jake Bruce, Mark Johnson, Niko Sünderhauf, Ian Reid, Stephen Gould, and Anton Van Den Hengel。2018b。视觉-语言导航：在真实环境中解读视觉接地的导航指令。在 IEEE 计算机视觉与模式识别会议论文集，3674-3683 页。


Shariq Farooq Bhat, Reiner Birkl, Diana Wofk, Peter Wonka, and Matthias Müller. 2023. Zoedepth: Zero-shot transfer by combining relative and metric depth. arXiv preprint arXiv:2302.12288.
Shariq Farooq Bhat, Reiner Birkl, Diana Wofk, Peter Wonka, and Matthias Müller。2023。Zoedepth：通过结合相对深度与度量深度实现零样本迁移。arXiv 预印本 arXiv:2302.12288。


Valts Blukis, Chris Paxton, Dieter Fox, Animesh Garg, and Yoav Artzi. 2022. A persistent spatial semantic representation for high-level natural language instruction execution. In Conference on Robot Learning, pages 706-717. PMLR.
Valts Blukis, Chris Paxton, Dieter Fox, Animesh Garg, and Yoav Artzi。2022。用于高级自然语言指令执行的持久空间语义表示。在 Robot Learning 会议，706-717 页。PMLR。


Anthony Brohan, Noah Brown, Justice Carbajal, Yev-gen Chebotar, Joseph Dabis, Chelsea Finn, Keerthana Gopalakrishnan, Karol Hausman, Alex Herzog, Jasmine Hsu, et al. 2022. Rt-1: Robotics transformer for real-world control at scale. arXiv preprint arXiv:2212.06817.
Anthony Brohan, Noah Brown, Justice Carbajal, Yev-gen Chebotar, Joseph Dabis, Chelsea Finn, Keerthana Gopalakrishnan, Karol Hausman, Alex Herzog, Jasmine Hsu, 等。2022。Rt-1：用于真实世界大规模控制的机器人 Transformer。arXiv 预印本 arXiv:2212.06817。


Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. 2020. Language models are few-shot learners. Advances in neural information processing systems, 33:1877-1901.
Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, 等。2020。语言模型是少样本学习者。Advances in neural information processing systems, 33:1877-1901。


Sébastien Bubeck, Varun Chandrasekaran, Ronen El-dan, Johannes Gehrke, Eric Horvitz, Ece Kamar, Peter Lee, Yin Tat Lee, Yuanzhi Li, Scott Lundberg, et al. 2023. Sparks of artificial general intelligence: Early experiments with gpt-4. arXiv preprint arXiv:2303.12712.
Sébastien Bubeck, Varun Chandrasekaran, Ronen El-dan, Johannes Gehrke, Eric Horvitz, Ece Kamar, Peter Lee, Yin Tat Lee, Yuanzhi Li, Scott Lundberg, 等。2023。人工通用智能的火花：gpt-4 的早期实验。arXiv 预印本 arXiv:2303.12712。


Yuchen Cao, Nilay Pande, Ayush Jain, Shikhar Sharma, Gabriel Sarch, Nikolaos Gkanatsios, Xian Zhou, and Katerina Fragkiadaki. Embodied symbiotic assistants that see, act, infer and chat.
Yuchen Cao, Nilay Pande, Ayush Jain, Shikhar Sharma, Gabriel Sarch, Nikolaos Gkanatsios, Xian Zhou, and Katerina Fragkiadaki。具身共生助手：能看、能做、能推理、能聊天。


Devendra Singh Chaplot, Dhiraj Gandhi, Saurabh Gupta, Abhinav Gupta, and Ruslan Salakhutdinov. 2020a. Learning to explore using active neural slam. arXiv preprint arXiv:2004.05155.
Devendra Singh Chaplot, Dhiraj Gandhi, Saurabh Gupta, Abhinav Gupta, and Ruslan Salakhutdinov。2020a。利用主动神经 SLAM 学习探索。arXiv 预印本 arXiv:2004.05155。


Devendra Singh Chaplot, Dhiraj Prakashchand Gandhi, Abhinav Gupta, and Russ R Salakhutdinov. 2020b. Object goal navigation using goal-oriented semantic exploration. Advances in Neural Information Processing Systems, 33.
Devendra Singh Chaplot, Dhiraj Prakashchand Gandhi, Abhinav Gupta, and Russ R Salakhutdinov。2020b。使用面向目标的语义探索进行目标物体导航。Advances in Neural Information Processing Systems, 33。


Wenhu Chen, Xueguang Ma, Xinyi Wang, and William W Cohen. 2022. Program of thoughts prompting: Disentangling computation from reasoning for numerical reasoning tasks. arXiv preprint arXiv:2211.12588.
文虎·陈，薛光·马，辛怡·王，以及威廉 W·科恩。2022。思维提示程序：为数值推理任务将计算与推理解耦。arXiv 预印本 arXiv:2211.12588。


Bin Dong, Fangao Zeng, Tiancai Wang, Xiangyu Zhang, and Yichen Wei. 2021. Solq: Segmenting objects by learning queries. Advances in Neural Information Processing Systems, 34:21898-21909.
宾·董，方高·曾，田才·王，向宇·张，以及乙晨·魏。2021。Solq：通过学习查询来分割物体。神经信息处理系统进展，34：21898-21909。


Linxi Fan, Yuke Zhu, Jiren Zhu, Zihua Liu, Orien Zeng, Anchit Gupta, Joan Creus-Costa, Silvio Savarese, and Li Fei-Fei. 2018. Surreal: Open-source reinforcement learning framework and robot manipulation benchmark. In Conference on Robot Learning, pages 767-782. PMLR.
林熙·范，郁可·朱，吉仁·朱，子华·刘，奥里恩·曾，安奇特·古普塔，琼·创斯-科斯塔，西尔维奥·萨瓦雷塞，以及李·菲-菲。2018。Surreal：开源强化学习框架与机器人操作基准。载于机器人学习会议，第 767-782 页。PMLR。


Chuang Gan, Siyuan Zhou, Jeremy Schwartz, Seth Alter, Abhishek Bhandwaldar, Dan Gutfreund, Daniel L.K. Yamins, James J. DiCarlo, Josh McDermott, Antonio Torralba, and Joshua B. Tenenbaum. 2022. The three-dworld transport challenge: A visually guided task-and-motion planning benchmark towards physically realistic embodied ai. In 2022 International Conference on Robotics and Automation (ICRA), pages 8847-8854.
创光·甘，思远·周，杰里米·施瓦茨，塞斯·奥尔特，阿比谢克·班瓦尔达尔，丹·古特弗雷恩特，丹尼尔 L.K. 雅明斯，詹姆斯 J·迪卡洛，乔什·麦克德莫特，安东尼奥·托拉尔巴，以及约书亚 B·滕瑟鲍姆。2022。三维世界运输挑战：面向物理真实具身 ai 的视觉引导任务与运动规划基准。载于 2022 年机器人与自动化国际会议（ICRA），第 8847-8854 页。


Luyu Gao, Aman Madaan, Shuyan Zhou, Uri Alon, Pengfei Liu, Yiming Yang, Jamie Callan, and Graham Neubig. 2022. Pal: Program-aided language models. arXiv preprint arXiv:2211.10435.
鲁宇·高，阿曼·马丹，舒言·周，乌里·阿隆，鹏飞·刘，易明·杨，杰米·卡兰，以及格雷厄姆·纽比格。2022。Pal：程序辅助语言模型。arXiv 预印本 arXiv:2211.10435。


Tianyu Gao, Adam Fisch, and Danqi Chen. 2020. Making pre-trained language models better few-shot learners. arXiv preprint arXiv:2012.15723.
田宇·高，亚当·费施，以及丹琪·陈。2020。让预训练语言模型成为更好的少样本学习者。arXiv 预印本 arXiv:2012.15723。


Theophile Gervet, Soumith Chintala, Dhruv Batra, Jitendra Malik, and Devendra Singh Chaplot. 2022. Navigating to objects in the real world. arXiv preprint arXiv:2212.00922.
泰奥菲尔·热尔维特，索米斯·钦塔拉，德鲁夫·巴特拉，贾耶恩德拉·马利克，以及德文德拉·辛格·查普洛特。2022。在真实世界中导航到物体。arXiv 预印本 arXiv:2212.00922。


Chenguang Huang, Oier Mees, Andy Zeng, and Wolfram Burgard. 2022a. Visual language maps for robot navigation. arXiv preprint arXiv:2210.05714.
成光·黄，奥尔·米斯，安迪·曾，以及沃尔弗拉姆·堡格德。2022a。用于机器人导航的视觉语言地图。arXiv 预印本 arXiv:2210.05714。


Siyuan Huang, Zhengkai Jiang, Hao Dong, Yu Qiao, Peng Gao, and Hongsheng Li. 2023. Instruct2act: Mapping multi-modality instructions to robotic actions with large language model. arXiv preprint arXiv:2305.11176.
思远·黄，郑凯·姜，郝东，尤乔·皮奥，鹏高，以及洪生·李。2023。Instruct2act：将多模态指令映射为大语言模型的机器人动作。arXiv 预印本 arXiv:2305.11176。


Wenlong Huang, Fei Xia, Ted Xiao, Harris Chan, Jacky Liang, Pete Florence, Andy Zeng, Jonathan Tompson, Igor Mordatch, Yevgen Chebotar, et al. 2022b. Inner monologue: Embodied reasoning through planning with language models. arXiv preprint arXiv:2207.05608.
文龙·黄，菲·夏，特德·肖，哈里斯·陈，杰基·梁，彼特·弗洛伦斯，安迪·曾，乔纳森·汤普森，伊戈尔·莫尔德奇，叶夫根·切博塔尔等。2022b。内心独白：借助语言模型进行规划来实现具身推理。arXiv 预印本 arXiv:2207.05608。


Yuki Inoue and Hiroki Ohashi. 2022. Prompter: Utilizing large language model prompting for a data efficient embodied instruction following. arXiv preprint arXiv:2211.03267.
由纪·井上和广树·大桥。2022。Prompter：利用大语言模型提示实现数据高效的具身指令跟随。arXiv 预印本 arXiv:2211.03267。


Chao Jia, Yinfei Yang, Ye Xia, Yi-Ting Chen, Zarana Parekh, Hieu Pham, Quoc Le, Yun-Hsuan Sung, Zhen Li, and Tom Duerig. 2021. Scaling up visual and vision-language representation learning with noisy text supervision. In International Conference on Machine Learning, pages 4904-4916. PMLR.
赵超，尹飞·杨，叶霞，易-廷·陈，扎拉娜·帕雷克，海乌·范，阔克·莱，云-宣·宋，郑莉，以及汤姆·杜里格。2021。用带噪文本监督扩大视觉与视觉-语言表征学习规模。载于机器学习国际会议，第 4904-4916 页。PMLR。


Zhengbao Jiang, Frank F. Xu, Luyu Gao, Zhiqing Sun, Qian Liu, Jane Dwivedi-Yu, Yiming Yang, Jamie Callan, and Graham Neubig. 2023. Active retrieval augmented generation.
郑宝·姜，弗兰克·F·许，鲁宇·高，致庆·孙，倩·刘，简·杜维迪-宇，易明·杨，杰米·卡兰，以及格雷厄姆·纽比格。2023。主动检索增强生成。


Dan Klein and Christopher D Manning. 2003. Accurate unlexicalized parsing. In Proceedings of the 41st annual meeting of the association for computational linguistics, pages 423-430.
丹·克莱因和克里斯托弗·D·曼宁。2003。准确的非词汇化解析。载于计算语言学协会第 41 届年度会议论文集，第 423-430 页。


Eric Kolve, Roozbeh Mottaghi, Winson Han, Eli VanderBilt, Luca Weihs, Alvaro Herrasti, Daniel Gordon, Yuke Zhu, Abhinav Gupta, and Ali Farhadi. 2017. AI2-THOR: An Interactive 3D Environment for Visual AI. arXiv.
埃里克·科尔夫，鲁兹贝赫·莫塔吉，温森·汉，埃利·范德比尔特，卢卡·魏斯，阿尔瓦罗·埃拉斯蒂，丹尼尔·戈登，郁可·朱，阿比纳夫·古普塔，以及阿里·法尔哈迪。2017。AI2-THOR：面向视觉 AI 的交互式三维环境。arXiv。


Jacob Krantz, Erik Wijmans, Arjun Majumdar, Dhruv Batra, and Stefan Lee. 2020. Beyond the nav-graph: Vision-and-language navigation in continuous environments. In Computer Vision-ECCV 2020: 16th European Conference, Glasgow, UK, August 23-28, 2020, Proceedings, Part XXVIII 16, pages 104-120. Springer.
Jacob Krantz，Erik Wijmans，Arjun Majumdar，Dhruv Batra，和 Stefan Lee。2020。超越导航图：连续环境中的视觉-语言导航。载于《计算机视觉—ECCV 2020：第16届欧洲会议，英国格拉斯哥，2020年8月23-28日，论文集，第XXVIII部分16》，第104-120页。Springer。


Alexander Ku, Peter Anderson, Roma Patel, Eugene Ie, and Jason Baldridge. 2020. Room-across-room: Multilingual vision-and-language navigation with dense spatiotemporal grounding. arXiv preprint arXiv:2010.07954.
Alexander Ku，Peter Anderson，Roma Patel，Eugene Ie，和 Jason Baldridge。2020。室内跨室：具有密集时空定位的多语言视觉-语言导航。arXiv预印本 arXiv:2010.07954。


Brian Lester, Rami Al-Rfou, and Noah Constant. 2021. The power of scale for parameter-efficient prompt tuning. arXiv preprint arXiv:2104.08691.
Brian Lester，Rami Al-Rfou，和 Noah Constant。2021。规模的力量：参数高效的提示微调。arXiv预印本 arXiv:2104.08691。


Jacky Liang, Wenlong Huang, Fei Xia, Peng Xu, Karol Hausman, Brian Ichter, Pete Florence, and Andy Zeng. 2022. Code as policies: Language model programs for embodied control. arXiv preprint arXiv:2209.07753.
Jacky Liang，Wenlong Huang，Fei Xia，Peng Xu，Karol Hausman，Brian Ichter，Pete Florence，和 Andy Zeng。2022。代码即策略：用于具身控制的语言模型程序。arXiv预印本 arXiv:2209.07753。


Percy Liang. 2016. Learning executable semantic parsers for natural language understanding. Communications of the ACM, 59(9):68-76.
Percy Liang。2016。为自然语言理解学习可执行的语义解析器。《ACM通讯》，59(9)：68-76。


Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C Lawrence Zitnick. 2014. Microsoft coco: Common objects in context. In Computer Vision-ECCV 2014: 13th European Conference, Zurich, Switzerland, September 6-12, 2014, Proceedings, Part V 13, pages 740-755. Springer.
Tsung-Yi Lin，Michael Maire，Serge Belongie，James Hays，Pietro Perona，Deva Ramanan，Piotr Dollár，和 C Lawrence Zitnick。2014。Microsoft COCO：语境中的常见物体。载于《计算机视觉—ECCV 2014：第13届欧洲会议，瑞士苏黎世，2014年9月6-12日，论文集，第V部分13》，第740-755页。Springer。


Haoyu Liu, Yang Liu, Hongkai He, and Hangfang Yang. 2022a. Lebp-language expectation & binding policy: A two-stream framework for embodied vision-and-language interaction task learning agents. arXiv preprint arXiv:2203.04637.
Haoyu Liu，Yang Liu，Hongkai He，和 Hangfang Yang。2022a。Lebp-语言期望与绑定策略：用于具身视觉-语言交互任务学习智能体的双流框架。arXiv预印本 arXiv:2203.04637。


Jiachang Liu, Dinghan Shen, Yizhe Zhang, Bill Dolan, Lawrence Carin, and Weizhu Chen. 2021. What makes good in-context examples for gpt-3? arXiv preprint arXiv:2101.06804.
Jiachang Liu，Dinghan Shen，Yizhe Zhang，Bill Dolan，Lawrence Carin，和 Weizhu Chen。2021。gpt-3 的好的情境示例是什么？arXiv预印本 arXiv:2101.06804。


Xiaotian Liu, Hector Palacios, and Christian Muise. 2022b. A planning based neural-symbolic approach for embodied instruction following. Interactions, 9(8):17.
Xiaotian Liu，Hector Palacios，和 Christian Muise。2022b。一种基于规划的神经-符号方法用于具身指令跟随。《Interactions》，9(8)：17。


Yiheng Liu, Tianle Han, Siyuan Ma, Jiayue Zhang, Yuanyuan Yang, Jiaming Tian, Hao He, Antong Li, Mengshen He, Zhengliang Liu, et al. 2023a. Summary of chatgpt/gpt-4 research and perspective towards the future of large language models. arXiv preprint arXiv:2304.01852.
Yiheng Liu，Tianle Han，Siyuan Ma，Jiayue Zhang，Yuanyuan Yang，Jiaming Tian，Hao He，Antong Li，Mengshen He，Zhengliang Liu，等。2023a。ChatGPT/GPT-4 研究综述及面向大语言模型未来的展望。arXiv预印本 arXiv:2304.01852。


Zeyi Liu, Arpit Bahety, and Shuran Song. 2023b. Reflect: Summarizing robot experiences for failure explanation and correction. arXiv preprint arXiv:2306.15724.
Zeyi Liu，Arpit Bahety，和 Shuran Song。2023b。Reflect：为失败解释与纠正而总结机器人的经验。arXiv预印本 arXiv:2306.15724。


So Yeon Min, Devendra Singh Chaplot, Pradeep Raviku-mar, Yonatan Bisk, and Ruslan Salakhutdinov. 2021. Film: Following instructions in language with modular methods.
So Yeon Min，Devendra Singh Chaplot，Pradeep Raviku-mar，Yonatan Bisk，和 Ruslan Salakhutdinov。2021。Film：用模块化方法在语言中跟随指令。


So Yeon Min, Hao Zhu, Ruslan Salakhutdinov, and Yonatan Bisk. 2022. Don't copy the teacher: Data and model challenges in embodied dialogue. arXiv preprint arXiv:2210.04443.
So Yeon Min，Hao Zhu，Ruslan Salakhutdinov，和 Yonatan Bisk。2022。别照搬老师：具身对话中的数据与模型挑战。arXiv预印本 arXiv:2210.04443。


Michael Murray and Maya Cakmak. 2022. Following natural language instructions for household tasks with landmark guided search and reinforced pose adjustment. IEEE Robotics and Automation Letters, 7(3):6870-6877.
Michael Murray 和 Maya Cakmak。2022。通过地标引导搜索与强化姿态调整，用自然语言指令完成家务任务。《IEEE机器人与自动化快报》，7(3)：6870-6877。


Reiichiro Nakano, Jacob Hilton, Suchir Balaji, Jeff Wu, Long Ouyang, Christina Kim, Christopher Hesse, Shantanu Jain, Vineet Kosaraju, William Saunders, et al. 2021. Webgpt: Browser-assisted question-answering with human feedback. arXiv preprint arXiv:2112.09332.
Reiichiro Nakano，Jacob Hilton，Suchir Balaji，Jeff Wu，Long Ouyang，Christina Kim，Christopher Hesse，Shantanu Jain，Vineet Kosaraju，William Saunders，等。2021。WebGPT：结合人类反馈的浏览器辅助问答。arXiv预印本 arXiv:2112.09332。


Pushmeet Kohli Nathan Silberman, Derek Hoiem and Rob Fergus. 2012. Indoor segmentation and support inference from rgbd images. In ECCV.
Pushmeet Kohli Nathan Silberman，Derek Hoiem 和 Rob Fergus。2012。基于RGBD图像的室内分割与支撑推断。载于ECCV。


Maxwell Nye, Anders Johan Andreassen, Guy Gur-Ari, Henryk Michalewski, Jacob Austin, David Bieber, David Dohan, Aitor Lewkowycz, Maarten Bosma, David Luan, et al. Show your work: Scratchpads for intermediate computation with language models, november 2021. URL http://arxiv.org/abs/2112.00114.
Maxwell Nye、Anders Johan Andreassen、Guy Gur-Ari、Henryk Michalewski、Jacob Austin、David Bieber、David Dohan、Aitor Lewkowycz、Maarten Bosma、David Luan等。展示你的计算过程：用于语言模型中间计算的Scratchpads，2021年11月。URL http://arxiv.org/abs/2112.00114。


Aishwarya Padmakumar, Jesse Thomason, Ayush Shri-vastava, Patrick Lange, Anjali Narayan-Chen, Span-dana Gella, Robinson Piramuthu, Gokhan Tur, and Dilek Hakkani-Tur. 2021. Teach: Task-driven embodied agents that chat.
Aishwarya Padmakumar、Jesse Thomason、Ayush Shri-vastava、Patrick Lange、Anjali Narayan-Chen、Span-dana Gella、Robinson Piramuthu、Gokhan Tur 和 Dilek Hakkani-Tur。2021年。Teach：面向任务的会聊天的具身智能体。


Alexander Pashevich, Cordelia Schmid, and Chen Sun. 2021. Episodic transformer for vision-and-language navigation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 15942- 15952.
Alexander Pashevich、Cordelia Schmid 和 Chen Sun。2021年。用于视觉与语言导航的分段式Transformer。载于：IEEE/CVF计算机视觉国际会议论文集，第15942-15952页。


Ethan Perez, Douwe Kiela, and Kyunghyun Cho. 2021. True few-shot learning with language models. Advances in neural information processing systems, 34:11054-11070.
Ethan Perez、Douwe Kiela 和 Kyunghyun Cho。2021年。使用语言模型进行真正的少样本学习。神经信息处理系统进展，第34卷：11054-11070。


Toran Bruce Richards. 2023. Auto-gpt: An autonomous gpt-4 experiment.
Toran Bruce Richards。2023年。Auto-gpt：一个自主版 GPT-4 实验。


Gabriel Sarch, Zhaoyuan Fang, Adam W Harley, Paul Schydlo, Michael J Tarr, Saurabh Gupta, and Katerina Fragkiadaki. 2022. Tidee: Tidying up novel rooms using visuo-semantic commonsense priors. In Computer Vision-ECCV 2022: 17th European Conference, Tel Aviv, Israel, October 23-27, 2022, Proceedings, Part XXXIX, pages 480-496. Springer.
Gabriel Sarch、Zhaoyuan Fang、Adam W Harley、Paul Schydlo、Michael J Tarr、Saurabh Gupta 和 Katerina Fragkiadaki。2022年。TIDEE：借助视觉语义常识先验整理新颖房间。载于：计算机视觉-ECCV 2022：第17届欧洲计算机视觉会议，2022年10月23-27日，会议论文集，第XXXIX部分，第480-496页。Springer。


Manolis Savva, Abhishek Kadian, Oleksandr Maksymets, Yili Zhao, Erik Wijmans, Bhavana Jain, Julian Straub, Jia Liu, Vladlen Koltun, Jitendra Malik, et al. 2019. Habitat: A platform for embodied ai research. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 9339-9347.
Manolis Savva、Abhishek Kadian、Oleksandr Maksymets、Yili Zhao、Erik Wijmans、Bhavana Jain、Julian Straub、Jia Liu、Vladlen Koltun、Jitendra Malik等。2019年。Habitat：一个用于具身AI研究的平台。载于：IEEE/CVF计算机视觉国际会议论文集，第9339-9347页。


Timo Schick and Hinrich Schütze. 2020. It's not just size that matters: Small language models are also few-shot learners. arXiv preprint arXiv:2009.07118.
Timo Schick 和 Hinrich Schütze。2020年。重要的不只是规模：小型语言模型同样是少样本学习者。arXiv预印本 arXiv:2009.07118。


Bokui Shen, Fei Xia, Chengshu Li, Roberto Martín-Martín, Linxi Fan, Guanzhi Wang, Claudia Pérez-D'Arpino, Shyamal Buch, Sanjana Srivastava, Lyne P. Tchapmi, Micael E. Tchapmi, Kent Vainio, Josiah Wong, Li Fei-Fei, and Silvio Savarese. 2021. igibson 1.0: a simulation environment for interactive tasks in large realistic scenes. In 2021 IEEE/RSJ International Conference on Intelligent Robots and Systems, page accepted. IEEE.
Bokui Shen、Fei Xia、Chengshu Li、Roberto Martín-Martín、Linxi Fan、Guanzhi Wang、Claudia Pérez-D'Arpino、Shyamal Buch、Sanjana Srivastava、Lyne P. Tchapmi、Micael E. Tchapmi、Kent Vainio、Josiah Wong、Li Fei-Fei 和 Silvio Savarese。2021年。iGibson 1.0：面向大型真实场景中交互任务的仿真环境。载于：2021年IEEE/RSJ智能机器人与系统国际会议，第页待接收。IEEE。


Weijia Shi, Sewon Min, Michihiro Yasunaga, Min-joon Seo, Rich James, Mike Lewis, Luke Zettle-moyer, and Wen-tau Yih. 2023. Replug: Retrieval-augmented black-box language models. arXiv preprint arXiv:2301.12652.
Weijia Shi、Sewon Min、Michihiro Yasunaga、Min-joon Seo、Rich James、Mike Lewis、Luke Zettle-moyer 和 Wen-tau Yih。2023年。Replug：检索增强的黑盒语言模型。arXiv预印本 arXiv:2301.12652。


Noah Shinn, Beck Labash, and Ashwin Gopinath. 2023. Reflexion: an autonomous agent with dynamic memory and self-reflection. arXiv preprint arXiv:2303.11366.
Noah Shinn、Beck Labash 和 Ashwin Gopinath。2023年。Reflexion：一种具有动态记忆和自我反思的自主智能体。arXiv预印本 arXiv:2303.11366。


Mohit Shridhar, Jesse Thomason, Daniel Gordon, Yonatan Bisk, Winson Han, Roozbeh Mottaghi, Luke Zettlemoyer, and Dieter Fox. 2020. Alfred: A benchmark for interpreting grounded instructions for everyday tasks. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10740-10749.
Mohit Shridhar、Jesse Thomason、Daniel Gordon、Yonatan Bisk、Winson Han、Roozbeh Mottaghi、Luke Zettlemoyer 和 Dieter Fox。2020年。Alfred：用于理解日常任务中具身指令的基准。载于：IEEE/CVF计算机视觉与模式识别会议论文集，第10740-10749页。


Ishika Singh, Valts Blukis, Arsalan Mousavian, Ankit Goyal, Danfei Xu, Jonathan Tremblay, Dieter Fox, Jesse Thomason, and Animesh Garg. 2022a. Progprompt: Generating situated robot task plans using large language models. arXiv preprint arXiv:2209.11302.
Ishika Singh、Valts Blukis、Arsalan Mousavian、Ankit Goyal、Danfei Xu、Jonathan Tremblay、Dieter Fox、Jesse Thomason 和 Animesh Garg。2022a。Progprompt：使用大型语言模型生成具身机器人任务计划。arXiv预印本 arXiv:2209.11302。


Kunal Pratap Singh, Luca Weihs, Alvaro Herrasti, Jonghyun Choi, Aniruddha Kembhavi, and Roozbeh Mottaghi. 2022b. Ask4help: Learning to leverage an expert for embodied tasks. Advances in Neural Information Processing Systems, 35:16221-16232.
Kunal Pratap Singh、Luca Weihs、Alvaro Herrasti、Jonghyun Choi、Aniruddha Kembhavi 和 Roozbeh Mottaghi。2022b。Ask4help：学会利用专家来完成具身任务。神经信息处理系统进展，第35卷：16221-16232。


Chan Hee Song, Jiaman Wu, Clayton Washington, Brian M. Sadler, Wei-Lun Chao, and Yu Su. 2022. Llm-planner: Few-shot grounded planning for embodied agents with large language models. arXiv preprint arXiv:2212.04088.
Chan Hee Song、Jiaman Wu、Clayton Washington、Brian M. Sadler、Wei-Lun Chao 和 Yu Su。2022。Llm-planner：面向具身代理的带大语言模型少样本有基础的规划。arXiv 预印本 arXiv:2212.04088。


Austin Stone, Ted Xiao, Yao Lu, Keerthana Gopalakrish-nan, Kuang-Huei Lee, Quan Vuong, Paul Wohlhart, Sean Kirmani, Brianna Zitkovich, Fei Xia, Chelsea Finn, and Karol Hausman. 2023. Open-world object manipulation using pre-trained vision-language model. In arXiv preprint.
Austin Stone、Ted Xiao、Yao Lu、Keerthana Gopalakrish-nan、Kuang-Huei Lee、Quan Vuong、Paul Wohlhart、Sean Kirmani、Brianna Zitkovich、Fei Xia、Chelsea Finn 和 Karol Hausman。2023。使用预训练视觉-语言模型进行开放世界目标操控。arXiv 预印本。


Sanjay Subramanian, Medhini Narasimhan, Kushal Khangaonkar, Kevin Yang, Arsha Nagrani, Cordelia Schmid, Andy Zeng, Trevor Darrell, and Dan Klein. 2023. Modular visual question answering via code generation. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics, Toronto, Canada. Association for Computational Linguistics.
Sanjay Subramanian、Medhini Narasimhan、Kushal Khangaonkar、Kevin Yang、Arsha Nagrani、Cordelia Schmid、Andy Zeng、Trevor Darrell 和 Dan Klein。2023。通过代码生成实现模块化视觉问答。载于第 61 届计算语言学协会年会，印度多伦多。计算语言学协会。


Alessandro Suglia, Qiaozi Gao, Jesse Thomason, Govind Thattai, and Gaurav S. Sukhatme. 2021. Embodied bert: A transformer model for embodied, language-guided visual task completion. In EMNLP 2021 Workshop on Novel Ideas in Learning-to-Learn through Interaction.
Alessandro Suglia、Qiaozi Gao、Jesse Thomason、Govind Thattai 和 Gaurav S. Sukhatme。2021。具身 BERT：用于具身、语言引导的视觉任务完成的变换器模型。载于 EMNLP 2021 研讨会：通过交互学习到学习的新想法。


Dídac Surís, Sachit Menon, and Carl Vondrick. 2023. Vipergpt: Visual inference via python execution for reasoning. arXiv preprint arXiv:2303.08128.
Dídac Surís、Sachit Menon 和 Carl Vondrick。2023。Vipergpt：通过执行 Python 进行视觉推理。arXiv 预印本 arXiv:2303.08128。


Stefanie Tellex, Thomas Kollar, Steven Dickerson, Matthew Walter, Ashis Banerjee, Seth Teller, and Nicholas Roy. 2011. Understanding natural language commands for robotic navigation and mobile manipulation. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 25, pages 1507-1514.
Stefanie Tellex、Thomas Kollar、Steven Dickerson、Matthew Walter、Ashis Banerjee、Seth Teller 和 Nicholas Roy。2011。理解用于机器人导航与移动操作的自然语言指令。载于 AAAI 人工智能会议论文集，第 25 卷，第 1507-1514 页。


Guanzhi Wang, Yuqi Xie, Yunfan Jiang, Ajay Man-dlekar, Chaowei Xiao, Yuke Zhu, Linxi Fan, and Anima Anandkumar. 2023a. Voyager: An open-ended embodied agent with large language models. arXiv preprint arXiv: Arxiv-2305.16291.
Guanzhi Wang、Yuqi Xie、Yunfan Jiang、Ajay Man-dlekar、Chaowei Xiao、Yuke Zhu、Linxi Fan 和 Anima Anandkumar。2023a。Voyager：带大语言模型的开放式具身智能体。arXiv 预印本 arXiv: Arxiv-2305.16291。


Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le, Ed Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. 2022. Self-consistency improves chain of thought reasoning in language models. arXiv preprint arXiv:2203.11171.
Xuezhi Wang、Jason Wei、Dale Schuurmans、Quoc Le、Ed Chi、Sharan Narang、Aakanksha Chowdhery 和 Denny Zhou。2022。自一致性提升语言模型中的链式思维推理。arXiv 预印本 arXiv:2203.11171。


Zihao Wang, Shaofei Cai, Anji Liu, Xiaojian Ma, and Yitao Liang. 2023b. Describe, explain, plan and select: Interactive planning with large language models enables open-world multi-task agents. arXiv preprint arXiv:2302.01560.
Zihao Wang、Shaofei Cai、Anji Liu、Xiaojian Ma 和 Yitao Liang。2023b。描述、解释、规划并选择：使用大语言模型进行交互式规划，使开放世界多任务智能体成为可能。arXiv 预印本 arXiv:2302.01560。


Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Ed Chi, Quoc Le, and Denny Zhou. 2022. Chain of thought prompting elicits reasoning in large language models. arXiv preprint arXiv:2201.11903.
Jason Wei、Xuezhi Wang、Dale Schuurmans、Maarten Bosma、Ed Chi、Quoc Le 和 Denny Zhou。2022。链式思维提示能引导大语言模型中的推理。arXiv 预印本 arXiv:2201.11903。


Erik Wijmans, Abhishek Kadian, Ari Morcos, Stefan Lee, Irfan Essa, Devi Parikh, Manolis Savva, and Dhruv Batra. 2020. Decentralized distributed ppo: Solving pointgoal navigation. In International Conference on Lefvarning Representations (ICLR).
Erik Wijmans、Abhishek Kadian、Ari Morcos、Stefan Lee、Irfan Essa、Devi Parikh、Manolis Savva 和 Dhruv Batra。2020。去中心化分布式 PPO：解决点目标导航问题。载于国际会议 Learning Representations（ICLR）。


Jimmy Wu, Rika Antonova, Adam Kan, Marion Lep-ert, Andy Zeng, Shuran Song, Jeannette Bohg, Szy-mon Rusinkiewicz, and Thomas Funkhouser. 2023a. Tidybot: Personalized robot assistance with large language models. arXiv preprint arXiv:2305.05658.
Jimmy Wu、Rika Antonova、Adam Kan、Marion Lep-ert、Andy Zeng、Shuran Song、Jeannette Bohg、Szy-mon Rusinkiewicz 和 Thomas Funkhouser。2023a。Tidybot：借助大语言模型提供个性化机器人协助。arXiv 预印本 arXiv:2305.05658。


Yue Wu, Yewen Fan, Paul Pu Liang, Amos Azaria, Yuanzhi Li, and Tom M Mitchell. 2023b. Read and reap the rewards: Learning to play atari with the help of instruction manuals. In NeurIPS.
Yue Wu、Yewen Fan、Paul Pu Liang、Amos Azaria、Yuanzhi Li 和 Tom M Mitchell。2023b。阅读并收获回报：在指令手册的帮助下学会玩 Atari。载于 NeurIPS。


Yue Wu, So Yeon Min, Yonatan Bisk, Ruslan Salakhut-dinov, Amos Azaria, Yuanzhi Li, Tom Mitchell, and Shrimai Prabhumoye. 2023c. Plan, eliminate, and track-language models are good teachers for embodied agents. arXiv preprint arXiv:2305.02412.
Yue Wu、So Yeon Min、Yonatan Bisk、Ruslan Salakhut-dinov、Amos Azaria、Yuanzhi Li、Tom Mitchell 和 Shrimai Prabhumoye。2023c。规划、消除与跟踪——语言模型是具身智能体的良师。arXiv 预印本 arXiv:2305.02412。


Yue Wu, So Yeon Min, Shrimai Prabhumoye, Yonatan Bisk, Ruslan Salakhutdinov, Amos Azaria, Tom Mitchell, and Yuanzhi Li. 2023d. Spring: Gpt-4 out-performs rl algorithms by studying papers and reasoning. In NeurIPS.
Yue Wu, So Yeon Min, Shrimai Prabhumoye, Yonatan Bisk, Ruslan Salakhutdinov, Amos Azaria, Tom Mitchell, 和 Yuanzhi Li. 2023d. Spring: 通过研究论文和推理，GPT-4 的表现优于强化学习算法。发表于 NeurIPS。


Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Thomas L Griffiths, Yuan Cao, and Karthik Narasimhan. 2023. Tree of thoughts: Deliberate problem solving with large language models. arXiv preprint arXiv:2305.10601.
Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Thomas L Griffiths, Yuan Cao, 和 Karthik Narasimhan. 2023. 思维树：借助大语言模型进行审慎的问题求解。arXiv 预印本 arXiv:2305.10601。


Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. 2022. React: Synergizing reasoning and acting in language models. arXiv preprint arXiv:2210.03629.
Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, 和 Yuan Cao. 2022. ReAct：在语言模型中协同推理与行动。arXiv 预印本 arXiv:2210.03629。


Tianhe Yu, Deirdre Quillen, Zhanpeng He, Ryan Julian, Karol Hausman, Chelsea Finn, and Sergey Levine. 2020. Meta-world: A benchmark and evaluation for multi-task and meta reinforcement learning. In Conference on Robot Learning, pages 1094-1100. PMLR.
Tianhe Yu, Deirdre Quillen, Zhanpeng He, Ryan Julian, Karol Hausman, Chelsea Finn, 和 Sergey Levine. 2020. Meta-world：多任务与元强化学习的基准与评测。发表于 Conference on Robot Learning，第 1094-1100 页。PMLR。


Tianhe Yu, Ted Xiao, Austin Stone, Jonathan Tomp-son, Anthony Brohan, Su Wang, Jaspiar Singh, Clayton Tan, Dee M, Jodilyn Peralta, Brian Ichter, Karol Hausman, and Fei Xia. 2023. Scaling robot learning with semantically imagined experience. In arXiv preprint arXiv:2302.11550.
Tianhe Yu, Ted Xiao, Austin Stone, Jonathan Tom-pson, Anthony Brohan, Su Wang, Jaspiar Singh, Clayton Tan, Dee M, Jodilyn Peralta, Brian Ichter, Karol Hausman, 和 Fei Xia. 2023. 通过语义想象经验扩展机器人学习。发表于 arXiv 预印本 arXiv:2302.11550。


Andy Zeng, Maria Attarian, Brian Ichter, Krzysztof Choromanski, Adrian Wong, Stefan Welker, Federico Tombari, Aveek Purohit, Michael Ryoo, Vikas Sindhwani, et al. 2022. Socratic models: Composing zero-shot multimodal reasoning with language. arXiv preprint arXiv:2204.00598.
Andy Zeng, Maria Attarian, Brian Ichter, Krzysztof Choromanski, Adrian Wong, Stefan Welker, Federico Tombari, Aveek Purohit, Michael Ryoo, Vikas Sindhwani, 等. 2022. 苏格拉底式模型：将零样本多模态推理与语言组合。arXiv 预印本 arXiv:2204.00598。


Yichi Zhang, Jianing Yang, Jiayi Pan, Shane Storks, Nikhil Devraj, Ziqiao Ma, Keunwoo Peter Yu, Yuwei Bao, and Joyce Chai. 2022. Danli: Deliberative agent for following natural language instructions. arXiv preprint arXiv:2210.12485.
Yichi Zhang, Jianing Yang, Jiayi Pan, Shane Storks, Nikhil Devraj, Ziqiao Ma, Keunwoo Peter Yu, Yuwei Bao, 和 Joyce Chai. 2022. Danli：用于遵循自然语言指令的审慎智能体。arXiv 预印本 arXiv:2210.12485。


Kaizhi Zheng, Kaiwen Zhou, Jing Gu, Yue Fan, Jialu Wang, Zonglin Li, Xuehai He, and Xin Eric Wang. 2022. Jarvis: A neuro-symbolic commonsense reasoning framework for conversational embodied agents.
Kaizhi Zheng, Kaiwen Zhou, Jing Gu, Yue Fan, Jialu Wang, Zonglin Li, Xuehai He, 和 Xin Eric Wang. 2022. Jarvis：面向对话式具身智能体的神经符号常识推理框架。


## A Analysis of User Personalization Failures
## 用户个性化失败分析


The three instances where the PLANNER made errors in the user personalization experiment (Section 4.4) involved logical mistakes or inappropriate alterations to parts of the original plan that were not requested for modification. For instance, in one case - involving a step to modify the plan from making one coffee to two coffees - the PLANNER includes placing two mugs in the coffee maker simultaneously, which is not a valid plan. In the other two instances, the PLANNER omits an object from the original plan that was not mentioned in the modification.
在用户个性化实验（第4.4节）中，PLANNER 发生错误的三处案例，分别源于逻辑失误或对原始计划部分内容的不恰当改动，而这些改动并未被要求修改。比如，其中一例涉及将计划从做一杯咖啡改为做两杯咖啡：PLANNER 同时在咖啡机中放入两个杯子，这不是一个有效的计划。另两例中，PLANNER 都从原始计划里遗漏了并未在改动说明中提及的对象。


## B User Feedback Details
## B 用户反馈详情


In the user feedback evaluation, once the agent has indicated completion of the task from the original input dialogue, the agent will query feedback from the user. If the simulator indicates success of the task, the agent will end the episode. If the simulator indicates the task is not successful, feedback will be given to the agent for additional planning. This feedback is programatically generated from the TEACh simulator metadata, which gives us information about if the task is successful, and what object state changes are missing in order to complete the task (e.g., bread slice is not toasted, etc.). For each object state that is incorrect, we form a sentence of the following form: "You failed to complete the subtask: subtask. For the object object: description of desired object state." We combine all subtask sentences to create the feedback. HELPER follows the same pipeline (including examples, retrieval, planning, etc.) to process the feedback as with the input dialogue in the normal TfD evaluation. We show experiments with one and two user feedback requests in Section 4.3 of the main paper (a second request is queried if the first user feedback fails to produce task success).
在用户反馈评估中，一旦智能体从原始对话中表明任务已完成，它将向用户征求反馈。如果模拟器指示任务成功，智能体将结束该回合。如果模拟器指示任务未成功，将把反馈提供给智能体以便进行额外规划。该反馈由 TEACh 模拟器元数据以程序方式生成：它告诉我们任务是否成功，以及为完成任务还缺少哪些物体状态变化（例如，面包片未烤制等）。对于每一个不正确的物体状态，我们都会形成如下句子：“你未能完成子任务：子任务。针对物体：所需物体状态的描述。”我们将所有子任务句子合并以生成反馈。HELPER 也遵循相同的处理流程（包括示例、检索、规划等），以与正常 TfD 评估中对输入对话的处理方式一致以处理反馈。我们在主论文第 4.3 节展示了一次和两次用户反馈请求的实验（若第一次用户反馈未能产生任务成功，则会追问第二次请求）。


## C User Personalization Inputs
## C 用户个性化输入


We provide a full list of the user personalization requests in Listing 1 for the user personalization experiments in Section 4.4.
我们在列表1中提供了第4.4节用户个性化实验所用的完整用户个性化请求列表。


## D Prompts
## D 提示


We provide our full API (Listing 2), corrective API (Listing 3), PLANNER prompt (Listing 4), replanning prompt (Listing 5), and LOCATOR prompt (Listing 6).
我们提供完整的 API（清单 2）、纠正 API（清单 3）、PLANNER 提示（清单 4）、重新规划提示（清单 5）以及 LOCATOR 提示（清单 6）。


## E Pre-conditions
## E 前置条件


An example of a pre-condition check for a macro-action is provided in Listing 7.
在清单 7 中给出了对宏动作进行前置条件检查的示例。


## F Example LLM inputs & Outputs
## F 示例 LLM 输入与输出


We provide examples of dialogue input, retrieved examples, and LLM output for a TEACh sample in Listing 8, Listing 9, and Listing 10.
我们在第 8、9 和 10 例中提供了 TEACh 样例的对话输入、检索到的示例以及 LLM 输出。


## G Simulation environment
## G 仿真环境


The TEACh dataset builds on the Ai2thor simulation environment (Kolve et al., 2017). At each time step the agent may choose from the following actions: Forward(), Backward(), Turn Left(), Turn Right(), Look Up(), Look Down(), Strafe Left(), Strafe Right(), Pickup(X), Place(X), Open(X), Close(X), ToggleOn(X), ToggleOff(X), Slice(X), and Pour(X), where X refers an object specified via a relative coordinate $\left( {x,y}\right)$ on the egocentric RGB frame. Navigation actions move the agent in discrete steps. We rotate in the yaw direction by 90 degrees, and rotate in the pitch direction by 30 degrees. The RGB and depth sensors are at a resolution of 480x480, a field of view of 90 degrees, and lie at a height of 0.9015 meters. The agent's coordinates are parameterized by a single $\left( {x,y,z}\right)$ coordinate triplet with $x$ and $z$ corresponding to movement in the horizontal plane and $y$ reserved for the vertical direction. The TEACh benchmark allows a maximum of 1000 steps and 30 API failures per episode.
TEACh 数据集建立在 Ai2thor 仿真环境（Kolve et al., 2017）之上。在每个时间步，智能体可以从以下动作中选择：Forward()、Backward()、Turn Left()、Turn Right()、Look Up()、Look Down()、Strafe Left()、Strafe Right()、Pickup(X)、Place(X)、Open(X)、Close(X)、ToggleOn(X)、ToggleOff(X)、Slice(X) 和 Pour(X)，其中 X 指通过自我中心 RGB 帧上的相对坐标 $\left( {x,y}\right)$ 指定的对象。导航动作使智能体以离散步长移动。我们在 yaw 方向旋转 90 度，在 pitch 方向旋转 30 度。RGB 和深度传感器的分辨率为 480x480，视场角为 90 度，安装高度为 0.9015 米。智能体的坐标由一个 $\left( {x,y,z}\right)$ 坐标三元组参数化，其中 $x$ 和 $z$ 对应水平平面内的移动，$y$ 则保留给垂直方向。TEACh 基准每个回合最多允许 1000 步和 30 次 API 失败。


## H Executor details
## H 执行器详情


### H.1 Semantic mapping and planning
### H.1 语义映射与规划


Obstacle map HELPER maintains a 2D overhead occupancy map of its environment $\in  {\mathbb{R}}^{H \times  W}$ that it updates at each time step from the input RGB-D stream. The map is used for exploration and navigation in the environment.
障碍物地图 HELPER 会维护一个二维俯视占据信息地图 $\in  {\mathbb{R}}^{H \times  W}$，用于刻画其环境，并在每个时间步基于输入 RGB-D 流进行更新。该地图用于环境中的探索与导航。


At every time step $t$ ,we unproject the input depth maps using intrinsic and extrinsic information of the camera to obtain a 3D occupancy map registered to the coordinate frame of the agent, similar to earlier navigation agents (Chaplot et al., 2020a). The 2D overhead maps of obstacles and free space are computed by projecting the 3D occupancy along the height direction at multiple height levels and summing. For each input RGB image, we run a SOLQ object segmentor (Dong et al., 2021) (pretrained on COCO (Lin et al., 2014) then finetuned on TEACh rooms) to localize each of 116 semantic object categories. For failure detection, we use a simple matching approach from Min et al. (2021) to compare RGB pixel values before and after taking an action.
在每个时间步 $t$，我们使用相机的内参和外参对输入深度图进行反投影，得到与智能体坐标系配准的三维占据信息图，类似于较早的导航智能体（Chaplot et al., 2020a）。将三维占据信息沿高度方向在多个高度层级投影并求和，可计算得到二维俯视障碍物与空闲空间地图。对于每一帧输入 RGB 图像，我们运行一个 SOLQ 目标分割器（Dong et al., 2021）（先在 COCO（Lin et al., 2014）上预训练，再在 TEACh 场景上微调），以定位 116 个语义目标类别中的每一个。为检测失败，我们采用 Min et al.（2021）提出的一种简单匹配方法，对比执行动作前后 RGB 像素值。


Object location and state tracking We maintain an object memory as a list of object detection 3D centroids and their predicted semantic labels $\left\{  {\left\lbrack  {{\left( X,Y,Z\right) }_{i},{\ell }_{i} \in  \{ 1\ldots N\} }\right\rbrack  ,i = 1..K}\right\}$ ,where $K$ is the number of objects detected thus far. The object centroids are expressed with respect to the coordinate system of the agent, and, similar to the semantic maps, updated over time using egomotion. We track previously detected objects by their 3D centroid $C \in  {\mathbb{R}}^{3}$ . We estimate the centroid by taking the 3D point corresponding to the median depth within the segmentation mask and bring it to a common coordinate frame. We do a simple form of non-maximum suppression on the object memory, by comparing the euclidean distance of centroids in the memory to new detected centroids of the same category, and keep the one with the highest score if they fall within a distance threshold.
目标位置与状态跟踪 我们维护一个目标记忆，将目标检测得到的三维质心及其预测语义标签存为列表 $\left\{  {\left\lbrack  {{\left( X,Y,Z\right) }_{i},{\ell }_{i} \in  \{ 1\ldots N\} }\right\rbrack  ,i = 1..K}\right\}$，其中 $K$ 表示到目前为止检测到的目标数量。质心相对于智能体坐标系表示，并且与语义地图类似，会随时间利用自运动信息更新。我们通过三维质心 $C \in  {\mathbb{R}}^{3}$ 来跟踪先前检测到的目标。我们通过在分割掩码内取中深度对应的三维点来估计质心，并将其转换到统一的坐标系。我们在目标记忆上做一种简单形式的非极大值抑制：将记忆中质心的欧氏距离与同类别新检测到的质心进行比较，若距离落入阈值内，则保留得分最高的那个。


For each object in the object memory, we maintain an object state dictionary with a pre-defined list of attributes. These attributes include: category label, centroid location, holding, detection score, can use, sliced, toasted, clean, cooked. For the binary attributes, these are initialized by sending the object crop, defined by the detector mask, to the VLM model, and checking its match to each of [f"The \{object_category\} is \{attribute\}", f"The \{object_category\} is not \{attribute\}"]. We found that initializing these attributes with the VLM gave only a marginal difference to initializing them to default values in the TEACh benchmark, so we do not use it for the TEACh evaluations. However, we anticipate a general method beyond dataset biases of TEACh would much benefit from such vision-based attribute classification.
对于目标记忆中的每个目标，我们维护一个目标状态字典，并包含一组预定义属性列表。这些属性包括：类别标签、质心位置、是否持有、检测得分、是否可用、是否切开、是否烤制、是否清洁、是否烹饪。对于二值属性，我们通过把由检测器掩码定义的目标裁剪块送入 VLM 模型，并检查它与下列任意项的匹配：[f"The \{object_category\} is \{attribute\}", f"The \{object_category\} is not \{attribute\}"]。我们发现用 VLM 初始化这些属性，相比在 TEACh 基准中用默认值初始化，只带来很小的差异，因此我们不在 TEACh 评估中使用它。不过，我们预期在超越 TEACh 数据集偏置的一般方法上，基于视觉的属性分类会带来很大收益。


Exploration and path planning HELPER explores the scene using a classical mapping method. We take the initial position of the agent to be the center coordinate in the map. We rotate the agent in-place and use the observations to instantiate an initial map. Second, the agent incrementally completes the maps by randomly sampling an unexplored, traversible location based on the 2D occupancy map built so far, and then navigates to the sampled location, accumulating the new information into the maps at each time step. The number of observations collected at each point in the 2D occupancy map is thresholded to determine whether a given map location is explored or not. Unexplored positions are sampled until the environment has been fully explored, meaning that the number of unexplored points is fewer than a predefined threshold.
探索与路径规划 HELPER 使用经典的映射方法来探索场景。我们将智能体的初始位置设为地图中的中心坐标。首先将智能体原地旋转，并利用观测来生成一个初始地图。随后，智能体会基于迄今为止构建的二维占据信息地图，随机采样一个尚未探索且可通行的位置，逐步补全地图；然后导航到该采样位置，并在每个时间步将新信息累积到地图中。二维占据信息地图中每个位置的观测次数会与阈值进行比较，以判断该位置是否已被探索。持续采样未探索的位置，直到环境被完全探索完成，也就是未探索点的数量少于预定义阈值。


To navigate to a goal location, we compute the geodesic distance to the goal from all map locations using graph search (Inoue and Ohashi, 2022) given the top-down occupancy map and the goal location in the map. We then simulate action sequences and greedily take the action sequence which results in the largest reduction in geodesic distance.
为导航到目标位置，我们在给定俯视占据信息地图以及地图中的目标位置的情况下，使用图搜索（Inoue and Ohashi, 2022）从所有地图位置计算到目标的测地距离。然后我们模拟一系列动作，并贪心选择能带来测地距离最大降幅的动作序列。


### H.2 2D-to-3D unprojection
### H.2 2D到3D反投影


For the $i$ -th view,a 2D pixel coordinate $\left( {u,v}\right)$ with depth $z$ is unprojected and transformed to its coordinate ${\left( X,Y,Z\right) }^{T}$ in the reference frame:
对于第 $i$ 个视角，将带深度 $z$ 的二维像素坐标 $\left( {u,v}\right)$ 反投影，并转换到参考坐标系中的其坐标 ${\left( X,Y,Z\right) }^{T}$：


$$
\left( {X,Y,Z,1}\right)  = {\mathbf{G}}_{i}^{-1}{\left( z\frac{u - {c}_{x}}{{f}_{x}},z\frac{v - {c}_{y}}{{f}_{y}},z,1\right) }^{T}
$$



(1)



where $\left( {{f}_{x},{f}_{y}}\right)$ and $\left( {{c}_{x},{c}_{y}}\right)$ are the focal lengths and center of the pinhole camera model and ${\mathbf{G}}_{i} \in \; {SE}\left( 3\right)$ is the camera pose for view $i$ relative to the reference view. This module unprojects each depth image ${I}_{i} \in  {\mathbb{R}}^{H \times  W \times  3}$ into a pointcloud in the reference frame ${P}_{i} \in  {\mathbb{R}}^{{M}_{i} \times  3}$ with ${M}_{i}$ being the number of pixels with an associated depth value.
其中，$\left( {{f}_{x},{f}_{y}}\right)$ 和 $\left( {{c}_{x},{c}_{y}}\right)$ 分别是针孔相机模型的焦距和中心，${\mathbf{G}}_{i} \in \; {SE}\left( 3\right)$ 是视角 $i$ 相对于参考视角的相机位姿。该模块将每个深度图 ${I}_{i} \in  {\mathbb{R}}^{H \times  W \times  3}$ 反投影到参考坐标系 ${P}_{i} \in  {\mathbb{R}}^{{M}_{i} \times  3}$ 的点云中，${M}_{i}$ 为具有对应深度值的像素数量。


## I Additional Experiments
## I 额外实验


### I.1 Alternate EDH Evaluation Split
### I.1 备用 EDH 评估划分


Currently, the leaderboard for the TEACh EDH benchmark is not active. Thus, we are not able to evaluate on the true test set for TEACh. We used the original validation seen and unseen splits, which have been used in most previous works (Pa-shevich et al., 2021; Zheng et al., 2022; Min et al., 2022; Zhang et al., 2022). In Table 4 we report the alternative validation and test split as mentioned in the TEACh github README, and also reported by DANLI (Zhang et al., 2022).
目前，TEACh EDH 基准的排行榜尚未启用。因此，我们无法在 TEACh 的真实测试集上进行评估。我们使用了原始的已见与未见划分，这也是大多数以往工作采用的划分（Pa-shevich 等，2021；Zheng 等，2022；Min 等，2022；Zhang 等，2022）。在表 4 中，我们报告了 TEACh github README 中提到的、以及 DANLI（Zhang 等，2022）同样报告的备用验证集与测试划分。


Table 4: Alternative TEACh Execution from Dialog History (EDH) evaluation split. Trajectory length weighted metrics are included in ( parentheses ). SR = success rate. GC = goal condition success rate. Note that Test Seen and Unseen are not the true TEACh test sets, but an alternative split of the validation set used until the true test evaluation is released, as mentioned in the TEACh github README, and also reported by DANLI (Zhang et al., 2022).
表 4：TEACh 执行自对话历史（EDH）的备用评估划分。轨迹长度加权指标包含在（括号）中。SR = 成功率。GC = 目标条件成功率。注意：已见与未见测试集并非真正的 TEACh 测试集，而是对验证集的一种替代划分，直至发布真实测试评估；该说明见 TEACh github README，并同样由 DANLI（Zhang 等，2022）报告。


<table><tr><td rowspan="3"></td><td colspan="4">Validation</td><td colspan="4">Test</td></tr><tr><td colspan="2">Unseen</td><td colspan="2">Seen</td><td colspan="2">Unseen</td><td colspan="2">Seen</td></tr><tr><td>SR</td><td>GC</td><td>SR</td><td>GC</td><td>SR</td><td>GC</td><td>SR</td><td>GC</td></tr><tr><td>E.T.</td><td>8.35 (0.86)</td><td>6.34 (3.69)</td><td>8.28 (1.13)</td><td>8.72 (3.82)</td><td>7.38 (0.97)</td><td>6.06 (3.17)</td><td>8.82 (0.29)</td><td>9.46 (3.03)</td></tr><tr><td>DANLI</td><td>17.25 (7.16)</td><td>23.88 (19.38)</td><td>16.89 (9.12)</td><td>25.10 (22.56)</td><td>16.71 (7.33)</td><td>23.00 (20.55)</td><td>18.63 (9.41)</td><td>24.77 (21.90)</td></tr><tr><td>HELPER</td><td>17.25 (3.22)</td><td>25.24 (8.12)</td><td>19.21 (4.72)</td><td>33.54 (10.95)</td><td>17.55 (2.59)</td><td>26.49 (7.67)</td><td>17.97 (3.44)</td><td>30.81 (8.93)</td></tr></table>
<table><tbody><tr><td rowspan="3"></td><td colspan="4">验证</td><td colspan="4">测试</td></tr><tr><td colspan="2">未见</td><td colspan="2">已见</td><td colspan="2">未见</td><td colspan="2">已见</td></tr><tr><td>SR</td><td>GC</td><td>SR</td><td>GC</td><td>SR</td><td>GC</td><td>SR</td><td>GC</td></tr><tr><td>E.T.</td><td>8.35 (0.86)</td><td>6.34 (3.69)</td><td>8.28 (1.13)</td><td>8.72 (3.82)</td><td>7.38 (0.97)</td><td>6.06 (3.17)</td><td>8.82 (0.29)</td><td>9.46 (3.03)</td></tr><tr><td>DANLI</td><td>17.25 (7.16)</td><td>23.88 (19.38)</td><td>16.89 (9.12)</td><td>25.10 (22.56)</td><td>16.71 (7.33)</td><td>23.00 (20.55)</td><td>18.63 (9.41)</td><td>24.77 (21.90)</td></tr><tr><td>HELPER</td><td>17.25 (3.22)</td><td>25.24 (8.12)</td><td>19.21 (4.72)</td><td>33.54 (10.95)</td><td>17.55 (2.59)</td><td>26.49 (7.67)</td><td>17.97 (3.44)</td><td>30.81 (8.93)</td></tr></tbody></table>


Listing 1: Full list of user personalization requests for the user personalization evaluation.
清单1：用户个性化评估中，针对用户的所有个性化请求完整列表。


---



original input to LLM:
输入给LLM的原始内容：


## No change:
## 无变化:


"Make me the Larry sandwich"
"给我做 Larry 三明治"


"Make me the David salad"
"给我做 David 沙拉"


"Make me the Dax salad"
"给我做 Dax 沙拉"


"Make me the Mary breakfast"
"给我做 Mary 早餐"


&nbsp;&nbsp;&nbsp;&nbsp;"Make me the Lion breakfast"
&nbsp;&nbsp;&nbsp;&nbsp;"给我做 Lion 早餐"


"Complete the Lax rearrangement"
"完成 Lax 重排"


&nbsp;&nbsp;&nbsp;&nbsp;"Complete the Pax rearrangement"
&nbsp;&nbsp;&nbsp;&nbsp;"完成 Pax 重排"


&nbsp;&nbsp;&nbsp;&nbsp;"Perform the Gax cleaning"
&nbsp;&nbsp;&nbsp;&nbsp;"执行 Gax 清洁"


"Make me the Gabe sandwich"
"给我做 Gabe 三明治"


"Perform the Kax cleaning"
"执行 Kax 清洁"


One change:
一次变化:


"Make me the Larry sandwich with four slices of lettuce"
"给我做 Larry 三明治，加四片生菜"


&nbsp;&nbsp;&nbsp;&nbsp;"Make me the David salad with a slice of potato"
&nbsp;&nbsp;&nbsp;&nbsp;"给我做 David 沙拉，加一片土豆"


"Make me the Dax salad without lettuce"
"给我做 Dax 沙拉，不要生菜"


"Make me the Mary breakfast with no coffee"
"给我做 Mary 早餐，不要咖啡"


&nbsp;&nbsp;&nbsp;&nbsp;"Make me the Lion breakfast with three slice of tomato"
&nbsp;&nbsp;&nbsp;&nbsp;"给我做 Lion 早餐，加三片番茄"


&nbsp;&nbsp;&nbsp;&nbsp;"Complete the Lax rearrangement with two pillows"
&nbsp;&nbsp;&nbsp;&nbsp;"完成 Lax 重新摆放，加两个枕头"


&nbsp;&nbsp;&nbsp;&nbsp;"Complete the Pax rearrangement but use one pencil instead of the the two pencils"
&nbsp;&nbsp;&nbsp;&nbsp;"完成 Pax 重新摆放，但用一支铅笔代替两支铅笔"


&nbsp;&nbsp;&nbsp;&nbsp;"Perform the Gax cleaning with three plates instead of two"
&nbsp;&nbsp;&nbsp;&nbsp;"执行 Gax 清洁，用三只盘子代替两个"


&nbsp;&nbsp;&nbsp;&nbsp;"Make me the Gabe sandwich with only 1 slice of tomato"
&nbsp;&nbsp;&nbsp;&nbsp;"给我做 Gabe 三明治，只放 1 片番茄"


&nbsp;&nbsp;&nbsp;&nbsp;"Perform the Kax cleaning with only a mug"
&nbsp;&nbsp;&nbsp;&nbsp;"执行 Kax 清洁，只用一个杯子"


---



[['Driver', 'What is my task?'], ['Commander', "Make me a sandwich. The name of this sandwich is
[['Driver', '我的任务是什么？'], ['Commander', "给我做一个三明治。这款三明治的名字是


called the Larry sandwich. The sandwich has two slices of toast, 3 slices of tomato, and 3
叫 Larry 三明治。这个三明治有两片吐司、3 片番茄和 3


slice of lettuce on a clean plate."]]
片生菜，放在一个干净的盘子上。"]]


[['Driver', 'What is my task?'], ['Commander', 'Make me a salad. The name of this salad is called
[['Driver', '我的任务是什么？'], ['Commander', "给我做一份沙拉。这份沙拉的名字叫做


the David salad. The salad has two slices of tomato and three slices of lettuce on a clean
David 沙拉。这份沙拉有两片番茄和三片生菜，放在一个干净的


plate.']]
盘子上。"]]


[['Driver', 'What is my task?'], ['Commander', "Make me a salad. The name of this salad is called
[['Driver', '我的任务是什么？'], ['Commander', "给我做一份沙拉。这份沙拉的名字叫做


the Dax salad. The salad has two slices of cooked potato. You'll need to cook the potato on the
Dax 沙拉。这份沙拉有两片熟土豆。你需要在


stove. The salad also has a slice of lettuce and a slice of tomato. Put all components on a
炉子上把土豆煮熟。这份沙拉还包括一片生菜和一片番茄。把所有配料放在一个


clean plate."]
干净的盘子上。"]


[['Driver', 'What is my task?'], ['Commander', 'Make me breakfast. The name of this breakfast is
[['Driver', '我的任务是什么？'], ['Commander', '给我做早餐。这个早餐的名字是


called the Mary breakfast. The breakfast has a mug of coffee, and two slices of toast on a
叫 Mary 早餐。早餐有一杯咖啡，以及放在一个干净盘子上的两片吐司。']


clean plate.']
']


[['Driver', 'What is my task?'], ['Commander', 'Make me breakfast. The name of this breakfast is
[['Driver', '我的任务是什么？'], ['Commander', '给我做早餐。这个早餐的名字是


called the Lion breakfast. The breakfast has a mug of coffee, and four slices of tomato on a
叫 Lion 早餐。早餐有一杯咖啡，以及放在一个干净盘子上的四片番茄。']]


clean plate.']]
']]


[['Driver', 'What is my task?'], ['Commander', 'Rearrange some objects. The name of this
[['Driver', '我的任务是什么？'], ['Commander', '重新摆放一些物体。这个


rearrangement is called the Lax rearrangement. Place three pillows on the sofa.']]
重新摆放的名字叫 Lax 重新摆放。把三个枕头放到沙发上。']]


[['Driver', 'What is my task?'], ['Commander', 'Rearrange some objects. The name of this
[['Driver', '我的任务是什么？'], ['Commander', '重新摆放一些物体。这个


rearrangement is called the Pax rearrangement. Place two pencils and two pens on the desk.']
重新摆放的名字叫 Pax 重新摆放。把两支铅笔和两支钢笔放到桌子上。']


[['Driver', 'What is my task?'], ['Commander', 'Clean some objects. The name of this cleaning is
[['Driver', '我的任务是什么？'], ['Commander', '清洁一些物体。这个清洁的名字是


called the Gax cleaning. Clean two plates and two cups.']]
叫 Gax 清洁。清洁两个盘子和两个杯子。']]


[['Driver', 'What is my task?']], ['Commander', "Make me a sandwich. The name of this sandwich is
[['Driver', '我的任务是什么？']], ['Commander', "给我做一个三明治。这个三明治的名字是


called the Gabe sandwich. The sandwich has two slices of toast, 2 slices of tomato, and 1 slice
叫 Gabe 三明治。这个三明治有两片吐司、2 片番茄和 1 片


of lettuce on a clean plate."]]
生菜，放在一个干净盘子上。"]]


[['Driver', 'What is my task?'], ['Commander', 'Clean some objects. The name of this cleaning is
[['Driver', '我的任务是什么？'], ['Commander', '清洁一些物体。这个清洁的名字是


called the Kax cleaning. Clean a mug and a pan.']]
称为 Kax 清洁。清洗一个杯子和一个平底锅。']]


Two changes:
两处改动：


"Make me the Larry sandwich with four slices of lettuce and two slices of tomato"
“给我做 Larry 三明治：四片生菜，两片番茄”


"Make me the David salad but add a slice of potato and add one slice of egg"
“给我做 David 沙拉，但加一片土豆，并再加一片蛋”


"Make me the Dax salad without lettuce and without potato"
“给我做 Dax 沙拉，不要生菜，也不要土豆”


"Make me the Mary breakfast with no coffee and add an egg"
“给我做 Mary 早餐：不要咖啡，并加一个蛋”


"Make me the Lion breakfast with three slice of tomato and two mugs of coffee"
“给我做 Lion 早餐：三片番茄，两杯咖啡”


"Complete the Lax rearrangement with two pillows and add a remote"
“完成 Lax 的重新摆放：加两个枕头，并加一个遥控器”


"Complete the Pax rearrangement but use one pencil instead of the two pencils and add a book"
“完成 Pax 的重新摆放，但用一支铅笔代替两支铅笔，并再加一本书”


"Perform the Gax cleaning with three plates instead of the two plates and include a fork"
“执行 Gax 清洁：用三只盘子代替两只盘子，并加一把叉子”


"Make me the Gabe sandwich with only 1 slice of tomato and two slices of lettuce"
“给我做 Gabe 三明治：只有 1 片番茄和两片生菜”


"Perform the Kax cleaning without the pan and include a spoon"
“执行 Kax 清洁：不要平底锅，并加一把勺子”


## Three changes:
## 三处改动：


"Make me the Larry sandwich with four slices of lettuce, two slices of tomato, and place all components directly on the countertop"
"用四片生菜、两片番茄给我做一份拉里三明治，并把所有部件直接放在台面上"


"Make me the David salad and add a slice of potato, add one slice of egg, and bring a fork with it"
"给我做一份大卫沙拉，加一片土豆，再加一片鸡蛋，并把叉子一起拿来"


"Make me the Dax salad without lettuce, without potato, and add an extra slice of tomato"
"给我做一份达克斯沙拉：不要生菜、不要土豆，再加一片额外的番茄"


"Make me the Mary breakfast with no coffee, add an egg, and add a cup filled with water"
"给我做一份玛丽早餐：不要咖啡，加一个鸡蛋，并加一杯装满水"


"Make me the Lion breakfast with three slice of tomato, two mugs of coffee, and add a fork"
"给我做一份狮子早餐：三片番茄，两杯咖啡，并加一把叉子"


"Complete the Lax rearrangement with two pillows, a remote, and place it on the arm chair instead"
"完成拉克斯的摆放重组：加两个枕头、一个遥控器，并把它放到扶手椅上"


"Complete the Pax rearrangement but use one pencil instead of the two pencils and include a book and a baseball bat"
"完成帕克斯的摆放重组，但用一支铅笔代替两支铅笔，并加入一本书和一个棒球棒"


"Perform the Gax cleaning with three plates instead of the two plates, include a fork, and do not clean any cups"
"进行加克斯的清洁：用三个盘子代替两个盘子，加入一把叉子，并且不要清洗任何杯子"


"Make me the Gabe sandwich with only 1 slice of tomato, two slices of lettuce, and add a slice of egg"
"给我做一份加比三明治：只用1片番茄、两片生菜，并加一片鸡蛋"


"Perform the Kax cleaning without the pan, include a spoon, and include a pot"
"进行凯克斯的清洁：不要用平底锅，加入一把勺子，并再加入一个锅"


Listing 2: Full API for the parametrized macro-actions $G$ used in the prompts.
列表2：用于提示中的参数化宏动作 $G$ 的完整API。


---



class InteractionObject:
class InteractionObject:


&nbsp;&nbsp;&nbsp;&nbsp;1111



&nbsp;&nbsp;&nbsp;&nbsp;This class represents an expression that uniquely identifies an object in the house.
&nbsp;&nbsp;&nbsp;&nbsp;这个类表示一种表达式，用于在屋内唯一标识某个物体。


&nbsp;&nbsp;&nbsp;&nbsp;def __init__(self, object_class: str, landmark: str = None, attributes: list = []):
&nbsp;&nbsp;&nbsp;&nbsp;def __init__(self, object_class: str, landmark: str = None, attributes: list = []):


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;object_class: object category of the interaction object (e.g., "Mug", "Apple")
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;object_class：交互对象的类别（例如“杯子”、“苹果”）


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;landmark: (optional if mentioned) landmark object category that the interaction object is in
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;地标： （如果已提及则可选）交互对象所在的地标物体类别


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;relation to (e.g., "CounterTop" for "apple is on the countertop")
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;与……的关系（例如：“apple is on the countertop”中的“CounterTop”）


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;attributes: (optional) list of strings of desired attributes for the object. These are not
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;属性： （可选）对象所需属性字符串列表。这些并不


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;necessarily attributes that currently exist, but ones that the object should eventually
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;一定是当前存在的属性，而是对象最终应当


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;have. Attributes can only be from the following: "toasted", "clean", "cooked"
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;具备的属性。属性只能来自以下：“toasted”“clean”“cooked”


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;...



&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;self.object_class = object_class
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;self.object_class = object_class


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;self.landmark = landmark
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;self.landmark = landmark


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;self.attributes = attributes
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;self.attributes = attributes


&nbsp;&nbsp;&nbsp;&nbsp;def pickup(self):
&nbsp;&nbsp;&nbsp;&nbsp;def pickup(self):


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"""pickup the object.
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"""拾取该物体。


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;This function assumes the object is in view.
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;该函数假设物体在视野中。


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Example:
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;示例：


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;dialogue: <Commander> Go get the lettuce on the kitchen counter.
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;dialogue: <Commander> 去把厨房台面上的生菜拿过来。


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Python script:
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Python脚本：


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;target_lettuce = InteractionObject("Lettuce", landmark = "CounterTop")
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;target_lettuce = InteractionObject("Lettuce", landmark = "CounterTop")


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;target_lettuce.go_to()
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;target_lettuce.go_to()


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;target_lettuce.pickup()
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;target_lettuce.pickup()


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;nnn
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;nnn


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;pass
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;pass


&nbsp;&nbsp;&nbsp;&nbsp;def place(self, landmark_name):
&nbsp;&nbsp;&nbsp;&nbsp;def place(self, landmark_name):


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"""put the interaction object on the landmark_name object.
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"""把交互物体放到 landmark_name 这个物体上。


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;landmark_name must be a class InteractionObject instance
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;landmark_name 必须是一个 InteractionObject 类的实例


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;This function assumes the robot has picked up an object and the landmark object is in view.
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;此函数假设机器人已拿起某个物体，并且地标物体在视野中。


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Example:
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;示例：


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;dialogue: <Commander> Put the lettuce on the kitchen counter.
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;dialogue: <Commander> 把生菜放到厨房台面上。


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Python script:
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Python 脚本：


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;target_lettuce = InteractionObject("Lettuce", landmark = "CounterTop")
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;target_lettuce = InteractionObject("Lettuce", landmark = "CounterTop")


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;target_lettuce.go_to()
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;target_lettuce.go_to()


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;target_lettuce.pickup()
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;target_lettuce.pickup()


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;target_countertop = InteractionObject("CounterTop")
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;target_countertop = InteractionObject("CounterTop")


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;target_countertop.go_to()
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;target_countertop.go_to()


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;target_lettuce.place(target_countertop)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;target_lettuce.place(target_countertop)


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;nnn
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;nnn


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;pass
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;pass


&nbsp;&nbsp;&nbsp;&nbsp;def slice(self):
&nbsp;&nbsp;&nbsp;&nbsp;def slice(self):


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"""slice the object into pieces.
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"""把对象切成几块。


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;This function assumes the agent is holding a knife and the agent has navigated to the object
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;该函数假设智能体正在拿着一把刀，并且智能体已导航到该对象


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;using go_to().
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;通过 go_to()。


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Example:
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;示例：


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;dialogue: <Commander> Cut the apple on the kitchen counter.
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;dialogue: <Commander> 把厨房台面上的苹果切开。


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Python script:
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Python脚本：


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;target_knife = InteractionObject("Knife") # first we need a knife to slice the apple with
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;target_knife = InteractionObject("Knife") # 首先我们需要一把刀来切苹果


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;target_knife.go_to()
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;target_knife.go_to()


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;target_knife.pickup()
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;target_knife.pickup()


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;target_apple = InteractionObject("Apple", landmark = "CounterTop")
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;target_apple = InteractionObject("Apple", landmark = "CounterTop")


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;target_apple.go_to()
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;target_apple.go_to()


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;target_apple.slice()
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;target_apple.slice()


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;nnn
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;nnn


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;pass
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;pass


---



---



def toggle_on(self):
def toggle_on(self):


&nbsp;&nbsp;&nbsp;&nbsp;"""toggles on the interaction object.
&nbsp;&nbsp;&nbsp;&nbsp;"""将交互对象打开。


&nbsp;&nbsp;&nbsp;&nbsp;This function assumes the interaction object is already off and the agent has navigated to
&nbsp;&nbsp;&nbsp;&nbsp;此函数假设交互对象已处于关闭状态，且智能体已导航到


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;the object.
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;该对象处。


&nbsp;&nbsp;&nbsp;&nbsp;Only some landmark objects can be toggled on. Lamps, stoves, and microwaves are some
&nbsp;&nbsp;&nbsp;&nbsp;只有部分地标对象可以被打开。灯、炉灶和微波炉是


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;examples of objects that can be toggled on.
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;一些可以被打开的对象示例。


&nbsp;&nbsp;&nbsp;&nbsp;Example:
&nbsp;&nbsp;&nbsp;&nbsp;示例：


&nbsp;&nbsp;&nbsp;&nbsp;dialogue: <Commander> Turn on the lamp.
&nbsp;&nbsp;&nbsp;&nbsp;dialogue: <Commander> 打开灯。


&nbsp;&nbsp;&nbsp;&nbsp;Python script:
&nbsp;&nbsp;&nbsp;&nbsp;Python script:


&nbsp;&nbsp;&nbsp;&nbsp;target_floorlamp = InteractionObject("FloorLamp")
&nbsp;&nbsp;&nbsp;&nbsp;target_floorlamp = InteractionObject("FloorLamp")


&nbsp;&nbsp;&nbsp;&nbsp;target_floorlamp.go_to()
&nbsp;&nbsp;&nbsp;&nbsp;target_floorlamp.go_to()


&nbsp;&nbsp;&nbsp;&nbsp;target_floorlamp.toggle_on()
&nbsp;&nbsp;&nbsp;&nbsp;target_floorlamp.toggle_on()


&nbsp;&nbsp;&nbsp;&nbsp;nnn
&nbsp;&nbsp;&nbsp;&nbsp;nnn


&nbsp;&nbsp;&nbsp;&nbsp;pass
&nbsp;&nbsp;&nbsp;&nbsp;pass


def toggle_off(self):
def toggle_off(self):


&nbsp;&nbsp;&nbsp;&nbsp;"""toggles off the interaction object.
&nbsp;&nbsp;&nbsp;&nbsp;"""关闭交互对象。


&nbsp;&nbsp;&nbsp;&nbsp;This function assumes the interaction object is already on and the agent has navigated to
&nbsp;&nbsp;&nbsp;&nbsp;此函数假设交互对象已处于开启状态，且智能体已导航到


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;the object.
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;该对象。


&nbsp;&nbsp;&nbsp;&nbsp;Only some objects can be toggled off. Lamps, stoves, and microwaves are some examples of
&nbsp;&nbsp;&nbsp;&nbsp;只有部分对象可以关闭。灯具、炉灶和微波炉是一些


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;objects that can be toggled off.
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;可被关闭的对象。


&nbsp;&nbsp;&nbsp;&nbsp;Example:
&nbsp;&nbsp;&nbsp;&nbsp;示例：


&nbsp;&nbsp;&nbsp;&nbsp;dialogue: <Commander> Turn off the lamp.
&nbsp;&nbsp;&nbsp;&nbsp;对话：<Commander> 关掉那盏灯。


&nbsp;&nbsp;&nbsp;&nbsp;Python script:
&nbsp;&nbsp;&nbsp;&nbsp;Python 脚本：


&nbsp;&nbsp;&nbsp;&nbsp;target_floorlamp = InteractionObject("FloorLamp")
&nbsp;&nbsp;&nbsp;&nbsp;target_floorlamp = InteractionObject("FloorLamp")


&nbsp;&nbsp;&nbsp;&nbsp;target_floorlamp.go_to()
&nbsp;&nbsp;&nbsp;&nbsp;target_floorlamp.go_to()


&nbsp;&nbsp;&nbsp;&nbsp;target_floorlamp.toggle_off()
&nbsp;&nbsp;&nbsp;&nbsp;target_floorlamp.toggle_off()


&nbsp;&nbsp;&nbsp;&nbsp;"""



&nbsp;&nbsp;&nbsp;&nbsp;pass
&nbsp;&nbsp;&nbsp;&nbsp;pass


def go_to(self):
def go_to(self):


&nbsp;&nbsp;&nbsp;&nbsp;"""Navigate to the object
&nbsp;&nbsp;&nbsp;&nbsp;"""导航到该对象


&nbsp;&nbsp;&nbsp;&nbsp;nnn
&nbsp;&nbsp;&nbsp;&nbsp;nnn


&nbsp;&nbsp;&nbsp;&nbsp;pass
&nbsp;&nbsp;&nbsp;&nbsp;pass


def open(self):
def open(self):


&nbsp;&nbsp;&nbsp;&nbsp;"""open the interaction object.
&nbsp;&nbsp;&nbsp;&nbsp;"""打开交互对象。


&nbsp;&nbsp;&nbsp;&nbsp;This function assumes the landmark object is already closed and the agent has already
&nbsp;&nbsp;&nbsp;&nbsp;该函数假设地标对象已处于关闭状态，且智能体已


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;navigated to the object.
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;导航到该对象。


&nbsp;&nbsp;&nbsp;&nbsp;Only some objects can be opened. Fridges, cabinets, and drawers are some example of objects
&nbsp;&nbsp;&nbsp;&nbsp;只有部分对象可以打开。冰箱、橱柜和抽屉是一些可以


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;that can be closed.
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;关闭的对象。


&nbsp;&nbsp;&nbsp;&nbsp;Example:
&nbsp;&nbsp;&nbsp;&nbsp;示例：


&nbsp;&nbsp;&nbsp;&nbsp;dialogue: <Commander> Get the lettuce in the fridge.
&nbsp;&nbsp;&nbsp;&nbsp;dialogue: <Commander> 把冰箱里的生菜拿出来。


&nbsp;&nbsp;&nbsp;&nbsp;Python script:
&nbsp;&nbsp;&nbsp;&nbsp;Python 脚本：


&nbsp;&nbsp;&nbsp;&nbsp;target_fridge = InteractionObject("Fridge")
&nbsp;&nbsp;&nbsp;&nbsp;target_fridge = InteractionObject("Fridge")


&nbsp;&nbsp;&nbsp;&nbsp;target_lettuce = InteractionObject("Lettuce", landmark = "Fridge")
&nbsp;&nbsp;&nbsp;&nbsp;target_lettuce = InteractionObject("Lettuce", landmark = "Fridge")


&nbsp;&nbsp;&nbsp;&nbsp;target_fridge.go_to()
&nbsp;&nbsp;&nbsp;&nbsp;target_fridge.go_to()


&nbsp;&nbsp;&nbsp;&nbsp;target_fridge.open()
&nbsp;&nbsp;&nbsp;&nbsp;target_fridge.open()


&nbsp;&nbsp;&nbsp;&nbsp;target_lettuce.pickup()
&nbsp;&nbsp;&nbsp;&nbsp;target_lettuce.pickup()


&nbsp;&nbsp;&nbsp;&nbsp;"""



&nbsp;&nbsp;&nbsp;&nbsp;pass
&nbsp;&nbsp;&nbsp;&nbsp;pass


def close(self):
def close(self):


&nbsp;&nbsp;&nbsp;&nbsp;"""close the interaction object.
&nbsp;&nbsp;&nbsp;&nbsp;"""关闭交互对象。


&nbsp;&nbsp;&nbsp;&nbsp;This function assumes the object is already open and the agent has already navigated to the
&nbsp;&nbsp;&nbsp;&nbsp;该函数假设对象已经处于打开状态，且智能体已经导航到


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;object.
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;该对象。


&nbsp;&nbsp;&nbsp;&nbsp;Only some objects can be closed. Fridges, cabinets, and drawers are some example of objects
&nbsp;&nbsp;&nbsp;&nbsp;只有部分对象可以关闭。冰箱、橱柜和抽屉都是一些可以


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;that can be closed.
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;关闭的对象。


&nbsp;&nbsp;&nbsp;&nbsp;nnn
&nbsp;&nbsp;&nbsp;&nbsp;nnn


&nbsp;&nbsp;&nbsp;&nbsp;pass
&nbsp;&nbsp;&nbsp;&nbsp;pass


def clean(self):
def clean(self):


---



---



&nbsp;&nbsp;&nbsp;&nbsp;"""wash the interaction object to clean it in the sink.
&nbsp;&nbsp;&nbsp;&nbsp;"""将交互对象在水槽中清洗，以便清洁它。


&nbsp;&nbsp;&nbsp;&nbsp;This function assumes the object is already picked up.
&nbsp;&nbsp;&nbsp;&nbsp;该函数假设对象已经被拿起。


&nbsp;&nbsp;&nbsp;&nbsp;Example:
&nbsp;&nbsp;&nbsp;&nbsp;示例：


&nbsp;&nbsp;&nbsp;&nbsp;dialogue: <Commander> Clean the bowl
&nbsp;&nbsp;&nbsp;&nbsp;对话：<Commander> 清洗碗


&nbsp;&nbsp;&nbsp;&nbsp;Python script:
&nbsp;&nbsp;&nbsp;&nbsp;Python脚本：


&nbsp;&nbsp;&nbsp;&nbsp;target_bowl = InteractionObject("Bowl", attributes = ["clean"])
&nbsp;&nbsp;&nbsp;&nbsp;target_bowl = InteractionObject("Bowl", attributes = ["clean"])


&nbsp;&nbsp;&nbsp;&nbsp;target_bowl.clean()
&nbsp;&nbsp;&nbsp;&nbsp;target_bowl.clean()


&nbsp;&nbsp;&nbsp;&nbsp;nnn
&nbsp;&nbsp;&nbsp;&nbsp;nnn


&nbsp;&nbsp;&nbsp;&nbsp;pass
&nbsp;&nbsp;&nbsp;&nbsp;放下


def put_down(self):
def put_down(self):


&nbsp;&nbsp;&nbsp;&nbsp;"""puts the interaction object currently in the agent's hand on the nearest available
&nbsp;&nbsp;&nbsp;&nbsp;"""将智能体手中的交互对象放到最近可用的


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;receptacle
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;容器中


&nbsp;&nbsp;&nbsp;&nbsp;This function assumes the object is already picked up.
&nbsp;&nbsp;&nbsp;&nbsp;此函数假定对象已被拾起。


&nbsp;&nbsp;&nbsp;&nbsp;This function is most often used when the holding object is no longer needed, and the agent
&nbsp;&nbsp;&nbsp;&nbsp;当手持对象已不再需要，而智能体


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;needs to pick up another object
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;需要拾起另一个对象时，最常使用此函数


&nbsp;&nbsp;&nbsp;&nbsp;nnn
&nbsp;&nbsp;&nbsp;&nbsp;nnn


&nbsp;&nbsp;&nbsp;&nbsp;pass
&nbsp;&nbsp;&nbsp;&nbsp;pass


def pour(self, landmark_name):
def pour(self, landmark_name):


&nbsp;&nbsp;&nbsp;&nbsp;"""pours the contents of the interaction object into the landmark object specified by the
&nbsp;&nbsp;&nbsp;&nbsp;"""将交互对象中的内容倒入由


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;landmark_name argument
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;landmark_name 参数指定的地标对象中


&nbsp;&nbsp;&nbsp;&nbsp;landmark_name must be a class InteractionObject instance
&nbsp;&nbsp;&nbsp;&nbsp;landmark_name 必须是类 InteractionObject 的实例


&nbsp;&nbsp;&nbsp;&nbsp;This function assumes the object is already picked up and the object is filled with liquid.
&nbsp;&nbsp;&nbsp;&nbsp;此函数假定对象已被拾起，且对象中已装有液体。


&nbsp;&nbsp;&nbsp;&nbsp;nnn
&nbsp;&nbsp;&nbsp;&nbsp;nnn


&nbsp;&nbsp;&nbsp;&nbsp;pass
&nbsp;&nbsp;&nbsp;&nbsp;pass


def fill_up(self):
def fill_up(self):


&nbsp;&nbsp;&nbsp;&nbsp;"""fill up the interaction object with water
&nbsp;&nbsp;&nbsp;&nbsp;"""用水把交互对象装满


&nbsp;&nbsp;&nbsp;&nbsp;This function assumes the object is already picked up. Note that only container objects can
&nbsp;&nbsp;&nbsp;&nbsp;此函数假设该对象已被拿起。注意，只有容器类对象才能


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;be filled with liquid.
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;装入液体。


&nbsp;&nbsp;&nbsp;&nbsp;pass
&nbsp;&nbsp;&nbsp;&nbsp;pass


def pickup_and_place(self, landmark_name):
def pickup_and_place(self, landmark_name):


&nbsp;&nbsp;&nbsp;&nbsp;"""go_to() and pickup() this interaction object, then go_to() and place() the interaction
&nbsp;&nbsp;&nbsp;&nbsp;"""先对该交互对象执行 go_to() 和 pickup()，再对该交互对象执行 go_to() 和 place() 将其放置


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;object on the landmark_name object.
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;到 landmark_name 对象上。


&nbsp;&nbsp;&nbsp;&nbsp;landmark_name must be a class InteractionObject instance
&nbsp;&nbsp;&nbsp;&nbsp;landmark_name 必须是一个类 InteractionObject 的实例


&nbsp;&nbsp;&nbsp;&nbsp;nnn
&nbsp;&nbsp;&nbsp;&nbsp;nnn


&nbsp;&nbsp;&nbsp;&nbsp;pass
&nbsp;&nbsp;&nbsp;&nbsp;pass


def empty(self):
def empty(self):


&nbsp;&nbsp;&nbsp;&nbsp;"""Empty the object of any other objects on/in it to clear it out.
&nbsp;&nbsp;&nbsp;&nbsp;"""把交互对象内/上可能存在的其他对象清空，以便把它腾出来。


&nbsp;&nbsp;&nbsp;&nbsp;Useful when the object is too full to place an object inside it.
&nbsp;&nbsp;&nbsp;&nbsp;当该对象装得太满而无法在其中放入其他对象时很有用。


&nbsp;&nbsp;&nbsp;&nbsp;Example:
&nbsp;&nbsp;&nbsp;&nbsp;示例：


&nbsp;&nbsp;&nbsp;&nbsp;dialogue: <Commander> Clear out the sink.
&nbsp;&nbsp;&nbsp;&nbsp;dialogue: <Commander> 把水槽清空。


&nbsp;&nbsp;&nbsp;&nbsp;Python script:
&nbsp;&nbsp;&nbsp;&nbsp;Python 脚本：


&nbsp;&nbsp;&nbsp;&nbsp;target_sink = InteractionObject("Sink")
&nbsp;&nbsp;&nbsp;&nbsp;target_sink = InteractionObject("水槽")


&nbsp;&nbsp;&nbsp;&nbsp;target_sink.empty()
&nbsp;&nbsp;&nbsp;&nbsp;target_sink.empty()


&nbsp;&nbsp;&nbsp;&nbsp;pass
&nbsp;&nbsp;&nbsp;&nbsp;pass


def cook(self):
def cook(self):


&nbsp;&nbsp;&nbsp;&nbsp;"""Cook the object
&nbsp;&nbsp;&nbsp;&nbsp;"""烹饪该物体


&nbsp;&nbsp;&nbsp;&nbsp;Example:
&nbsp;&nbsp;&nbsp;&nbsp;示例：


&nbsp;&nbsp;&nbsp;&nbsp;dialogue: <Commander> Cook the potato.
&nbsp;&nbsp;&nbsp;&nbsp;对话：<Commander> 把土豆煮了。


&nbsp;&nbsp;&nbsp;&nbsp;Python script:
&nbsp;&nbsp;&nbsp;&nbsp;Python 脚本：


&nbsp;&nbsp;&nbsp;&nbsp;target_potato = InteractionObject("Potato", attributes = ["cooked"])
&nbsp;&nbsp;&nbsp;&nbsp;target_potato = InteractionObject("土豆", attributes = ["已煮熟"])


&nbsp;&nbsp;&nbsp;&nbsp;target_potato.cook()
&nbsp;&nbsp;&nbsp;&nbsp;target_potato.cook()


&nbsp;&nbsp;&nbsp;&nbsp;nnn
&nbsp;&nbsp;&nbsp;&nbsp;nnn


---



---



def toast(self):
def toast(self):


&nbsp;&nbsp;&nbsp;&nbsp;"""Toast a bread slice in a toaster
&nbsp;&nbsp;&nbsp;&nbsp;"""在烤面包机里烤面包片


&nbsp;&nbsp;&nbsp;&nbsp;Toasting is only supported with slices of bread
&nbsp;&nbsp;&nbsp;&nbsp;只有面包片才能进行烘烤


&nbsp;&nbsp;&nbsp;&nbsp;Example:
&nbsp;&nbsp;&nbsp;&nbsp;示例：


&nbsp;&nbsp;&nbsp;&nbsp;dialogue: <Commander> Get me a toasted bread slice.
&nbsp;&nbsp;&nbsp;&nbsp;对话：<Commander> 给我一片烤面包。


&nbsp;&nbsp;&nbsp;&nbsp;Python script:
&nbsp;&nbsp;&nbsp;&nbsp;Python 脚本：


&nbsp;&nbsp;&nbsp;&nbsp;target_breadslice = InteractionObject("BreadSliced", attributes = ["toasted"])
&nbsp;&nbsp;&nbsp;&nbsp;target_breadslice = InteractionObject("BreadSliced", attributes = ["toasted"])


&nbsp;&nbsp;&nbsp;&nbsp;target_breadslice.toast()
&nbsp;&nbsp;&nbsp;&nbsp;target_breadslice.toast()


&nbsp;&nbsp;&nbsp;&nbsp;nm



&nbsp;&nbsp;&nbsp;&nbsp;pass
&nbsp;&nbsp;&nbsp;&nbsp;pass


---



Listing 3: Full Corrective API for the parametrized corrective macro-actions ${G}_{\text{ corrective }}$ used in the prompts.
表 3：用于提示中参数化纠错宏动作 ${G}_{\text{ corrective }}$ 的完整纠错 API。


---



class AgentCorrective:
类 AgentCorrective：


&nbsp;&nbsp;&nbsp;&nbsp;...



&nbsp;&nbsp;&nbsp;&nbsp;This class represents agent corrective actions that can be taken to fix a subgoal error
&nbsp;&nbsp;&nbsp;&nbsp;该类表示可用于修复子目标错误的智能体纠错动作


&nbsp;&nbsp;&nbsp;&nbsp;Example usage:
&nbsp;&nbsp;&nbsp;&nbsp;用法示例：


&nbsp;&nbsp;&nbsp;&nbsp;agent = AgentCorrective()
&nbsp;&nbsp;&nbsp;&nbsp;agent = AgentCorrective()


&nbsp;&nbsp;&nbsp;&nbsp;agent.move_back()
&nbsp;&nbsp;&nbsp;&nbsp;agent.move_back()


&nbsp;&nbsp;&nbsp;&nbsp;111



&nbsp;&nbsp;&nbsp;&nbsp;def move_back(self):
&nbsp;&nbsp;&nbsp;&nbsp;def move_back(self):


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"""Step backwards away from the object
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"""后退一步，离开该物体


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Useful when the object is too close for the agent to interact with it
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;当物体离得太近，智能体无法与其交互时很有用


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;nnn
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;nnn


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;pass
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;pass


&nbsp;&nbsp;&nbsp;&nbsp;def move_closer(self):
&nbsp;&nbsp;&nbsp;&nbsp;def move_closer(self):


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"""Step forward to towards the object to get closer to it
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"""向前迈一步，朝向目标物体以便靠近它"""


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Useful when the object is too far for the agent to interact with it
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;当物体距离过远、代理无法与其互动时很有用


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"""



&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;pass
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;pass


&nbsp;&nbsp;&nbsp;&nbsp;def move_alternate_viewpoint(self):
&nbsp;&nbsp;&nbsp;&nbsp;def move_alternate_viewpoint(self):


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"""Move to an alternate viewpoint to look at the object
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"""移动到另一个视角去观察物体"""


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Useful when the object is occluded or an interaction is failing due to collision or
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;当物体被遮挡，或因碰撞导致交互失败时很有用，或


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;occlusion.
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;遮挡。


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"""



&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;pass
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;pass


---



Listing 4: Full Prompt for the PLANNER. \{\} indicates areas that are replaced in the prompt.
列表4：PLANNER的完整提示词。 \{\} 表示提示词中将被替换的区域。


You are an adept at translating human dialogues into sequences of actions for household robots. Given a dialogue between a <Driver> and a <Commander>, you convert the conversation into a Python program to be executed by a robot.
你擅长将人类对话翻译成家庭机器人可执行的动作序列。给定 <Driver> 与 <Commander> 之间的一段对话，你将把对话转换为可由机器人执行的 Python 程序。


\{API\}
\{API\}


Write a script using Python and the InteractionObject class and functions defined above that could be executed by a household robot.
使用 Python 和上面定义的 InteractionObject 类及其函数编写一个脚本，使其能够由家庭机器人执行。


## \{RETRIEVED_EXAMPLES\}
## \{RETRIEVED_EXAMPLES\}


Adhere to these stringent guidelines:
请遵循这些严格准则：


1. Use only the classes and functions defined previously. Do not create functions that are not provided above.
1. 仅使用先前定义的类和函数。不要创建上面未提供的函数。


2. Make sure that you output a consistent plan. For example, opening of the same object should not occur in successive steps.
2. 确保输出的计划一致。例如，连续步骤中不应对同一个物体进行相同的开场操作。


3. Make sure the output is consistent with the proper affordances of objects. For example, a couch cannot be opened, so your output should never include the open() function for this object, but a fridge can be opened.
3. 确保输出与物体的正确功能特性（affordances）一致。例如，沙发不能被打开，所以你的输出绝不能包含该物体的 open() 函数；但冰箱可以被打开。


4. The input is dialogue between <Driver> and <Commander>. Interpret the dialogue into robot actions. Do not output any dialogue.
4. 输入是 <Driver> 与 <Commander> 之间的对话。将对话解读为机器人动作。不要输出任何对话。


5. Object categories should only be chosen from the following classes: ShowerDoor, Cabinet, CounterTop, Sink, Towel, HandTowel, TowelHolder, SoapBar, ToiletPaper, ToiletPaperHanger, HandTowelHolder, SoapBottle, GarbageCan, Candle, ScrubBrush, Plunger, SinkBasin, Cloth, SprayBottle, Toilet, Faucet, ShowerHead, Box, Bed, Book, DeskLamp, BasketBall, Pen, Pillow, Pencil, CellPhone, KeyChain, Painting, CreditCard, AlarmClock, CD, Laptop, Drawer, SideTable, Chair, Blinds, Desk, Curtains, Dresser, Watch, Television, WateringCan, Newspaper, FloorLamp, RemoteControl, HousePlant, Statue, Ottoman, ArmChair, Sofa, DogBed, BaseballBat, TennisRacket, VacuumCleaner, Mug, ShelvingUnit, Shelf, StoveBurner, Apple, Lettuce, Bottle, Egg, Microwave, CoffeeMachine, Fork, Fridge, WineBottle, Spatula, Bread, Tomato, Pan, Cup, Pot, SaltShaker, Potato, PepperShaker, ButterKnife, StoveKnob, Toaster, DishSponge, Spoon, Plate, Knife, DiningTable, Bowl, LaundryHamper, Vase, Stool, CoffeeTable, Poster, Bathtub, TissueBox, Footstool, BathtubBasin, ShowerCurtain, TVStand, Boots, RoomDecor, PaperTowelRoll, Ladle, Kettle, Safe, GarbageBag, TeddyBear, TableTopDecor, Dumbbell, Desktop, AluminumFoil, Window, LightSwitch, AppleSliced, BreadSliced, LettuceSliced, PotatoSliced, TomatoSliced
5. 物体类别只能从以下类中选择：ShowerDoor, Cabinet, CounterTop, Sink, Towel, HandTowel, TowelHolder, SoapBar, ToiletPaper, ToiletPaperHanger, HandTowelHolder, SoapBottle, GarbageCan, Candle, ScrubBrush, Plunger, SinkBasin, Cloth, SprayBottle, Toilet, Faucet, ShowerHead, Box, Bed, Book, DeskLamp, BasketBall, Pen, Pillow, Pencil, CellPhone, KeyChain, Painting, CreditCard, AlarmClock, CD, Laptop, Drawer, SideTable, Chair, Blinds, Desk, Curtains, Dresser, Watch, Television, WateringCan, Newspaper, FloorLamp, RemoteControl, HousePlant, Statue, Ottoman, ArmChair, Sofa, DogBed, BaseballBat, TennisRacket, VacuumCleaner, Mug, ShelvingUnit, Shelf, StoveBurner, Apple, Lettuce, Bottle, Egg, Microwave, CoffeeMachine, Fork, Fridge, WineBottle, Spatula, Bread, Tomato, Pan, Cup, Pot, SaltShaker, Potato, PepperShaker, ButterKnife, StoveKnob, Toaster, DishSponge, Spoon, Plate, Knife, DiningTable, Bowl, LaundryHamper, Vase, Stool, CoffeeTable, Poster, Bathtub, TissueBox, Footstool, BathtubBasin, ShowerCurtain, TVStand, Boots, RoomDecor, PaperTowelRoll, Ladle, Kettle, Safe, GarbageBag, TeddyBear, TableTopDecor, Dumbbell, Desktop, AluminumFoil, Window, LightSwitch, AppleSliced, BreadSliced, LettuceSliced, PotatoSliced, TomatoSliced


6. You can only pick up one object at a time. If the agent is holding an object, the agent should place or put down the object before attempting to pick up a second object.
6. 你一次只能拿起一个物体。若代理正在拿着某个物体，在尝试拿起第二个物体之前，必须先放下或放置该物体。


7. Each object instance should instantiate a different InteractionObject class even if two object instances are the same object category.
7. 即使两个物体实例属于同一物体类别，每个物体实例也应实例化一个不同的 InteractionObject 类。


Follow the output format provided earlier. Think step by step to carry out the instruction.
请遵循先前提供的输出格式。一步一步思考以完成该指令。


Write a Python script that could be executed by a household robot for the following: dialogue: \{command\}
编写一段 Python 脚本，可由家用机器人执行以下任务：对话：\{command\}


Python script:
Python 脚本：


Listing 5: Full Prompt for the RECTIFIER. \{\} indicates areas that are replaced in the prompt.
清单 5：RECTIFIER 的完整提示。 \{\} 表示提示中将被替换的区域。


You are an excellent interpreter of human instructions for household tasks. Given a failed action subgoal by a household robot, dialogue instructions between robot <Driver> and user <Commander>, and information about the environment and failure, you provide a sequence of robotic subgoal actions to overcome the failure.
你是非常出色的家务任务人类指令解读者。给定家用机器人失败的动作子目标、机器人 <Driver> 与用户 <Commander> 之间的对话，以及关于环境和失败的信息，你将提供一系列机器人子目标动作来克服该失败。


\{API\}
\{API\}


## \{API_CORRECTIVE\}
## \{API_CORRECTIVE\}


Information about the failure and environment are given as follows:
失败和环境的信息如下：


Failed subgoal: The robotic subgoal for which the failure occured.
失败的子目标：发生故障的机器人子目标。


Execution error: feedback as to why the failed subgoal occurred.
执行错误：说明失败子目标为何会发生的反馈。


Input dialogue: full dialogue instructions between robot <Driver> and user <Commander> for the complete task. This may or may not be useful.
输入对话：完整任务中机器人 <Driver> 与用户 <Commander> 之间的完整对话指令。这可能有用，也可能无用。


I will give you examples of the input and output you will generate.
我将给你示例，展示你将要生成的输入与输出。


\{retrieved_plans\}
\{retrieved_plans\}


Fix the subgoal exectuion error using only the InteractionObject class and functions defined above that could be executed by a household robot. Follow these rules very strictly:
仅使用上文定义的 InteractionObject 类及其函数，修复子目标执行错误；这些内容必须能被家用机器人执行。请严格遵守以下规则：


1. Important! Use only the classes and functions defined previously. Do not create functions or additional code that are not provided in the above API. Do not include if-else statements.
1. 重要！只使用前面已定义的类和函数。不要创建在上述 API 中未提供的函数或额外代码。不要包含 if-else 语句。


2. Important! Make sure that you output a consistent plan. For example, opening of the same object should not occur in successive steps.
2. 重要！务必输出一个一致的计划。例如，对同一对象的打开操作不应连续出现在相邻步骤中。


3. Important! Make sure the output is consistent with the proper affordances of objects. For example, a couch cannot be opened, so your output should never include the open() function for this object, but a fridge can be opened.
3. 重要！确保输出与物体的恰当可操作性一致。例如，沙发不能被打开，所以你的输出绝不应包含针对该物体的 open() 函数；但冰箱可以被打开。


4. Important! The dialogue is between <Driver> and <Commander>. The dialogue may or may not be helpful. Do not output any dialogue.
4. 重要！对话是在 <Driver> 和 <Commander> 之间。对话可能有用，也可能没用。不要输出任何对话。


5. Important! Object classes should only be chosen from the following classes: ShowerDoor, Cabinet, CounterTop, Sink, Towel, HandTowel, TowelHolder, SoapBar, ToiletPaper, ToiletPaperHanger, HandTowelHolder, SoapBottle, GarbageCan, Candle, ScrubBrush, Plunger, SinkBasin, Cloth, SprayBottle, Toilet, Faucet, ShowerHead, Box, Bed, Book, DeskLamp, BasketBall, Pen, Pillow, Pencil, CellPhone, KeyChain, Painting, CreditCard, AlarmClock, CD, Laptop, Drawer, SideTable, Chair, Blinds, Desk, Curtains, Dresser, Watch, Television, WateringCan, Newspaper, FloorLamp, RemoteControl, HousePlant, Statue, Ottoman, ArmChair, Sofa, DogBed, BaseballBat, TennisRacket, VacuumCleaner, Mug, ShelvingUnit, Shelf, StoveBurner, Apple, Lettuce, Bottle, Egg, Microwave, CoffeeMachine, Fork, Fridge, WineBottle, Spatula, Bread, Tomato, Pan, Cup, Pot, SaltShaker, Potato, PepperShaker, ButterKnife, StoveKnob, Toaster, DishSponge, Spoon, Plate, Knife, DiningTable, Bowl, LaundryHamper, Vase, Stool, CoffeeTable, Poster, Bathtub, TissueBox, Footstool, BathtubBasin, ShowerCurtain, TVStand, Boots, RoomDecor, PaperTowelRoll, Ladle, Kettle, Safe, GarbageBag, TeddyBear, TableTopDecor, Dumbbell, Desktop, AluminumFoil, Window, LightSwitch, AppleSliced, BreadSliced, LettuceSliced, PotatoSliced, TomatoSliced
5. 重要！对象类别只能从以下类别中选择：ShowerDoor、Cabinet、CounterTop、Sink、Towel、HandTowel、TowelHolder、SoapBar、ToiletPaper、ToiletPaperHanger、HandTowelHolder、SoapBottle、GarbageCan、Candle、ScrubBrush、Plunger、SinkBasin、Cloth、SprayBottle、Toilet、Faucet、ShowerHead、Box、Bed、Book、DeskLamp、BasketBall、Pen、Pillow、Pencil、CellPhone、KeyChain、Painting、CreditCard、AlarmClock、CD、Laptop、Drawer、SideTable、Chair、Blinds、Desk、Curtains、Dresser、Watch、Television、WateringCan、Newspaper、FloorLamp、RemoteControl、HousePlant、Statue、Ottoman、ArmChair、Sofa、DogBed、BaseballBat、TennisRacket、VacuumCleaner、Mug、ShelvingUnit、Shelf、StoveBurner、Apple、Lettuce、Bottle、Egg、Microwave、CoffeeMachine、Fork、Fridge、WineBottle、Spatula、Bread、Tomato、Pan、Cup、Pot、SaltShaker、Potato、PepperShaker、ButterKnife、StoveKnob、Toaster、DishSponge、Spoon、Plate、Knife、DiningTable、Bowl、LaundryHamper、Vase、Stool、CoffeeTable、Poster、Bathtub、TissueBox、Footstool、BathtubBasin、ShowerCurtain、TVStand、Boots、RoomDecor、PaperTowelRoll、Ladle、Kettle、Safe、GarbageBag、TeddyBear、TableTopDecor、Dumbbell、Desktop、AluminumFoil、Window、LightSwitch、AppleSliced、BreadSliced、LettuceSliced、PotatoSliced、TomatoSliced


6. Important! You can only pick up one object at a time. If the agent is holding an object, the agent should place or put down the object before attempting to pick up a second object.
6. 重要！你一次只能拿起一个物体。如果代理正拿着一个物体，那么在尝试拿起第二个物体之前，代理应先放下或置下该物体。


7. Important! Each object instance should instantiate a different InteractionObject class even if two object instances are the same object category.
7. 重要！即使两个对象实例属于同一对象类别，每个对象实例也应实例化为不同的 InteractionObject 类。


8. Important! Your plan should ONLY fix the failed subgoal. Do not include plans for other parts of the dialogue or future plan that are irrelevant to the execution error and failed subgoal.
8. 重要！你的计划应只修复失败的子目标。不要包含与执行错误和失败子目标无关的对话其他部分或未来计划。


9. Important! output "do_nothing()" if the agent should not take any corrective actions.
9. 重要！如果代理不需要采取任何纠正行动，请输出“do_nothing()”。


Adhere to the output format I defined above. Think step by step to carry out the instruction.
遵循我在上文定义的输出格式。一步一步地思考以完成指令。


Make use of the following information to help you fix the failed subgoal:
使用以下信息来帮助你修复失败的子目标：


Failed subgoal: ...
失败的子目标：...


Execution error: ...
执行错误：...


Input dialogue: ...
输入对话：...


You should respond in the following format:
你应当以如下格式回复：


Explain: Are there any steps missing to complete the subgoal? Why did the failed subgoal occur? What does the execution error imply for how to fix your future plan?
解释：完成子目标是否缺少步骤？为什么会发生失败的子目标？执行错误意味着什么，会如何影响你未来计划的修复方式？


Plan (Python script): A Python script to only fix the execution error.
计划（Python 脚本）：仅用于修复执行错误的 Python 脚本。


Explain:
解释：


Listing 6: Full Prompt for the LOCATOR. \{\} indicates areas that are replaced in the prompt.
列表 6：LOCATOR 的完整提示。{{}} 表示提示中被替换的区域。


You are a household robot trying to locate objects within a house.
你是一名家用机器人，试图在房子里定位物体。


You will be given a target object category, your task is to output the top 3 most likely object categories that the target object category is likely to be found near: \{OBJECT_CLASSES\}
将给你一个目标物体类别；你的任务是输出最可能找到目标物体类别的 3 个最可能相邻物体类别：\{OBJECT_CLASSES\}


For your answer, take into account commonsense co-occurances of objects within a house and (if relevant) any hints given by the instruction dialogue between the robot <Driver> and user <Commander>.
在作答时，请考虑家中物体的常识性共现情况，并（如相关）考虑机器人 <Driver> 与用户 <Commander> 之间指令对话提供的任何线索。


For example, if given the target object category is "Knife" and the following dialogue: "<Commander> hi, make a slice of tomato. <Driver> where is the tomato? <Driver> where is the knife? <Commander> in the sink.", you might output the following top 3 most likely object categories: "answer: Sink, CounterTop, Cabinet". Important: Your output should ONLY a list (3 words seperated by commas) of three object categories from the list above.
例如，如果给定的目标物体类别是“Knife”，且如下对话：“<Commander> hi, make a slice of tomato. <Driver> where is the tomato? <Driver> where is the knife? <Commander> in the sink.”，你可能输出如下 3 个最可能相邻物体类别：“answer: Sink, CounterTop, Cabinet”。重要：你的输出只应是一个列表（三个词，用逗号分隔），从上面的列表中选出的三个物体类别。


What are the top 3 most likely object categories for where to find the target category \{INPUT_TARGET_OBJECT\} near? Here is the dialogue that may or may not contain relevant information for your answer: \{INPUT_DIALOGUE\}
在 \{INPUT_TARGET_OBJECT\} 这一目标类别附近，最可能找到它的位置的 3 个物体类别是什么？下面是可能包含也可能不包含相关信息的对话：\{INPUT_DIALOGUE\}


answer:
answer:


Listing 7: Example of a precondition check for the slice macro-action
清单7：切片宏动作的前置条件检查示例


---



if subgoal=="Slice":
if subgoal=="Slice":


&nbsp;&nbsp;&nbsp;&nbsp;#contraint check for "slice" macro-action
&nbsp;&nbsp;&nbsp;&nbsp;##“slice”宏动作的约束检查


&nbsp;&nbsp;&nbsp;&nbsp;if object_tracker.get_label_of_holding()!="Knife":
&nbsp;&nbsp;&nbsp;&nbsp;if object_tracker.get_label_of_holding()!="Knife":


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;#first need to be holding a knife to slice
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;##要进行切片，首先需要拿着一把刀


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;program_to_add = [
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;program_to_add = [


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;'target_knife = InteractionObject("Knife")',
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;'target_knife = InteractionObject("Knife")',


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;'target_knife.go_to()',
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;'target_knife.go_to()',


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;'target_knife.pickup()',
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;'target_knife.pickup()',


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;]



---



Listing 8: Example of dialogue input, retrieved examples, and LLM output for a TEACh sample
清单8：TEACh样例的对话输入、检索示例以及LLM输出示例


Dialogue input:
对话输入：


<Driver> how can I help you today?<Commander> can you please make me a salad on a clean plate with tomato and cooked potato?<Driver> does the salad require chopped lettuce?<Commander> nope!<Driver> is that all?<Commander> can you place them on a plate?<Driver> are they not already on a plate?
<Driver> 今天我能怎么帮您？<Commander> 请你用番茄和熟土豆，放在一个干净的盘子里给我做沙拉，好吗？<Driver> 沙拉需要切碎的生菜吗？<Commander> 不需要！<Driver> 就这些吗？<Commander> 你能把它们放到盘子里吗？<Driver> 它们不是已经在盘子上了吗？


## Retrieved Examples:
## 已检索示例：


Example 1:
示例 1：


dialogue: <Driver> What should I do today? <Commander> hi, make a slice of tomato. <Driver> where is the tomato? <Driver> where is the knife? <Commander> in the sink. <Driver> Tomato sliced. What next? <Commander> slice the potato. <Driver> Where is the potato? <Commander> in the microwave. <Commander> place all salad components on a plate. <Driver> How many slices of potato? <Commander> all salad components need to be place on a plate. <Driver> Where is the plate? <Commander> plate. <Commander> try the tomato piece. <Driver> done. <Commander> we are finished.
对话：<Driver> 今天我该做什么？ <Commander> 你好，切一片番茄。 <Driver> 番茄在哪？ <Driver> 刀子在哪？ <Commander> 在水槽里。 <Driver> 番茄切好了。接下来呢？ <Commander> 切土豆。 <Driver> 土豆在哪？ <Commander> 在微波炉里。 <Commander> 把所有沙拉配料放到盘子上。 <Driver> 土豆要切多少片？ <Commander> 所有沙拉配料都需要放到盘子上。 <Driver> 盘子在哪？ <Commander> 盘子。 <Commander> 尝一口番茄那块。 <Driver> 完成。 <Commander> 我们结束了。


Python script:
Python 脚本：


target_tomato = InteractionObject("Tomato")
target_tomato = InteractionObject("Tomato")


target_knife = InteractionObject("Knife") # need knife to slice tomato
target_knife = InteractionObject("Knife") # 需要刀来切番茄


target_knife.go_to() # first get knife
target_knife.go_to() # 先把刀拿到手


target_knife.pickup()
target_knife.pickup()


target_tomato.go_to()
target_tomato.go_to()


target_tomato.slice()
target_tomato.slice()


target_tomato_slice1 = InteractionObject("TomatoSliced") # instantiate a tomato slice object after slicing tomato
target_tomato_slice1 = InteractionObject("TomatoSliced") # 切完番茄后实例化一份番茄片对象


target_potato = InteractionObject("Potato", landmark = "Microwave")
target_potato = InteractionObject("Potato", landmark = "Microwave")


target_potato.go_to()
target_potato.go_to()


target_potato.slice()
target_potato.slice()


target_potato_slice1 = InteractionObject("PotatoSliced") # instantiate a potato slice object after slicing potato
target_potato_slice1 = InteractionObject("PotatoSliced") # 切完土豆后实例化一份土豆片对象


target_knife.put_down() # put down the knife to free up the agent's hand
target_knife.put_down() # 放下刀子，腾出代理的手


target_plate = InteractionObject("Plate")
target_plate = InteractionObject("Plate")


target_tomato_slice1.pickup_and_place(target_plate) # pick up the tomato slice and place it on the plate
target_tomato_slice1.pickup_and_place(target_plate) # 把番茄切片拿起来并放到盘子上


target_potato_slice1.pickup_and_place(target_plate) # pick up the potato slice and place it on the plate
target_potato_slice1.pickup_and_place(target_plate) # 把土豆切片拿起来并放到盘子上


Example 2:
例子2：


dialogue: <Driver> how can I help? <Commander> please serve 4 slices of tomato on a plate. <Driver> sure. <Driver> where can i find the tomato? <Driver> an knife. <Commander> Tomato on countertop and knife in fridge. <Commander> there should be a plate on the right cupboard by the sink. <Driver> completed. <Commander> great work thanks.
对话：<Driver> 我该怎么帮你？ <Commander> 请在盘子上端上4片番茄。 <Driver> 好的。 <Driver> 我去哪里找番茄？ <Driver> 去拿一把刀。 <Commander> 番茄在台面上，刀在冰箱里。 <Commander> 水槽旁右边的橱柜里应该有一个盘子。 <Driver> 已完成。 <Commander> 干得好，谢谢。


Python script:
Python脚本：


target_tomato = InteractionObject("Tomato")
target_tomato = InteractionObject("Tomato")


target_knife = InteractionObject("Knife") # need knife to slice tomato
target_knife = InteractionObject("Knife") # 需要刀来切番茄


target_knife.go_to() # first get knife
target_knife.go_to() # 先拿到刀


target_knife.pickup()
target_knife.pickup()


target_tomato.go_to()
target_tomato.go_to()


target_tomato.slice()
target_tomato.slice()


target_plate = InteractionObject("Plate", landmark = "Sink")
target_plate = InteractionObject("Plate", landmark = "Sink")


#place 4 slices of the cut tomato on the plate
#把切好的番茄4片放到盘子里


target_tomato_slice1 = InteractionObject("TomatoSliced") # instantiate first slice of tomato
target_tomato_slice1 = InteractionObject("TomatoSliced") # 实例化第一片番茄切片


target_tomato_slice2 = InteractionObject("TomatoSliced") # instantiate second slice of tomato
target_tomato_slice2 = InteractionObject("TomatoSliced") # 实例化第二片番茄切片


target_tomato_slice3 = InteractionObject("TomatoSliced") # instantiate third slice of tomato
target_tomato_slice3 = InteractionObject("TomatoSliced") # 实例化第三片番茄切片


target_tomato_slice4 = InteractionObject("TomatoSliced") # instantiate fourth slice of tomato
target_tomato_slice4 = InteractionObject("TomatoSliced") # 实例化第四片番茄切片


target_tomato_slice1.pickup_and_place(target_plate) # pick up the first tomato slice and place it on the plate
target_tomato_slice1.pickup_and_place(target_plate) # 把第一片番茄切片拿起并放到盘子上


target_tomato_slice2.pickup_and_place(target_plate) # pick up the second tomato slice and place it on the plate
target_tomato_slice2.pickup_and_place(target_plate) # 把第二片番茄切片拿起并放到盘子上


target_tomato_slice3.pickup_and_place(target_plate) # pick up the third tomato slice and place it on the plate
target_tomato_slice3.pickup_and_place(target_plate) # 把第三片番茄切片拿起并放到盘子上


target_tomato_slice4.pickup_and_place(target_plate) # pick up the fourth tomato slice and place it on the plate
target_tomato_slice4.pickup_and_place(target_plate) # 把第四片番茄切片拿起并放到盘子上


Example 3:
示例 3：


dialogue: <Driver> hi, what is task. <Commander> make a plate of toast. <Commander> bread in the sink. <Driver> knife. <Commander> on the chair beside the table. <Commander> clean the plate. <Driver> there is a clean one, do i need to clean the other one. <Commander> you can use the clean one. <Driver> done. <Commander> good job. <Driver> :).
dialogue: <Driver> 嗨，这个任务是什么。 <Commander> 做一盘吐司。 <Commander> 面包放在水槽里。 <Driver> 刀。 <Commander> 放在桌子旁边的椅子上。 <Commander> 把盘子洗干净。 <Driver> 有个干净的了，我还需要洗另一个吗。 <Commander> 你可以用那个干净的。 <Driver> 搞定了。 <Commander> 干得好。 <Driver> :).


Python script:
Python 脚本：


target_bread = InteractionObject("Bread", landmark = "Sink")
target_bread = InteractionObject("Bread", landmark = "Sink")


target_knife = InteractionObject("Knife", landmark = "Chair")
target_knife = InteractionObject("Knife", landmark = "Chair")


target_toaster = InteractionObject("Toaster") # need knife to slice bread
target_toaster = InteractionObject("Toaster") # 需要刀来切面包


target_knife.go_to() # first get knife
target_knife.go_to() # 先拿到刀


---



target_knife.pickup()
target_knife.pickup()


target_bread.go_to()
target_bread.go_to()


target_bread.slice() # slice bread using knife
target_bread.slice() # 用刀切面包


target_knife.put_down() # put down the knife to free up the agent's hand
target_knife.put_down() # 放下刀具以腾出智能体的手


target_bread_slice1 = InteractionObject("BreadSliced") # instantiate a bread slice object after
target_bread_slice1 = InteractionObject("BreadSliced") # 在此之后实例化一片面包切片对象


&nbsp;&nbsp;&nbsp;&nbsp;slicing bread
&nbsp;&nbsp;&nbsp;&nbsp;切面包


target_bread_slice1.pickup_and_place(target_toaster) # pick up the bread slice and place it in the
target_bread_slice1.pickup_and_place(target_toaster) # 抓起面包切片并放入


&nbsp;&nbsp;&nbsp;&nbsp;toaster
&nbsp;&nbsp;&nbsp;&nbsp;烤面包机


target_toaster.toggle_on() # turn on toaster to toast bread
target_toaster.toggle_on() # 打开烤面包机给面包加热


target_toaster.toggle_off() # turn off toaster to complete toasting process
target_toaster.toggle_off() # 关闭烤面包机以完成加热过程


#place toast on plate
# 把吐司放到盘子上


target_plate = InteractionObject("Plate")
target_plate = InteractionObject("Plate")


target_plate.clean()
target_plate.clean()


target_plate.put_down()
target_plate.put_down()


target_bread_slice1.pickup_and_place(target_plate) # pick up the toasted bread slice and place it on
target_bread_slice1.pickup_and_place(target_plate) # 抓起已烤好的面包切片并放到


&nbsp;&nbsp;&nbsp;&nbsp;the plate
&nbsp;&nbsp;&nbsp;&nbsp;盘子上


LLM output:
LLM 输出：


target_tomato = InteractionObject("Tomato")
target_tomato = InteractionObject("Tomato")


target_potato = InteractionObject("Potato")
target_potato = InteractionObject("Potato")


target_plate = InteractionObject("Plate")
target_plate = InteractionObject("盘子")


target_stove = InteractionObject("StoveBurner")
target_stove = InteractionObject("炉灶")


#Clean the plate
#清洗盘子


target_plate.go_to()
target_plate.go_to()


target_plate.clean()
target_plate.clean()


#Cook the potato
#烹饪土豆


target_potato.go_to()
target_potato.go_to()


target_potato.pickup()
target_potato.pickup()


target_stove.go_to()
target_stove.go_to()


target_potato.place(target_stove)
target_potato.place(target_stove)


target_stove.toggle_on()
target_stove.toggle_on()


target_stove.toggle_off()
target_stove.toggle_off()


#Place the cooked potato on the plate
#将煮好的土豆放到盘子上


target_potato = InteractionObject("Potato")
target_potato = InteractionObject("土豆")


target_potato.pickup()
target_potato.pickup()


target_plate.go_to()
target_plate.go_to()


target_potato.place(target_plate)
target_potato.place(target_plate)


#Place the tomato on the plate
#把番茄放到盘子上


target_tomato.go_to()
target_tomato.go_to()


target_tomato.pickup()
target_tomato.pickup()


target_plate.go_to()
target_plate.go_to()


target_tomato.place(target_plate)
target_tomato.place(target_plate)


---



Listing 9: Example of dialogue input, retrieved examples, and LLM output for a TEACh sample
图9：TEACh样本的对话输入、检索到的示例和LLM输出


---



Dialogue input:
对话输入：


<Driver> what do i do today<Commander> Hi. Please place two tissue boxes on a table. One is on the
<Driver>我今天要做什么<Commander> 你好。请把两个纸巾盒放到桌子上。一个在


&nbsp;&nbsp;&nbsp;&nbsp;gray chair in the corner.<Commander> The other is under the tv<Commander> tv*<Driver> where is
&nbsp;&nbsp;&nbsp;&nbsp;角落里的灰色椅子上。<Commander> 另一个在电视下面<Commander> 电视*<Driver> 另一个


&nbsp;&nbsp;&nbsp;&nbsp;the other one
&nbsp;&nbsp;&nbsp;&nbsp;在哪


## Retrieved Examples:
## 检索示例：


Example 1:
示例 1：


dialogue: <Driver> hi, what is task. <Commander> make a plate of toast. <Commander> bread in the sink.
dialogue: <Driver> 你好，任务是什么。 <Commander> 做一盘吐司。 <Commander> 面包在水槽里。


&nbsp;&nbsp;&nbsp;&nbsp;<Driver> knife. <Commander> on the chair beside the table. <Commander> clean the plate. <Driver>
&nbsp;&nbsp;&nbsp;&nbsp;<Driver> 刀。 <Commander> 在桌子旁边的椅子上。 <Commander> 清洗盘子。 <Driver>


&nbsp;&nbsp;&nbsp;&nbsp;there is a clean one, do i need to clean the other one. <Commander> you can use the clean one.
&nbsp;&nbsp;&nbsp;&nbsp;有一个是干净的，我需要清洗另一个吗。 <Commander> 你可以用干净的那个。


&nbsp;&nbsp;&nbsp;&nbsp;<Driver> done. <Commander> good job. <Driver> :).
&nbsp;&nbsp;&nbsp;&nbsp;<Driver> 完成了。 <Commander> 干得好。 <Driver> :)。


Python script:
Python script:


target_bread = InteractionObject("Bread", landmark = "Sink")
target_bread = InteractionObject("Bread", landmark = "Sink")


target_knife = InteractionObject("Knife", landmark = "Chair")
target_knife = InteractionObject("Knife", landmark = "Chair")


target_toaster = InteractionObject("Toaster") # need knife to slice bread
target_toaster = InteractionObject("Toaster") # need knife to slice bread


target_knife.go_to() # first get knife
target_knife.go_to() # first get knife


target_knife.pickup()
target_knife.pickup()


target_bread.go_to()
target_bread.go_to()


target_bread.slice() # slice bread using knife
target_bread.slice() # slice bread using knife


target_knife.put_down() # put down the knife to free up the agent's hand
target_knife.put_down() # put down the knife to free up the agent's hand


target_bread_slice1 = InteractionObject("BreadSliced") # instantiate a bread slice object after
target_bread_slice1 = InteractionObject("BreadSliced") # instantiate a bread slice object after


&nbsp;&nbsp;&nbsp;&nbsp;slicing bread
&nbsp;&nbsp;&nbsp;&nbsp;切面包


target_bread_slice1.pickup_and_place(target_toaster) # pick up the bread slice and place it in the
target_bread_slice1.pickup_and_place(target_toaster) # 拿起这片面包并把它放到


&nbsp;&nbsp;&nbsp;&nbsp;toaster
&nbsp;&nbsp;&nbsp;&nbsp;烤面包机


target_toaster.toggle_on() # turn on toaster to toast bread
target_toaster.toggle_on() # 打开烤面包机为面包加热


target_toaster.toggle_off() # turn off toaster to complete toasting process
target_toaster.toggle_off() # 关闭烤面包机以完成烘烤过程


#place toast on plate
#把吐司放到盘子上


target_plate = InteractionObject("Plate")
target_plate = InteractionObject("Plate")


target_plate.clean()
target_plate.clean()


target_plate.put_down()
target_plate.put_down()


target_bread_slice1.pickup_and_place(target_plate) # pick up the toasted bread slice and place it on
target_bread_slice1.pickup_and_place(target_plate) # 拿起已烘好的面包片并把它放到


&nbsp;&nbsp;&nbsp;&nbsp;the plate
&nbsp;&nbsp;&nbsp;&nbsp;盘子上


---



Example 2:
示例2：


dialogue: <Driver> how can I help? <Commander> please serve 4 slices of tomato on a plate. <Driver> sure. <Driver> where can i find the tomato? <Driver> an knife. <Commander> Tomato on countertop and knife in fridge. <Commander> there should be a plate on the right cupboard by the sink. <Driver> completed. <Commander> great work thanks.
对话：<Driver>我能怎么帮你？ <Commander>请在盘子上端上4片番茄。 <Driver>好的。 <Driver>我去哪儿找番茄？ <Driver>在一把刀。 <Commander>番茄在台面上，刀在冰箱里。 <Commander>水槽旁右侧的柜子里应该有一个盘子。 <Driver>完成。 <Commander>干得好，谢谢。


Python script:
Python脚本：


target_tomato = InteractionObject("Tomato", landmark = "CounterTop")
target_tomato = InteractionObject("Tomato", landmark = "CounterTop")


target_knife = InteractionObject("Knife", landmark = "Fridge") # need knife to slice tomato
target_knife = InteractionObject("Knife", landmark = "Fridge") # 需要刀来切番茄


target_knife.go_to() # first get knife
target_knife.go_to() # 先拿到刀


target_knife.pickup()
target_knife.pickup()


target_tomato.go_to()
target_tomato.go_to()


target_tomato.slice()
target_tomato.slice()


target_plate = InteractionObject("Plate", landmark = "Sink")
target_plate = InteractionObject("Plate", landmark = "Sink")


#place 4 slices of the cut tomato on the plate
#把切好的番茄四片放到盘子里


target_tomato_slice1 = InteractionObject("TomatoSliced") # instantiate first slice of tomato
target_tomato_slice1 = InteractionObject("TomatoSliced") # 创建第一片番茄


target_tomato_slice2 = InteractionObject("TomatoSliced") # instantiate second slice of tomato
target_tomato_slice2 = InteractionObject("TomatoSliced") # 创建第二片番茄


target_tomato_slice3 = InteractionObject("TomatoSliced") # instantiate third slice of tomato
target_tomato_slice3 = InteractionObject("TomatoSliced") # 创建第三片番茄


target_tomato_slice4 = InteractionObject("TomatoSliced") # instantiate fourth slice of tomato
target_tomato_slice4 = InteractionObject("TomatoSliced") # 创建第四片番茄


target_tomato_slice1.pickup_and_place(target_plate) # pick up the first tomato slice and place it on the plate
target_tomato_slice1.pickup_and_place(target_plate) # 拿起第一片番茄并放到盘子上


target_tomato_slice2.pickup_and_place(target_plate) # pick up the second tomato slice and place it on the plate
target_tomato_slice2.pickup_and_place(target_plate) # 拿起第二片番茄并放到盘子上


target_tomato_slice3.pickup_and_place(target_plate) # pick up the third tomato slice and place it on the plate
target_tomato_slice3.pickup_and_place(target_plate) # 拿起第三片番茄并放到盘子上


target_tomato_slice4.pickup_and_place(target_plate) # pick up the fourth tomato slice and place it on the plate
target_tomato_slice4.pickup_and_place(target_plate) # 拿起第四片番茄并放到盘子上


Example 3:
示例 3：


dialogue: <Driver> How can I help? <Commander> put all the scrub brushes on the counter top. <Driver> where can I find them? <Commander> one is on top of the toilet. <Driver> there is only a cloth. <Driver> I put it on the counter top. <Driver> are there more? <Commander> can you try with the brush to the side of the toilet please? <Driver> okay.
dialogue: <Driver> 我怎么帮你？ <Commander> 把所有刷子放到台面上。 <Driver> 我在哪能找到它们？ <Commander> 有一个在马桶上方。 <Driver> 只有一块布。 <Driver> 我把它放到台面上了。 <Driver> 还有更多吗？ <Commander> 你能试试在马桶旁边找找刷子吗？ <Driver> 好的。


Python script:
Python 脚本：


target_scrub_brush1 = InteractionObject("ScrubBrush", landmark = "Toilet")
target_scrub_brush1 = InteractionObject("ScrubBrush", landmark = "Toilet")


target_counter_top = InteractionObject("CounterTop")
target_counter_top = InteractionObject("CounterTop")


target_scrub_brush1.pickup_and_place(target_counter_top) # pick up the first ScrubBrush and place it on the CounterTop
target_scrub_brush1.pickup_and_place(target_counter_top) # 拿起第一个 ScrubBrush，并把它放到 CounterTop 上


target_scrub_brush2 = InteractionObject("ScrubBrush", landmark = "Toilet")
target_scrub_brush2 = InteractionObject("ScrubBrush", landmark = "Toilet")


target_scrub_brush2.pickup_and_place(target_counter_top) # pick up the second ScrubBrush and place it on the CounterTop
target_scrub_brush2.pickup_and_place(target_counter_top) # 拿起第二个 ScrubBrush，并把它放到 CounterTop 上


LLM output:
LLM 输出：


target_tissue_box1 = InteractionObject("TissueBox", landmark = "Chair")
target_tissue_box1 = InteractionObject("TissueBox", landmark = "Chair")


target_table = InteractionObject("Table")
target_table = InteractionObject("Table")


target_tissue_box1.pickup_and_place(target_table) # pick up the first TissueBox and place it on the Table
target_tissue_box1.pickup_and_place(target_table) # 拿起第一个 TissueBox，并把它放到 Table 上


target_tissue_box2 = InteractionObject("TissueBox", landmark = "TVStand")
target_tissue_box2 = InteractionObject("TissueBox", landmark = "TVStand")


target_tissue_box2.pickup_and_place(target_table) # pick up the second TissueBox and place it on the Table
target_tissue_box2.pickup_and_place(target_table) # 拿起第二个 TissueBox，并把它放到 Table 上


Listing 10: Example of dialogue input, retrieved examples, and LLM output for a TEACh sample
表 10：TEACh 样例的对话输入、检索示例以及 LLM 输出示例


---



Dialogue input:
对话输入：


<Driver> hi how can i help<Commander> Can you prepare me a coffee in a clean mug<Commander> There
<Driver> 嗨，我能怎么帮你<Commander> 你能在一个干净的杯子里给我准备一杯咖啡吗<Commander> 有


&nbsp;&nbsp;&nbsp;&nbsp;should be a mug on the white table<Commander> it might need cleaning first<Commander> that's
&nbsp;&nbsp;&nbsp;&nbsp;白色桌子上应该有一个杯子<Commander> 也许需要先清洗<Commander> 那就


&nbsp;&nbsp;&nbsp;&nbsp;great<Driver> are they not already on a plate?
&nbsp;&nbsp;&nbsp;&nbsp;great<Driver> 它们不是已经放在盘子上了吗？


Retrieved Examples:
检索示例：


Example 1:
示例 1：


dialogue: <Commander> Prepare coffee in a clean mug.
dialogue: <Commander> 在一个干净的杯子里准备咖啡。


Python script:
Python script:


target_mug = InteractionObject("Mug")
target_mug = InteractionObject("Mug")


target_mug.go_to()
target_mug.go_to()


target_mug.pickup()
target_mug.pickup()


#clean mug in the sink
#在水槽里清洗杯子


target_mug.clean()
target_mug.clean()


target_coffee_machine = InteractionObject("CoffeeMachine")
target_coffee_machine = InteractionObject("CoffeeMachine")


target_coffee_machine.go_to()
target_coffee_machine.go_to()


target_mug.place(target_coffee_machine)
target_mug.place(target_coffee_machine)


target_coffee_machine.toggle_on() # turn on coffee machine to make coffee
target_coffee_machine.toggle_on() # 打开咖啡机煮咖啡


target_coffee_machine.toggle_off() # turn off coffee machine after done
target_coffee_machine.toggle_off() # 完成后关闭咖啡机


Example 2:
示例 2：


dialogue: <Driver> what can i for you today? <Commander> could you boil a potato? <Driver> sure thing!
对话：<Driver>今天我能为你做什么？<Commander>你能把一个土豆煮一下吗？<Driver>没问题！


&nbsp;&nbsp;&nbsp;&nbsp;<Driver> are there any pots? <Commander> pots for boiling potato? <Driver> ves. <Commander> just
&nbsp;&nbsp;&nbsp;&nbsp;<Driver>有锅吗？<Commander>煮土豆用的锅？<Driver>有。<Commander>只是


&nbsp;&nbsp;&nbsp;&nbsp;the one that you were holding earlier and a couple others of the same size. <Commander> does it
&nbsp;&nbsp;&nbsp;&nbsp;你之前拿着的那个，还有几个同样大小的。<Commander>它是不是


&nbsp;&nbsp;&nbsp;&nbsp;<Commander> pour the mugs water into the pot. <Driver> thanks. <Commander> you're welcome!
&nbsp;&nbsp;&nbsp;&nbsp;<Commander>把杯子里的水倒进锅里。<Driver>谢谢。<Commander>不客气！


&nbsp;&nbsp;&nbsp;&nbsp;<Commander> excellent thank you!
&nbsp;&nbsp;&nbsp;&nbsp;<Commander>太好了，谢谢！


Python script:
Python 脚本：


target_pot = InteractionObject("Pot")
target_pot = InteractionObject("Pot")


target_mug = InteractionObject("Mug", landmark = "Pot")
target_mug = InteractionObject("Mug", landmark = "Pot")


target_mug.go_to()
target_mug.go_to()


target_mug.pickup()
target_mug.pickup()


target_mug.pour(target_pot)
target_mug.pour(target_pot)


target_mug.put_down() # put down mug on any location to free up agent's hand
target_mug.put_down() # 把杯子放在任何位置，以释放智能体的手


target_potato = InteractionObject("Potato")
target_potato = InteractionObject("Potato")


target_potato.pickup_and_place(target_pot) # pick up the potato and place it in the pot
target_potato.pickup_and_place(target_pot) # 拿起土豆并把它放进锅里


target_stove_knob = InteractionObject("StoveKnob") # turn on the stove to boil potato by turning on
target_stove_knob = InteractionObject("StoveKnob") # 通过打开炉子来把土豆煮熟


&nbsp;&nbsp;&nbsp;&nbsp;the stove
&nbsp;&nbsp;&nbsp;&nbsp;炉子


target_stove_knob.go_to()
target_stove_knob.go_to()


target_stove_knob.toggle_on() # boil water
target_stove_knob.toggle_on() # 煮水


target_stove_knob.toggle_off() # complete boiling process by turning off
target_stove_knob.toggle_off() # 通过关闭完成沸腾过程


Example 3:
例 3：


dialogue: <Driver> hi, what is task. <Commander> make a plate of toast. <Commander> bread in the sink.
对话：<Driver> 嗨，这个任务是什么。<Commander> 做一盘吐司。<Commander> 面包在水槽里。


&nbsp;&nbsp;&nbsp;&nbsp;<Driver> knife. <Commander> on the chair beside the table. <Commander> clean the plate. <Driver>
&nbsp;&nbsp;&nbsp;&nbsp;<Driver> 刀。<Commander> 在桌子旁边的椅子上。<Commander> 把盘子洗干净。<Driver>


&nbsp;&nbsp;&nbsp;&nbsp;there is a clean one, do i need to clean the other one. <Commander> you can use the clean one.
&nbsp;&nbsp;&nbsp;&nbsp;有一个是干净的，我需要再洗另一个吗。<Commander> 你可以用那个干净的。


&nbsp;&nbsp;&nbsp;&nbsp;<Driver> done. <Commander> good job. <Driver> :).
&nbsp;&nbsp;&nbsp;&nbsp;<Driver> 好了。<Commander> 干得好。<Driver> ：）。


Python script:
Python 脚本：


target_bread = InteractionObject("Bread", landmark = "Sink")
target_bread = InteractionObject("面包", landmark = "水槽")


target_knife = InteractionObject("Knife", landmark = "Chair")
target_knife = InteractionObject("刀", landmark = "椅子")


target_toaster = InteractionObject("Toaster", "Toaster") # need knife to slice bread
target_toaster = InteractionObject("烤面包机", "烤面包机") # 需要刀来切面包


target_knife.go_to() # first get knife
target_knife.go_to() # 先拿刀


target_knife.pickup()
target_knife.pickup()


target_bread.go_to()
target_bread.go_to()


target_bread.slice() # slice bread using knife
target_bread.slice() # 用刀切面包


target_knife.put_down() # put down the knife to free up the agent's hand
target_knife.put_down() # 放下刀具，腾出代理的手


target_bread_slice1 = InteractionObject("BreadSliced") # instantiate a bread slice object after
target_bread_slice1 = InteractionObject("BreadSliced") # 在之后实例化一片面包


&nbsp;&nbsp;&nbsp;&nbsp;slicing bread
&nbsp;&nbsp;&nbsp;&nbsp;切面包


target_bread_slice1.pickup_and_place(target_toaster) # pick up the bread slice and place it in the
target_bread_slice1.pickup_and_place(target_toaster) # 抓起这片面包并把它放入


&nbsp;&nbsp;&nbsp;&nbsp;toaster
&nbsp;&nbsp;&nbsp;&nbsp;烤面包机


target_toaster.toggle_on() # turn on toaster to toast bread
target_toaster.toggle_on() # 打开烤面包机给面包加热


target_toaster.toggle_off() # turn off toaster to complete toasting process
target_toaster.toggle_off() # 关闭烤面包机，完成加热过程


#place toast on plate
#把吐司放到盘子上


target_plate = InteractionObject("Plate")
target_plate = InteractionObject("Plate")


target_plate.clean()
target_plate.clean()


target_plate.put_down()
target_plate.put_down()


target_bread_slice1.pickup_and_place(target_plate) # pick up the toasted bread slice and place it on
target_bread_slice1.pickup_and_place(target_plate) # 抓起已烤好的面包片并放到


&nbsp;&nbsp;&nbsp;&nbsp;the plate
&nbsp;&nbsp;&nbsp;&nbsp;盘子上


---



---



LLM output:
LLM 输出：


target_table = InteractionObject("Table")
target_table = InteractionObject("Table")


target_mug = InteractionObject("Mug", landmark = "Table")
target_mug = InteractionObject("Mug", landmark = "Table")


target_mug.go_to()
target_mug.go_to()


target_mug.pickup()
target_mug.pickup()


#clean mug in the sink
#在水槽里清洗杯子


&nbsp;&nbsp;&nbsp;&nbsp;target_sink = InteractionObject("Sink")
&nbsp;&nbsp;&nbsp;&nbsp;target_sink = InteractionObject("Sink")


&nbsp;&nbsp;&nbsp;&nbsp;target_mug.place(target_sink)
&nbsp;&nbsp;&nbsp;&nbsp;target_mug.place(target_sink)


&nbsp;&nbsp;&nbsp;&nbsp;target_mug.clean()
&nbsp;&nbsp;&nbsp;&nbsp;target_mug.clean()


&nbsp;&nbsp;&nbsp;&nbsp;target_mug.pickup()
&nbsp;&nbsp;&nbsp;&nbsp;target_mug.pickup()


&nbsp;&nbsp;&nbsp;&nbsp;target_coffee_machine = InteractionObject("CoffeeMachine")
&nbsp;&nbsp;&nbsp;&nbsp;target_coffee_machine = InteractionObject("CoffeeMachine")


&nbsp;&nbsp;&nbsp;&nbsp;target_coffee_machine.go_to()
&nbsp;&nbsp;&nbsp;&nbsp;target_coffee_machine.go_to()


&nbsp;&nbsp;&nbsp;&nbsp;target_mug.place(target_coffee_machine)
&nbsp;&nbsp;&nbsp;&nbsp;target_mug.place(target_coffee_machine)


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;target_coffee_machine.toggle_on() # turn on coffee machine to make coffee
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;target_coffee_machine.toggle_on() # 打开咖啡机煮咖啡


&nbsp;&nbsp;&nbsp;&nbsp;target_coffee_machine.toggle_off() # turn off coffee machine after done
&nbsp;&nbsp;&nbsp;&nbsp;target_coffee_machine.toggle_off() # 完成后关闭咖啡机


---
