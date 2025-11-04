# A Survey on GUI Agents with Foundation Models Enhanced by Reinforcement Learning

Jiahao Li, Kaer Huang

Lenovo Research

jiahaoli0301@gmail.com

## Abstract

Graphical User Interface (GUI) agents, driven by Multi-modal Large Language Models (MLLMs), have emerged as a promising paradigm for enabling intelligent interaction with digital systems. This paper provides a structured summary of recent advances in GUI agents, focusing on architectures enhanced by Reinforcement Learning (RL). We first formalize GUI agent tasks as Markov Decision Processes and discuss typical execution environments and evaluation metrics. We then review the modular architecture of (M)LLM-based GUI agents, covering Perception, Planning, and Acting modules, and trace their evolution through representative works. Furthermore, we categorize GUI agent training methodologies into Prompt-based, Supervised Fine-Tuning (SFT)-based, and RL-based approaches, highlighting the progression from simple prompt engineering to dynamic policy learning via RL. Our summary illustrates how recent innovations in multimodal perception, decision reasoning, and adaptive action generation have significantly improved the generalization and robustness of GUI agents in complex real-world environments. We conclude by identifying key challenges and future directions for building more capable and reliable GUI agents.

## Contents

1 Introduction

2 Problem Formulation 2

3 Benchmarks 2

4 (M)LLM-based GUI Agent Architectures 3

4.1 Perception 3

4.2 Planning 4

4.3 Acting 5

5 (M)LLM-based GUI Agent Taxonomy 5

5.1 Modal Taxonomy 6

5.1.1 LLM-based Agent 6

5.1.2 MLLM-based Agent 6

5.2 Training Taxonomy 7

5.2.1 Prompt-based Methods 7

5.2.2 SFT-based Method 7

5.2.3 RL-based Method 8

6 Challenges 9

7 Conclusions 9

## 1 Introduction

With the continuous advancements of LLMs in the fields of Natural Language Processing (NLP) and Computer Vision (CV), LLMs have demonstrated impressive performance in various downstream tasks after being trained on large-scale corpora using the "next token prediction" paradigm ${}^{\left\lbrack  {23}\right\rbrack  \left\lbrack  {30}\right\rbrack  \left\lbrack  {21}\right\rbrack  }$ . LLMs have progressively evolved from a simple conversational chatbot ${}^{\left\lbrack  {24}\right\rbrack  \left\lbrack  {16}\right\rbrack  }{}^{\left\lbrack  5\right\rbrack  \left\lbrack  6\right\rbrack  }$ to autonomous agents ${}^{\left\lbrack  1\right\rbrack  \left\lbrack  {26}\right\rbrack  \left\lbrack  {19}\right\rbrack  }$ capable of planning ${}^{\left\lbrack  {29}\right\rbrack  }$ ,tool use ${}^{\left\lbrack  {22}\right\rbrack  }$ ,and memory management ${}^{\left\lbrack  {17}\right\rbrack  \left\lbrack  {18}\right\rbrack  }$ to perform and complete complex tasks. Building on this foundation, a series of works have emerged that utilize LLMs to interact with digital systems ${}^{\left\lbrack  {28}\right\rbrack  \left\lbrack  8\right\rbrack  }\left\lbrack  8\right\rbrack  \left\lbrack  {36}\right\rbrack  \left\lbrack  {13}\right\rbrack  \left\lbrack  {20}\right\rbrack  \left\lbrack  {14}\right\rbrack  \left\lbrack  3\right\rbrack  \left\lbrack  4\right\rbrack  \left\lbrack  {34}\right\rbrack  \left\lbrack  {10}\right\rbrack  \left\lbrack  {25}\right\rbrack$ (e.g., smartphones or computers), where LLMs interact with digital devices through Graphical User Interfaces (GUIs) in the same way as humans.

<!-- Meanless: oSASSIN FISO COSIONSAS -->


This paradigm holds significant research and development potential, as GUIs are integral to nearly all digital devices, smartphones, tablets, desktops, and even televisions, used in daily human activities such as work, learning, and leisure. In the era of rapid AI development, using LLM-based agents to control digital devices to fulfill human user needs can significantly enhance the user experience, offering more convenient and efficient services, and demonstrating great value potential. However, previous studies based on traditional rule-based and RL methods have struggled with tasks resembling human interactions ${}^{\left\lbrack  9\right\rbrack  }$ , limiting their applicability. Recent advances in LLMs and MLLMs have significantly bolstered their capabilities in semantic comprehension and cognitive reasoning. Agents built on (M)LLMs can effectively interpret and integrate textual and visual input, enabling them to devise detailed strategies to tackle complex tasks. These breakthroughs also provide opportunities to address previously mentioned challenges in handling complex tasks, making it possible for agents to autonomously complete user tasks in GUIs.

This paper aims to summarize recent learning experiences and provide an overview of the current novel and influential approaches for GUI agents. Specifically, it covers the foundational concepts of GUI agents, the classification of execution environments, architectural frameworks, and key methodological categories.

## 2 Problem Formulation

A GUI agent refers to an agent that autonomously interacts with digital devices (such as smartphones or desktop computers) through GUIs. It makes decisions and generates plans based on user-defined task instructions and the current screen context, executing actions through interaction with clickable or typeable elements to achieve user goals.

The interaction process between GUI agents and their environments is commonly formalized as a Markov Decision Process (MDP), denoted as $M = \{ S,A,T,r,\gamma \}$ . In this formulation, S represents the state space, defined as the set of all possible screen captures of the digital device. A denotes the set of permissible actions, such as clicking, typing, and other forms of interaction on the screen. The transition function $T : S \times  A \times  S \rightarrow  \left\lbrack  {0,1}\right\rbrack$ defines the probability distribution over the next states given a current state and an action. The reward function $r : S \times  A \rightarrow  R$ assigns a scalar feedback value to each state-action pair. $\gamma$ is the discount factor. The goal of the GUI agent is to learn a decision policy $\pi$ that maximizes returns. We typically represent $\pi \left( {a \mid  s}\right)  \in  \left\lbrack  {0,1}\right\rbrack$ as the probability of selecting action a in state s under the policy $\pi$ .

## 3 Benchmarks

The execution environments of GUI agents usually include static environment datasets and dynamic interactive environments. In static settings, the environment remains constant, allowing GUI agents to work within predefined datasets without needing to account for environmental changes in action planning. In dynamic environments, the state may evolve after each action, necessitating that agents observe the updated state to determine subsequent actions. Dynamic environments are generally classified into simulated and real-world settings. Simulated environments are usually cleaner, free from disturbances like pop-up ads, and provide a standardized interactive setting for testing. In real environments, the agent must observe and interact with the digital device like humans. Although this setup more accurately reflects real-world scenarios, it also exposes agents to numerous unpredictable factors.

<!-- Meanless: 2 -->


The success rate is the most common metric used to evaluate a GUI agent, reflecting its ability to complete tasks effectively. In addition, execution efficiency is another common evaluation metric, often measured by the number of steps, time, or cost required to complete a task. Notably, for dynamic interactive environments, due to uncertainty from environmental changes, the fixed evaluation metrics mentioned earlier may not be applicable. As a result, human experts are required for manual assessment, which compromises reproducibility and requires significant labor costs. Consequently, recent research has explored leveraging MLLMs for automating the evaluation process, aiming to reduce human involvement. However, due to the hallucination issues in LLMs, ensuring evaluation accuracy while reducing human effort remains a critical research challenge.

## 4 (M)LLM-based GUI Agent Architectures

GUI agents are expected to autonomously operate digital devices operations to meet human task requirements. Typically, GUI agents receive a task instruction and the current screen state as input, and generate a sequence of planned actions to achieve the objective. As depicted in Figure 1.The architecture generally consists of three key modules: Perception, Planning, and Acting.

### 4.1 Perception

The Perception module is responsible for perceiving and understanding the GUI by extracting semantic information from interactive elements like buttons, text boxes, and icons, laying the groundwork for subsequent planning and action execution. For unimodal language models, the GUI perception module mainly relies on accessible APIs provided by operating systems or applications, extracting the type and position of interface elements via the view hierarchy and feeding this symbolic information into the language model. These methods offer structured interface data suitable for model processing, though their effectiveness heavily depends on the developers' proper implementation. However, when dealing with highly dynamic elements such as custom drawing canvases or game environments, the view hierarchy often fails to capture complete visual content, rendering the perception module ineffective. In contrast, for MLLMs, GUI agents can directly perceive the environment using visual information from screen-shots. For example, OmniParser ${}^{\left\lbrack  {12}\right\rbrack  }$ uses existing MLLMs (such as GPT-4V) to convert screen captures into structured representations of User Interface(UI) elements, extracting interface information directly from visual inputs and avoiding reliance on Accessibility APIs. As MLLMs continue to evolve, GUI perception has increasingly incorporated multimodal input sources for richer understanding. MP-GUI ${}^{\left\lbrack  {28}\right\rbrack  }$ introduces three specialized perceivers: a Textual Perceiver (TxP) for extracting text information, a Graphical Perceiver (GaP) for detecting graphical elements such as icons and buttons, and a Spatial Perceiver (SpP) for modeling spatial relationships between elements. By fusing the outputs of these perceivers, the model gains a more comprehensive understanding of the interface. Furthermore, ${\text{ CogAgent }}^{\left\lbrack  8\right\rbrack  }$ introduces a lightweight high-resolution cross-attention module that supplements the original low-resolution large image encoder with a high-resolution small image encoder, enhancing perception of complex GUI interfaces through cross-attention with the underlying Vision-Language Models(VLMs). Real-world GUI environments are dynamic, requiring perception modules to handle interface changes (e.g., loading animations, pop-ups) and noise (e.g., ads, notifications). UI-Hawk ${}^{\left\lbrack  {36}\right\rbrack  }$ highlights the use of historical screen context in GUI navigation, leveraging history-aware visual encoders and efficient resampling modules to enhance comprehension of changing interfaces. Additionally, the perception of screen visuals may raise privacy concerns due to the presence of sensitive information.

<!-- Meanless: 3 -->


<!-- Media -->

<!-- figureText: Digital Systems<br>Perception<br>Planning<br>Acting<br>Task Instruction Interface Information<br>High-level Operation Plans<br>Unimodal Perceiver<br>Input Block<br>Planner<br>Action Maker<br>MultiModal Perceiver<br>Dynamic Planning -->

<img src="https://cdn.noedgeai.com/bo_d41gkq3ef24c73d3u15g_3.jpg?x=175&y=191&w=1303&h=728&r=0"/>

Figure 1: Architecture of GUI agents powered by (Multi-modal) Large Language Models. Comprising three interconnected modules: Perception, Planning, and Acting. The Perception module is responsible for perceiving and understanding the GUI state, the Planning module formulates high-level action strategies, and the Acting module executes operations, collectively enabling intelligent interaction with the GUI.

<!-- Media -->

### 4.2 Planning

In modern (M)LLM-driven GUI agent architectures, the Planning module is responsible for translating perceived interface information and user task requirements into concrete action plans. With the development of VLMs, recent studies have explored the potential of using these models for planning and reasoning. RL4VLM[32] introduced an approach that integrates reinforcement learning with chain-of-thought (CoT) reasoning in VLMs, where the model first outputs intermediate reasoning steps before the final action, and is fine-tuned using environment-based reward signals. This method employs CoT reasoning to guide the VLM in generating intermediate reasoning steps, leading to the final text-based action output. To further refine the planning process, Mobile-Agent-v2[25] introduced an additional module within their multi-agent framework to generate more detailed execution plans. The system decouples the roles of the Planner, Decision Maker, and Reflector: the Planner outlines task progression to streamline navigation based on prior actions. The Reflector reviews execution outcomes after each step, providing feedback to dynamically adjust the plan. The recent dynamic planning method, D-PoT ${}^{\left\lbrack  {37}\right\rbrack  }$ , breaks the traditional two-stage process of pre-defined planning followed by execution and feedback, integrating planning and feedback into a unified process: Following each action, the model captures the latest screen state and execution history, continuously revising the action sequence in real time until the task is successfully completed.

<!-- Meanless: 4 -->


### 4.3 Acting

The GUI Acting module is responsible for converting high-level operation plans generated by the planning module into executable interactive actions on the interface, such as clicking, swiping, and text input [GUI-Odyssey ${}^{\left\lbrack  {11}\right\rbrack  }$ ; AppAgent ${}^{\left\lbrack  {35}\right\rbrack  }$ . The design of this module directly affects the execution efficiency and task success rate of agents in real-world environments. UI-R1 ${}^{\left\lbrack  {13}\right\rbrack  }$ introduces a carefully designed unified reward mechanism and integrates multimodal inputs to guide the model in learning generalizable operational strategies for mobile interfaces. This method significantly improves performance in action type recognition and target localization, especially in identifying action types and locating UI elements. Subsequently, dynamic decision-making and history-aware mechanisms were incorporated into the Acting module to better handle long-horizon tasks and environmental changes. For instance, the D-PoT framework ${}^{\left\lbrack  {37}\right\rbrack  }$ incorporates the concept of "Dynamic Planning of Thoughts" into the action generation process. After each action execution, the agent updates its remaining plan based on the latest interface feedback and execution history, enabling "plan-as-you-execute" behavior. This mechanism not only enhances adaptability in execution but also significantly improves the completion rate of complex tasks. To further improve generalization and cross-platform adaptability, GUI-R1 ${}^{\left\lbrack  {14}\right\rbrack  }$ proposed the Group Relative Policy Optimization (GRPO) algorithm, which models shared strategies across tasks, enabling a single model to be applicable across platforms such as Windows, Linux, Android, and Web. Auto-GUI ${}^{\lbrack {38}\rbrack }$ introduces a multi-modal chain-style action generation mechanism, combining visual and textual information to interact directly with the interface, thus avoiding the need for environment parsing or reliance on APIs. Overall, the Acting module is evolving from static mapping to an advanced module equipped with dynamic planning, history awareness, cross-platform adaptability, and end-to-end reasoning capabilities, with reinforcement learning, multimodal fusion, and module coordination becoming its core driving forces.

## 5 (M)LLM-based GUI Agent Taxonomy

With the development of MLLMs, the perception mechanisms of GUI agents have evolved from single-modality (e.g., text) to multimodal (e.g., a combination of vision and language). At the same time, training paradigms have diversified-from simple prompt-based guidance, to supervised fine-tuning, and further to RL-based policy optimization-providing strong technical support for GUI agents in diverse application scenarios. Figure 2 shows the taxonomy of GUI agents enhanced by foundation models.

<!-- Meanless: 5 -->


<!-- Media -->

<!-- figureText: LLM-based Agent<br>Processing language-based inputs (Text, OCR, DOM, HTLM) to understand and complete GUI tasks by modeling them as natural language problems.<br>Modal Taxonomy<br>MLLM-based Agent<br>Utilizing MLLMs to simultaneously perceive visual and linguistic information from screenshots and user instructions, enhancing generalization in complex GUIs.<br>Guiding (M)LLMs at inference with lightweight strategy using well-crafted prompts, integrating instructions, states, and actions through multimodal fusion and contextual guidance.<br>(M)LLM-based GUI Agent<br>Prompt-based Methods<br>Training Taxonomy<br>SFT-based Method<br>Fine-tuning pretrained (V)LMs or (M)LLMs on large datasets to learn the mapping from GUI states to actions, resulting in stable and precise policies.<br>RL-based Method<br>Enhancing long-term planning and decision-making by learning optimal policies through interaction and environment feedback. -->

<img src="https://cdn.noedgeai.com/bo_d41gkq3ef24c73d3u15g_5.jpg?x=156&y=172&w=1341&h=755&r=0"/>

Figure 2: Taxonomy of (M)LLM-based GUI Agents. This figure presents a classification of GUI agents based on their underlying model modalities and training methodologies.

<!-- Media -->

### 5.1 Modal Taxonomy

#### 5.1.1 LLM-based Agent

Early GUI agents primarily relied on LLMs to perform tasks. These approaches were unimodal, as the agents could only process language-based inputs such as user task descriptions, screen text extracted via OCR, or structured DOM/HTML data to understand and complete tasks. This typically required GUI agents to first convert the screen content into text-based inputs. Therefore, during the perception stage, the screen had to be parsed and translated into a set of object descriptions. The core idea behind such methods is to model GUI tasks as problems of natural language understanding and generation. The LLM learns to select appropriate actions in different contexts through prompting, fine-tuning, or reinforcement learning. Socratic Models ${}^{\left\lbrack  {31}\right\rbrack  }$ proposed transforming structured environmental data (e.g., HTML trees of web pages) into natural language prompts, from which the LLM generates operation steps, thereby mapping "language to action". We-bGPT ${}^{\left\lbrack  {15}\right\rbrack  }$ integrated LLMs with search engine APIs to perform web-based question answering tasks. While LLM-based approaches demand relatively low data and deployment costs, they struggle to directly handle visual signals (e.g., button shapes, colors), have limited capacity to model interface structures, and thus suffer from poor generalization, making them difficult to apply to complex real-world graphical interfaces.

#### 5.1.2 MLLM-based Agent

GUI interfaces are essentially multimodal interactive systems, consisting of visual components such as buttons, menus, and input boxes, which may not contain easily extractable structured textual information. As a result, traditional LLM-based approaches heavily rely on OCR and DOM structures, making it difficult to generalize to real-world apps or web applications. With the rapid development of MLLMs, GUI agents powered by MLLMs have increasingly become a focus of research in this field. Compared to traditional language-only methods, MLLMs can simultaneously perceive both visual and linguistic information, allowing them to complete tasks directly based on screen captures (or video frames) and textual instructions. This capability significantly enhances agents' generalization and adaptability in complex GUI environments. For example, WebGUM ${}^{\left\lbrack  6\right\rbrack  }$ introduced a multimodal task decomposition capability, where the model takes screen images and task instructions as input and performs staged perception, intent reasoning, and action generation, achieving notable results on MiniWoB++ and real-world web tasks. SeeAct ${}^{\left\lbrack  {39}\right\rbrack  }$ introduces a vision-language agent capable of visual context awareness and action localization, using MLLMs like GPT- 4V to reason over screen images and generate semantically relevant click or input actions. UI-TARS ${}^{\left\lbrack  {20}\right\rbrack  }$ proposed an agent architecture that combines native system operations with MLLM perceptual capabilities, emphasizing high-precision recognition and decision-making for complex GUI components in the Android system. Using screen snapshots and task descriptions, the model applies vision-language reasoning for holistic perception and operational planning.

<!-- Meanless: 6 -->


### 5.2 Training Taxonomy

#### 5.2.1 Prompt-based Methods

Prompt-based methods offer a lightweight strategy for building GUI agents by harnessing the pretrained strengths of (M)LLMs. Through well-crafted prompts, these models are guided to perform tasks at inference time without parameter tuning or high compute resources. The core of prompt-based methods lies in structured prompt engineering, which integrates user instructions, environment states, and action specifications into model-understandable inputs through multimodal fusion and contextual guidance. Zeng et al.[2022] proposed the Socratic Models framework ${}^{\left\lbrack  {31}\right\rbrack  }$ , which uses multi-round prompts to convert visual or structured data (e.g., HTML trees or OCR-extracted text) into natural language, enabling LLMs to output actions and complete zero-shot multimodal tasks such as video QA and robot planning. Subsequently, researchers extended prompt-based approaches to real-world web automation. WebAgent ${}^{\left\lbrack  7\right\rbrack  }$ incorporates HTML summaries and Python subprogram generation in its prompts, breaking long HTML documents into task-relevant segments and generating executable code for end-to-end web automation. With the rapid advancement of MLLMs, prompt-based approaches have further evolved to take screen-shots directly as input. UFO2[34] integrates VLMs and LLMs to construct multimodal inputs (screenshots + Accessibility API control attributes), generating cross-platform action sequences through hierarchical prompts to support automation across Windows, web, and mobile platforms.

#### 5.2.2 SFT-based Method

Supervised Fine-Tuning (SFT) is a widely adopted training paradigm for GUI agents. Unlike zero-shot or few-shot learning, this approach fine-tunes pretrained (V)LMs or MLLMs on large annotated datasets, enabling models to learn the mapping from GUI states to concrete actions, resulting in more stable and precise behavioral policies tailored to specific GUI tasks. SFT offers advantages such as a stable training process and effective specialization for target tasks. Auto-GUI ${}^{\lbrack {38}\rbrack }$ applies supervised fine-tuning on high-quality GUI interaction datasets (including screenshots, task descriptions, action histories, and annotated next-step actions), using GUI states and language tasks as input to predict the next action-such as action type, click coordinates, and text input. This method achieved remarkable performance on several GUI agent benchmarks, such as AITW and Mind2Web. Some efforts focus on building general-purpose GUI understanding models first, and then applying SFT on downstream tasks. For example, GUICourse ${}^{\left\lbrack  4\right\rbrack  }$ pretrains general VLMs (like Qwen-VL and MiniCPM-V) on the GUIEnv dataset to enhance OCR and element localization; Falcon-UI ${}^{\left\lbrack  3\right\rbrack  }$ is pretrained on the Insight-UI dataset to enhance the model's understanding of GUI environments. This model are then fine-tuned on Android and Web GUI datasets (e.g., AITW, AITZ, Android Control, and Mind2Web), achieving performance comparable to larger models like Qwen2VL; InfiGUIAgent ${}^{\left\lbrack  {10}\right\rbrack  }$ proposes a two-stage supervised fine-tuning framework. The first stage boosts GUI perception and localization, while the second uses synthetic data to teach reasoning and self-correction in complex multistep scenarios. Additionally, many works focus on dataset construction and task diversity. TongUI ${}^{\left\lbrack  {33}\right\rbrack  }$ constructs the GUI-Net dataset, containing 143K interaction records, by automatically crawling GUI operation traces from online tutorials. In conclusion, SFT-based methods continuously improve model robustness and task success rates in real GUI environments by expanding data scale and diversity and adopting staged training processes.

<!-- Meanless: 7 -->


#### 5.2.3 RL-based Method

RL is primarily used in GUI agent training to enhance long-term planning and decision-making capabilities in complex task scenarios. Recent research highlights RL's crucial role in enabling agents to learn optimal policies via interaction, particularly in handling intricate, multi-step tasks to boost decision-making and task success. Unlike SFT, which depends on human-labeled expert data, RL explores and learns policies through trial with environment feedback. This makes it well-suited for real GUI scenarios with long task sequences, ambiguous goals, and sparse feedback.

Recently, many works have integrated RL into MLLMs. RL4VLM ${}^{\left\lbrack  {32}\right\rbrack  }$ applies Proximal Policy Optimization (PPO) to VLMs, generating CoT reasoning at each step, converting the output into actions, receiving environment rewards and fine-tuning the VLMs accordingly. To balance data efficiency and cross-platform generalization, GUI-R1 ${}^{\left\lbrack  {14}\right\rbrack  }$ employs RL to enhance the GUI operation capabilities of VLMs in complex real-world scenarios. It incorporates DeepSeek-R1-style "regularized rewards" into GUI action prediction, using a unified action space and the Group Relative Policy Optimization (GRPO) algorithm to train across platforms such as Windows, Linux, MacOS, Android, and Web. For long-horizon tasks spanning multiple apps or APIs, short-term RL often suffers from local optima. Chen et al. [2025] propose LOOP ${}^{\left\lbrack  2\right\rbrack  }$ , a memory-efficient variant of PPO that trains IDAs (Intelligent Digital Agents) directly in target environments. It avoids value networks and requires only a single LLM instance in memory to perform end-to-end RL training within the AppWorld environment. Due to the high cost and brittleness of real environment interaction, environment-free RL approaches have emerged. Zheng et al. (2025) propose VEM (Value Environment Model) ${}^{\left\lbrack  {40}\right\rbrack  }$ , where a value function is trained on offline GUI interaction data to estimate long-term returns for arbitrary state-action pairs, serving as "virtual rewards" to guide policy optimization. In addition, asynchronous interaction approaches have been incorporated into RL training ${}^{\left\lbrack  {27}\right\rbrack  \left\lbrack  {26}\right\rbrack  }$ . Trajectories generated by GUI agents in simulated environments are stored in a replay buffer, serving as data for subsequent online model training.

<!-- Meanless: 8 -->


## 6 Challenges

Despite the rapid progress in (M)LLM-based GUI agents, several challenges remain:

Perception under Dynamic and Noisy Interfaces: Real-world GUIs are often dynamic, featuring pop-ups, advertisements, or frequent layout changes. Although modern GUI Perception module integrate multimodal inputs and history-aware modeling, accurately perceiving and adapting to these unpredictable variations remains a major difficulty.

Long-Horizon Planning and Execution: GUI tasks typically require multiple sequential operations. Ensuring consistency across multi-step plans, especially when intermediate states deviate from expectations, challenges current Planning modules. Dynamic replanning strategies ${}^{\left\lbrack  {37}\right\rbrack  }$ have made progress, but long-horizon scalable and reliable reasoning remains open.

Data Efficiency and Generalization: SFT-based approaches heavily rely on large annotated datasets. Despite advances like GUI-R1[14] using small datasets, achieving strong generalization across different platforms (Windows, Android, Web) with minimal supervision is still an ongoing challenge.

Evaluation and Benchmarking: Current benchmarks like Mind2Web and AITW primarily focus on success rates. However, finer-grained evaluations (e.g., plan optimality, robustness under interface drift) and standardized human-in-the-loop assessments are urgently needed to fully measure agent capabilities.

## 7 Conclusions

This paper presents a structured summary of recent developments in GUI agents enhanced by foundation models and RL techniques. We first introduced the task formulation, key execution environments, and standard evaluation metrics. Then, we reviewed the modular architecture of GUI agents - covering Perception, Planning, and Acting - along with representative advances in each component. We also discussed three major training paradigms: Prompt-based methods for lightweight deployment, SFT-based methods for domain-specific adaptation, and RL-based methods for dynamic policy learning.

Through this review, we observe a clear trend: GUI agents are evolving from static rule-based systems toward adaptive, perception-driven, and reasoning-capable agents, powered by MLLMs and optimized via RL. Despite impressive gains, challenges in perception robustness, long-horizon reasoning, data efficiency, and evaluation remain to be addressed.

Looking forward, integrating better semantic grounding, continual learning, human feedback, and asynchronous interaction will be crucial for building GUI agents that are truly capable of autonomously navigating complex, dynamic digital environments.

<!-- Meanless: 9 -->


## References

[1] Sébastien Bubeck, Varun Chandrasekaran, Ronen Eldan, Johannes Gehrke, Eric Horvitz, Ece Kamar, Peter Lee, Yin Tat Lee, Yuanzhi Li, Scott Lundberg, Harsha Nori, Hamid Palangi, Marco Tulio Ribeiro, and Yi Zhang. Sparks of artificial general intelligence: Early experiments with gpt-4, 2023.

[2] Kevin Chen, Marco Cusumano-Towner, Brody Huval, Aleksei Petrenko, Jackson Hamburger, Vladlen Koltun, and Philipp Krähenbühl. Reinforcement learning for long-horizon interactive llm agents, 2025.

[3] Wentong Chen, Junbo Cui, Jinyi Hu, Yujia Qin, Junjie Fang, Yue Zhao, Chongyi Wang, Jun Liu, Guirong Chen, Yupeng Huo, Yuan Yao, Yankai Lin, Zhiyuan Liu, and Maosong Sun. Guicourse: From general vision language models to versatile gui agents, 2024.

[4] Wentong Chen, Junbo Cui, Jinyi Hu, Yujia Qin, Junjie Fang, Yue Zhao, Chongyi Wang, Jun Liu, Guirong Chen, Yupeng Huo, Yuan Yao, Yankai Lin, Zhiyuan Liu, and Maosong Sun. Guicourse: From general vision language models to versatile gui agents, 2024.

[5] Sumit Kumar Dam, Choong Seon Hong, Yu Qiao, and Chaoning Zhang. A complete survey on llm-based ai chatbots, 2024.

[6] Hiroki Furuta, Kuang-Huei Lee, Ofir Nachum, Yutaka Matsuo, Aleksandra Faust, Shixi-ang Shane Gu, and Izzeddin Gur. Multimodal web navigation with instruction-finetuned foundation models, 2024.

[7] Izzeddin Gur, Hiroki Furuta, Austin Huang, Mustafa Safdari, Yutaka Matsuo, Douglas Eck, and Aleksandra Faust. A real-world webagent with planning, long context understanding, and program synthesis, 2024.

[8] Wenyi Hong, Weihan Wang, Qingsong Lv, Jiazheng Xu, Wenmeng Yu, Junhui Ji, Yan Wang, Zihan Wang, Yuxuan Zhang, Juanzi Li, Bin Xu, Yuxiao Dong, Ming Ding, and Jie Tang. Cogagent: A visual language model for gui agents, 2024.

[9] Evan Zheran Liu, Kelvin Guu, Panupong Pasupat, Tianlin Shi, and Percy Liang. Reinforcement learning on web interfaces using workflow-guided exploration, 2018.

[10] Yuhang Liu, Pengxiang Li, Zishu Wei, Congkai Xie, Xueyu Hu, Xinchen Xu, Shengyu Zhang, Xiaotian Han, Hongxia Yang, and Fei Wu. Infiguiagent: A multimodal generalist gui agent with native reasoning and reflection, 2025.

[11] Quanfeng Lu, Wenqi Shao, Zitao Liu, Fanqing Meng, Boxuan Li, Botong Chen, Siyuan Huang, Kaipeng Zhang, Yu Qiao, and Ping Luo. Gui odyssey: A comprehensive dataset for cross-app gui navigation on mobile devices, 2024.

[12] Yadong Lu, Jianwei Yang, Yelong Shen, and Ahmed Awadallah. Omniparser for pure vision based gui agent, 2024.

[13] Zhengxi Lu, Yuxiang Chai, Yaxuan Guo, Xi Yin, Liang Liu, Hao Wang, Han Xiao, Shuai Ren, Guanjing Xiong, and Hongsheng Li. Ui-r1: Enhancing action prediction of gui agents by reinforcement learning, 2025.

<!-- Meanless: 10 -->


[14] Run Luo, Lu Wang, Wanwei He, and Xiaobo Xia. Gui-r1 : A generalist r1-style vision-language action model for gui agents, 2025.

[15] Reiichiro Nakano, Jacob Hilton, Suchir Balaji, Jeff Wu, Long Ouyang, Christina Kim, Christopher Hesse, Shantanu Jain, Vineet Kosaraju, William Saunders, Xu Jiang, Karl Cobbe, Tyna Eloundou, Gretchen Krueger, Kevin Button, Matthew Knight, Benjamin Chess, and John Schulman. Webgpt: Browser-assisted question-answering with human feedback, 2022.

[16] Dang Nguyen, Jian Chen, Yu Wang, Gang Wu, Namyong Park, Zhengmian Hu, Hanjia Lyu, Junda Wu, Ryan Aponte, Yu Xia, Xintong Li, Jing Shi, Hongjie Chen, Viet Dac Lai, Zhouhang Xie, Sungchul Kim, Ruiyi Zhang, Tong Yu, Mehrab Tanjim, Nesreen K. Ahmed, Puneet Mathur, Seunghyun Yoon, Lina Yao, Branislav Kveton, Thien Huu Nguyen, Trung Bui, Tianyi Zhou, Ryan A. Rossi, and Franck Dernoncourt. Gui agents: A survey, 2024.

[17] Joon Sung Park, Joseph C. O'Brien, Carrie J. Cai, Meredith Ringel Morris, Percy Liang, and Michael S. Bernstein. Generative agents: Interactive simulacra of human behavior, 2023.

[18] Chen Qian, Jiahao Li, Yufan Dang, Wei Liu, YiFei Wang, Zihao Xie, Weize Chen, Cheng Yang, Yingli Zhang, Zhiyuan Liu, and Maosong Sun. Iterative experience refinement of software-developing agents, 2024.

[19] Chen Qian, Wei Liu, Hongzhang Liu, Nuo Chen, Yufan Dang, Jiahao Li, Cheng Yang, Weize Chen, Yusheng Su, Xin Cong, Juyuan Xu, Dahai Li, Zhiyuan Liu, and Maosong Sun. ChatDev: Communicative agents for software development. In Lun-Wei Ku, Andre Martins, and Vivek Srikumar, editors, Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 15174-15186, Bangkok, Thailand, August 2024. Association for Computational Linguistics.

[20] Yujia Qin, Yining Ye, Junjie Fang, Haoming Wang, Shihao Liang, Shizuo Tian, Junda Zhang, Jiahao Li, Yunxin Li, Shijue Huang, Wanjun Zhong, Kuanye Li, Jiale Yang, Yu Miao, Woyu Lin, Longxiang Liu, Xu Jiang, Qianli Ma, Jingyu Li, Xiaojun Xiao, Kai Cai, Chuang Li, Yaowei Zheng, Chaolin Jin, Chen Li, Xiao Zhou, Minchao Wang, Haoli Chen, Zhaojian Li, Haihua Yang, Haifeng Liu, Feng Lin, Tao Peng, Xin Liu, and Guang Shi. Ui-tars: Pioneering automated gui interaction with native agents, 2025.

[21] Qwen, :, An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, Huan Lin, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jingren Zhou, Junyang Lin, Kai Dang, Keming Lu, Keqin Bao, Kexin Yang, Le Yu, Mei Li, Mingfeng Xue, Pei Zhang, Qin Zhu, Rui Men, Runji Lin, Tianhao Li, Tianyi Tang, Tingyu Xia, Xingzhang Ren, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yu Wan, Yuqiong Liu, Zeyu Cui, Zhenru Zhang, and Zihan Qiu. Qwen2.5 technical report, 2025.

[22] Timo Schick, Jane Dwivedi-Yu, Roberto Dessì, Roberta Raileanu, Maria Lomeli, Luke Zettlemoyer, Nicola Cancedda, and Thomas Scialom. Toolformer: Language models can teach themselves to use tools, 2023.

[23] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, Aurelien Rodriguez, Armand Joulin, Edouard Grave, and Guillaume Lample. Llama: Open and efficient foundation language models, 2023.

<!-- Meanless: 11 -->


[24] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, Dan Bikel, Lukas Blecher, Cristian Canton Ferrer, Moya Chen, Guillem Cucurull, David Esiobu, Jude Fernandes, Jeremy Fu, Wenyin Fu, Brian Fuller, Cynthia Gao, Vedanuj Goswami, Naman Goyal, Anthony Hartshorn, Saghar Hosseini, Rui Hou, Hakan Inan, Marcin Kardas, Viktor Kerkez, Madian Khabsa, Isabel Kloumann, Artem Korenev, Punit Singh Koura, Marie-Anne Lachaux, Thibaut Lavril, Jenya Lee, Diana Liskovich, Yinghai Lu, Yuning Mao, Xavier Martinet, Todor Mihaylov, Pushkar Mishra, Igor Molybog, Yixin Nie, Andrew Poulton, Jeremy Reizenstein, Rashi Rungta, Kalyan Saladi, Alan Schelten, Ruan Silva, Eric Michael Smith, Ranjan Subramanian, Xiaoqing Ellen Tan, Binh Tang, Ross Taylor, Adina Williams, Jian Xiang Kuan, Puxin Xu, Zheng Yan, Iliyan Zarov, Yuchen Zhang, Angela Fan, Melanie Kambadur, Sharan Narang, Aurelien Rodriguez, Robert Stojnic, Sergey Edunov, and Thomas Scialom. Llama 2: Open foundation and fine-tuned chat models, 2023.

[25] Junyang Wang, Haiyang Xu, Haitao Jia, Xi Zhang, Ming Yan, Weizhou Shen, Ji Zhang, Fei Huang, and Jitao Sang. Mobile-agent-v2: Mobile device operation assistant with effective navigation via multi-agent collaboration, 2024.

[26] Lei Wang, Chen Ma, Xueyang Feng, Zeyu Zhang, Hao Yang, Jingsen Zhang, Zhiyuan Chen, Jiakai Tang, Xu Chen, Yankai Lin, Wayne Xin Zhao, Zhewei Wei, and Jirong Wen. A survey on large language model based autonomous agents. Frontiers of Computer Science, 18(6), March 2024.

[27] Taiyi Wang, Zhihao Wu, Jianheng Liu, Jianye Hao, Jun Wang, and Kun Shao. Distrl: An asynchronous distributed reinforcement learning framework for on-device control agents, 2025.

[28] Ziwei Wang, Weizhi Chen, Leyang Yang, Sheng Zhou, Shengchu Zhao, Hanbei Zhan, Jiongchao Jin, Liangcheng Li, Zirui Shao, and Jiajun Bu. Mp-gui: Modality perception with mllms for gui understanding, 2025.

[29] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed Chi, Quoc Le, and Denny Zhou. Chain-of-thought prompting elicits reasoning in large language models, 2023.

[30] Gokul Yenduri, Ramalingam M, Chemmalar Selvi G, Supriya Y, Gautam Srivastava, Praveen Kumar Reddy Maddikunta, Deepti Raj G, Rutvij H Jhaveri, Prabadevi B, Weizheng Wang, Athanasios V. Vasilakos, and Thippa Reddy Gadekallu. Generative pre-trained transformer: A comprehensive review on enabling technologies, potential applications, emerging challenges, and future directions, 2023.

[31] Andy Zeng, Maria Attarian, Brian Ichter, Krzysztof Choromanski, Adrian Wong, Stefan Welker, Federico Tombari, Aveek Purohit, Michael Ryoo, Vikas Sindhwani, Johnny Lee, Vincent Vanhoucke, and Pete Florence. Socratic models: Composing zero-shot multimodal reasoning with language, 2022.

<!-- Meanless: 12 -->


[32] Yuexiang Zhai, Hao Bai, Zipeng Lin, Jiayi Pan, Shengbang Tong, Yifei Zhou, Alane Suhr, Saining Xie, Yann LeCun, Yi Ma, and Sergey Levine. Fine-tuning large vision-language models as decision-making agents via reinforcement learning, 2024.

[33] Bofei Zhang, Zirui Shang, Zhi Gao, Wang Zhang, Rui Xie, Xiaojian Ma, Tao Yuan, Xinxiao Wu, Song-Chun Zhu, and Qing Li. Tongui: Building generalized gui agents by learning from multimodal web tutorials, 2025.

[34] Chaoyun Zhang, He Huang, Chiming Ni, Jian Mu, Si Qin, Shilin He, Lu Wang, Fangkai Yang, Pu Zhao, Chao Du, Liqun Li, Yu Kang, Zhao Jiang, Suzhen Zheng, Rujia Wang, Jiaxu Qian, Minghua Ma, Jian-Guang Lou, Qingwei Lin, Saravan Rajmohan, and Dongmei Zhang. Ufo2: The desktop agentos, 2025.

[35] Chi Zhang, Zhao Yang, Jiaxuan Liu, Yucheng Han, Xin Chen, Zebiao Huang, Bin Fu, and Gang Yu. Appagent: Multimodal agents as smartphone users, 2023.

[36] J. Zhang, Y. Yu, M. Liao, W. Li, J. Wu, and Z. Wei. Ui-hawk: Unleashing the screen stream understanding for gui agents. Preprints, 2024.

[37] Shaoqing Zhang, Zhuosheng Zhang, Kehai Chen, Xinbei Ma, Muyun Yang, Tiejun Zhao, and Min Zhang. Dynamic planning for llm-based graphical user interface automation, 2024.

[38] Zhuosheng Zhang and Aston Zhang. You only look at screens: Multimodal chain-of-action agents, 2024.

[39] Boyuan Zheng, Boyu Gou, Jihyung Kil, Huan Sun, and Yu Su. Gpt-4v(ision) is a generalist web agent, if grounded, 2024.

[40] Jiani Zheng, Lu Wang, Fangkai Yang, Chaoyun Zhang, Lingrui Mei, Wenjie Yin, Qing-wei Lin, Dongmei Zhang, Saravan Rajmohan, and Qi Zhang. Vem: Environment-free exploration for training gui agent with value environment model, 2025.

<!-- Meanless: 13 -->