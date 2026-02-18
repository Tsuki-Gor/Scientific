# RealWebAssist: A Benchmark for Long-Horizon Web Assistance with Real-World Users
# RealWebAssist：面向真实用户的长程网络辅助基准


Suyu Ye ${}^{*1}$ , Haojun Shi ${}^{*1}$ , Darren Shih ${}^{1}$ , Hyokun Yun ${}^{2}$ , Tanya G. Roosta ${}^{2}$ , Tianmin Shu ${}^{1}$
Suyu Ye ${}^{*1}$ , Haojun Shi ${}^{*1}$ , Darren Shih ${}^{1}$ , Hyokun Yun ${}^{2}$ , Tanya G. Roosta ${}^{2}$ , Tianmin Shu ${}^{1}$


${}^{1}$ Johns Hopkins University,
${}^{1}$ 约翰霍普金斯大学，


${}^{2}$ Amazon.com
${}^{2}$ 亚马逊 (Amazon.com)


\{sye10, hshi33, dshih5, tianmin.shu\}@jhu.edu, \{yunhyoku,troosta\}@amazon.com
\{sye10, hshi33, dshih5, tianmin.shu\}@jhu.edu, \{yunhyoku,troosta\}@amazon.com


## Abstract
## 摘要


To achieve successful assistance with long-horizon web-based tasks, AI agents must be able to sequentially follow real-world user instructions over a long period. Unlike existing web-based agent benchmarks, sequential instruction following in the real world poses significant challenges beyond performing a single, clearly defined task. For instance, real-world human instructions can be ambiguous, require different levels of AI assistance, and may evolve over time, reflecting changes in the user's mental state. To address this gap, we introduce RealWebAssist, a novel benchmark designed to evaluate sequential instruction-following in realistic scenarios involving long-horizon interactions with the web, visual GUI grounding, and understanding ambiguous real-world user instructions. RealWebAssist includes a dataset of sequential instructions collected from real-world human users. Each user instructs a web-based assistant to perform a series of tasks on multiple websites. A successful agent must reason about the true intent behind each instruction, keep track of the mental state of the user, understand user-specific routines, and ground the intended tasks to actions on the correct GUI elements. Our experimental results show that state-of-the-art models struggle to understand and ground user instructions, posing critical challenges in following real-world user instructions for long-horizon web assistance.
为了成功辅助完成长程网络任务，AI 智能体必须能够长期连续遵循真实用户的指令。与现有的网络智能体基准不同，现实世界中的序列指令遵循面临着超出执行单一、明确任务的重大挑战。例如，真实的指令可能具有模糊性，需要不同程度的辅助，并随时间演变以反映用户心理状态的变化。为填补这一空白，我们推出了 RealWebAssist，这是一个旨在评估现实场景下序列指令遵循的新型基准，涉及长程网络交互、视觉 GUI 定位以及对模糊指令的理解。RealWebAssist 包含从真实用户收集的序列指令数据集。每位用户指挥一个网络助手在多个网站上执行一系列任务。成功的智能体必须推理指令背后的真实意图，跟踪用户的心理状态，理解用户特定习惯，并将预期任务落实到正确的 GUI 元素操作上。实验结果表明，最先进的模型在理解和定位用户指令方面仍面临巨大困难，为实现长程网络辅助提出了严峻挑战。


## Introduction
## 引言


As an integral part of people's daily life, many of our everyday tasks are performed on the internet. With the tremendous advances in open-ended agents driven by large reasoning models (LRMs) and vision-language models (VLMs), there has been increasing interest in engineering web-based agents that can assist humans with complex tasks on the web following humans' instructions (Zheng et al. 2024a; Nakano et al. 2022). Recent works have demonstrated the promising performance of web-based agents on planning (Putta et al. 2024; Wang et al. 2024; Yao et al. 2023) and Graphical User Interface (GUI) grounding (Cheng et al. 2024; Wu et al. 2024b; Gou et al. 2024; Yang et al. 2024; Xu et al. 2024), across diverse websites, tasks, and GUI interfaces.
作为日常生活不可或缺的一部分，我们的许多日常任务都在互联网上完成。随着大型推理模型 (LRM) 和视觉语言模型 (VLM) 驱动的开放式智能体的飞速发展，人们对开发能够根据指令辅助人类处理复杂网络任务的智能体产生了浓厚兴趣 (Zheng et al. 2024a; Nakano et al. 2022)。近期研究已证明，网络智能体在跨网站、跨任务及跨 GUI 界面的规划 (Putta et al. 2024; Wang et al. 2024; Yao et al. 2023) 与图形用户界面 (GUI) 定位 (Cheng et al. 2024; Wu et al. 2024b; Gou et al. 2024; Yang et al. 2024; Xu et al. 2024) 方面表现出良好的性能。


Despite these encouraging results, there have not been systematic studies on long-horizon web assistance with real-world users. Existing benchmarks (e.g., (Zhou et al. 2023; Deng et al. 2024; Cheng et al. 2024; Yao et al. 2022; Jang et al. 2024)) typically focus on performing a task based on a single instruction. Additionally, the instructions in the current benchmarks were not collected from real users during natural web use sessions, lacking the realism of real user instructions. As a result, these benchmarks do not capture the full complexity of real users' web behavior and instructions.
尽管取得了这些令人鼓舞的成果，但目前尚未对面向真实用户的长程网络辅助进行系统研究。现有的基准（例如 (Zhou et al. 2023; Deng et al. 2024; Cheng et al. 2024; Yao et al. 2022; Jang et al. 2024)）通常侧重于根据单一指令执行任务。此外，当前基准中的指令并非从真实的自然网络使用会话中收集，缺乏真实指令的真实感。因此，这些基准无法捕捉真实用户网络行为和指令的完整复杂性。


To bridge this gap, we propose RealWebAssist, the first sequential instruction following benchmark that evaluates long-horizon web assistance with real-world users. As illustrated in Figure 1, to perform a task, a user will instruct an AI assistant in a long sequence. Based on the past instructions and screenshots, the AI assistant must execute one or a few steps of actions to perform the latest instruction. Additionally, a user can engage in repeated interactions over a series of tasks with the assistant in a long session up to 40 minutes. To construct RealWebAssist, we recruited real users to instruct an assistant to perform multiple real-world tasks on the web. We created a large dataset with real user instructions (in both speech and text) for diverse real-world tasks and websites (as shown in Figure 2).
为缩小这一差距，我们提出了 RealWebAssist，这是首个评估面向真实用户的长程网络辅助的序列指令遵循基准。如图 1 所示，为了完成任务，用户会以长序列指令指挥 AI 助手。基于过去的指令和截图，助手必须执行一个或几个动作步骤来完成最新指令。此外，用户可以在长达 40 分钟的长会话中与助手就一系列任务进行多次交互。为构建 RealWebAssist，我们招募了真实用户来指挥助手在网上执行多项现实任务，并创建了一个包含针对多样化任务和网站的真实指令（语音和文本形式）的大型数据集（如图 2 所示）。


The sequential instruction following tasks in our RealWe-bAssist benchmark reflect the natural human behavior on the web. First, real-world users may not initially know what they are looking for. Thus, they need to engage in information seeking on multiple web pages (e.g., step 1-2 in Figure 1), sometimes even across websites. Second, based on new information such as product reviews, users may change their minds (e.g., step 3). Third, users give simple instructions that are seemingly ambiguous out of the context but could be interpreted based on spatial and temporal context via pragmatic reasoning (Goodman and Frank 2016; Fried et al. 2023). For instance, the third instruction in Figure 1 does not explicitly describe which product, but an intelligent assistant should be able to infer the true user intent and correctly select the product in the user's mind. Lastly, in our benchmark, users can browse the websites and have the autonomy to make critical decisions (such as purchasing) on their own, which is complementary to existing benchmarks that focus on agents' planning ability to fully complete the tasks without human involvement.
我们的 RealWebAssist 基准测试中的顺序指令遵循任务反映了网页上的自然人类行为。首先，现实用户最初可能并不清楚自己的目标。因此，他们需要在多个网页（如附图 1 中的步骤 1-2），甚至跨网站进行信息搜索。其次，根据产品评论等新信息，用户可能会改变主意（如步骤 3）。第三，用户给出的简单指令脱离语境时看似模糊，但可以通过语用推理（Goodman and Frank 2016; Fried et al. 2023）基于时空语境进行解读。例如，图 1 中的第三条指令并未明确说明是哪款产品，但智能助手应能推断出用户的真实意图并准确选择。最后，在我们的基准测试中，用户可以浏览网站并自主做出关键决策（如购买），这与现有关注智能体在无人类参与下完全胜任任务规划能力的基准测试形成了互补。


We systematically evaluate state-of-the-art models, including GUI grounding, VLMs, and large reasoning models. Experimental results reveal that these models lack several key abilities, including grounding, understanding user intents, reasoning about spatial and temporal context, and adapting to user-specific routines.
我们系统地评估了包括 GUI 定位、VLM 和大型推理模型在内的尖端模型。实验结果表明，这些模型缺乏几项关键能力，包括定位、理解用户意图、对时空语境进行推理以及适应用户特定的习惯。


---



*These authors contributed equally.
*这些作者贡献相同。


Copyright © 2026, Association for the Advancement of Artificial Intelligence (www.aaai.org). All rights reserved.
版权所有 © 2026，人工智能促进协会 (www.aaai.org)。保留所有权利。


---



<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/02/2026_02_18__00_22_27_0f643b.jpg"/>



Figure 1: An example sequential instruction following task with a real-world user. The red circles indicate the correct actions based on the user's spoken instructions. Sequential instructions introduce unique challenges, such as the need to retain and reason over past context. For instance, the instruction in step 3 requires information from step 1 to be correctly interpreted.
图 1：现实用户参与的顺序指令遵循任务示例。红圈表示基于用户口头指令的正确动作。顺序指令引入了独特的挑战，例如需要保留并对过去的语境进行推理。例如，步骤 3 中的指令需要步骤 1 的信息才能被正确解读。


<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/02/2026_02_18__00_22_27_d98c4a.jpg"/>



Figure 2: Examples of general task categories (left) and websites visited (right) in RealWebAssist. The tasks span a wide range of real-world scenarios, from shopping to food & entertainment to travel planning, which encourages users to visit many different websites.
图 2：RealWebAssist 中的通用任务类别（左）和访问网站（右）示例。任务涵盖了从购物、餐饮娱乐到旅行规划等广泛的现实场景，促使用户访问许多不同的网站。


## Related Works
## 相关工作


Web Agent Benchmarks. Existing web agent benchmarks primarily evaluate the performance of web agents on tasks with clearly defined, unambiguous instructions, often overlooking the complexities of real-world users' behavior and their instructions to an AI assistant. On WebArena (Zhou et al. 2023), Mind2Web (Deng et al. 2024), and WebShop (Yao et al. 2022), an agent follows a single instruction to perform an isolated task. While they offer an evaluation of an agent's planning capacity, they lack the evaluation of an agent's ability to follow a long sequence of user instructions on long-horizon web tasks. There have also been GUI grounding benchmarks, such as ScreenSpot (Cheng et al. 2024), that focused on grounding simple instructions to clicking actions on webpages. These instructions only instruct web agents to click web elements rather than reaching a user goal (e.g., purchasing an item). WebLINX (Lù, Kasner, and Reddy 2024) features sequential instruction following. However, the instructions were generated by annotators who received detailed guidelines and extensive training, rather than by actual users. The resulting instructions do not capture the nuances and complexity of real-world user instructions that naturally emerge in interactions with an assistant. In contrast, RealWebAssist consists of sequential instruction following tasks for assisting real-world users, providing a novel set of challenges necessary for long-horizon web assistance for real-world users. Table 1 summarizes key differences between RealWebAssist and prior benchmarks.
网页智能体基准测试。现有的网页智能体基准测试主要评估智能体在定义明确、无歧义指令下的任务表现，往往忽视了现实用户行为及其对 AI 助手指令的复杂性。在 WebArena (Zhou et al. 2023)、Mind2Web (Deng et al. 2024) 和 WebShop (Yao et al. 2022) 中，智能体遵循单一指令执行孤立任务。虽然它们评估了智能体的规划能力，但缺乏对智能体在长程网页任务中遵循长序列用户指令能力的评估。此外还有如 ScreenSpot (Cheng et al. 2024) 等 GUI 定位基准测试，侧重于将简单指令定位为网页上的点击操作。这些指令仅指挥智能体点击网页元素，而非实现用户目标（如购买物品）。WebLINX (Lù, Kasner, and Reddy 2024) 虽然包含顺序指令遵循，但指令是由接受过详细指导和广泛培训的标注员生成的，而非真实用户。由此产生的指令未能捕捉到与助手交互时自然产生的现实用户指令的细微差别和复杂性。相比之下，RealWebAssist 包含协助现实用户的顺序指令遵循任务，为实现现实用户的长程网页协助提供了必要的新挑战。表 1 总结了 RealWebAssist 与先前基准测试的主要区别。


Autonomous Web Agents. There have been many recent works on engineering autonomous web agents through retrieval augmented planning (Kim et al. 2024; Zhou et al. 2024; Wu et al. 2024a; He et al. 2024; Pan et al. 2024), finetuning (Hong et al. 2024; Gur et al. 2024; Deng et al. 2024; Pang et al. 2024; Zhang and Zhang 2024), learning workflows (Zhang et al. 2023; Wang et al. 2024; Zheng et al. 2024b; Majumder et al. 2023; Cai et al. 2024), reinforcement learning (Liu et al. 2018; Shi et al. 2017; Nogueira and Cho 2016; Humphreys et al. 2022), and combinations of these methods (Liu et al. 2023; Putta et al. 2024). These works focus on planning for a single task. However, there has not been much work on understanding and following real-world users' sequential instructions on long-horizon tasks.
自主网页智能体。近期有许多工作通过检索增强规划 (Kim et al. 2024; Zhou et al. 2024; Wu et al. 2024a; He et al. 2024; Pan et al. 2024)、微调 (Hong et al. 2024; Gur et al. 2024; Deng et al. 2024; Pang et al. 2024; Zhang and Zhang 2024)、学习工作流 (Zhang et al. 2023; Wang et al. 2024; Zheng et al. 2024b; Majumder et al. 2023; Cai et al. 2024)、强化学习 (Liu et al. 2018; Shi et al. 2017; Nogueira and Cho 2016; Humphreys et al. 2022) 及其组合方法 (Liu et al. 2023; Putta et al. 2024) 来构建自主网页智能体。这些工作专注于单一任务的规划。然而，关于理解并遵循现实用户在长程任务中的顺序指令的研究仍然较少。


GUI Grounding. One key ability for web agents in many assistance tasks is to ground instructions to clicking actions on a webpage. Recent works have explored VLM finetuning (e.g., (Gou et al. 2024; Wu et al. 2024b; Yang et al. 2024, 2025; Wu et al. 2025; Qin et al. 2025; Xu et al. 2025; Yuan et al. 2025)) as well as prompting pretrained VLMs with segmentations of web elements (e.g., (Yang et al. 2023)) for enabling GUI grounding. These methods generate coordinates or bounding boxes on webpages to indicate where to click.
图形用户界面（GUI）定位。在许多辅助任务中，网络智能体的一项关键能力是将指令转化为网页上的点击操作。近期的研究探索了视觉语言模型（VLM）微调（例如（Gou等人，2024年；Wu等人，2024b；Yang等人，2024年、2025年；Wu等人，2025年；Qin等人，2025年；Xu等人，2025年；Yuan等人，2025年）），以及使用网页元素分割来提示预训练的视觉语言模型（例如（Yang等人，2023年））以实现图形用户界面定位。这些方法会在网页上生成坐标或边界框，以指示点击位置。


<table><tr><td>Benchmark</td><td>Real User</td><td>Sequential Instructions</td><td>Real Websites</td><td>$\mathbf{{GUI}}$ Grounding</td><td>Speech</td><td># Instructions</td></tr><tr><td>SreenSpot (Cheng et al. 2024)</td><td>✘</td><td>✘</td><td>✓</td><td>✓</td><td>✘</td><td>1200+</td></tr><tr><td>WebArena (Zhou et al. 2023)</td><td>✘</td><td>✘</td><td>✘</td><td>✘</td><td>✘</td><td>812</td></tr><tr><td>Mind2Web (Deng et al. 2024)</td><td>✘</td><td>✘</td><td>✓</td><td>✘</td><td>✘</td><td>2000+</td></tr><tr><td>WebLINX (Lù, Kasner, and Reddy 2024)</td><td>✘</td><td>✓</td><td>✓</td><td>✘</td><td>✘</td><td>512</td></tr><tr><td>VideoWebArena (Jang et al. 2024)</td><td>✘</td><td>✘</td><td>✘</td><td>✘</td><td>✓</td><td>2021</td></tr><tr><td>WebShop (Yao et al. 2022)</td><td>✘</td><td>✘</td><td>✘</td><td>✘</td><td>✘</td><td>12087</td></tr><tr><td>BearCubs (Song et al. 2025)</td><td>✘</td><td>✘</td><td>✓</td><td>✘</td><td>✘</td><td>111</td></tr><tr><td>RealWebAssist (Ours)</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td>1885</td></tr></table>
<table><tbody><tr><td>基准测试</td><td>真实用户</td><td>序列指令</td><td>真实网站</td><td>$\mathbf{{GUI}}$ 地标定位</td><td>语音</td><td># 指令数</td></tr><tr><td>SreenSpot (Cheng et al. 2024)</td><td>✘</td><td>✘</td><td>✓</td><td>✓</td><td>✘</td><td>1200+</td></tr><tr><td>WebArena (Zhou et al. 2023)</td><td>✘</td><td>✘</td><td>✘</td><td>✘</td><td>✘</td><td>812</td></tr><tr><td>Mind2Web (Deng et al. 2024)</td><td>✘</td><td>✘</td><td>✓</td><td>✘</td><td>✘</td><td>2000+</td></tr><tr><td>WebLINX (Lù, Kasner, and Reddy 2024)</td><td>✘</td><td>✓</td><td>✓</td><td>✘</td><td>✘</td><td>512</td></tr><tr><td>VideoWebArena (Jang et al. 2024)</td><td>✘</td><td>✘</td><td>✘</td><td>✘</td><td>✓</td><td>2021</td></tr><tr><td>WebShop (Yao et al. 2022)</td><td>✘</td><td>✘</td><td>✘</td><td>✘</td><td>✘</td><td>12087</td></tr><tr><td>BearCubs (Song et al. 2025)</td><td>✘</td><td>✘</td><td>✓</td><td>✘</td><td>✘</td><td>111</td></tr><tr><td>RealWebAssist (本研究)</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td>1885</td></tr></tbody></table>


Table 1: Comparison between RealWebAssist and existing web agent benchmarks on several key aspects: (1) whether instructions were given by real-world users instead of annotators, (2) whether there is a sequence of instructions, (3) whether there are real-world websites, (4) whether the agent needs to execute actions by selecting coordinates on webpages, (5) whether the instructions are speech instructions, and (6) the number of total instructions.
表 1：RealWebAssist 与现有网络智能体基准在几个关键方面的比较：(1) 指令是否由真实用户而非标注者提供，(2) 是否存在指令序列，(3) 是否包含真实世界的网站，(4) 智能体是否需要通过在网页上选择坐标来执行动作，(5) 指令是否为语音指令，以及 (6) 总指令数量。


<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/02/2026_02_18__00_22_27_9a8912.jpg"/>



Figure 3: Multiple actions can satisfy a user's intent. A web agent's action is considered correct if the coordinate they provide is within one of the annotated correct regions.
图 3：多个动作可以满足用户的意图。如果网络智能体提供的坐标位于标注的正解区域内，则其动作被视为正确。


They have only been trained on low-level instructions that clearly refer to web elements. It remains unclear if they can understand real-world user instructions that must be interpreted considering context or may refer to high-level goals.
它们仅在明确指代网页元素的低级指令上接受过训练。目前尚不清楚它们是否能理解必须结合上下文解释或可能指代高级目标的真实用户指令。


## RealWebAssist Benchmark
## RealWebAssist 基准测试


## Problem Setup
## 问题设置


RealWebAssist evaluates agents' ability to follow long-horizon, sequential web instructions to assist users with their high-level goals. In each task, a human user will try to reach an open-ended goal such as "buy formal outfits for a formal event" by instructing the assistant through a series of spoken instructions. The dataset is collected from interactions between human users and human assistants in a human experiment. To evaluate agents, we use the human assistants' actions to evaluate the agents' success.
RealWebAssist 评估智能体遵循长程、序列化网络指令以协助用户实现高级目标的能力。在每个任务中，人类用户将尝试通过一系列语音指令指导助手来实现一个开放式目标，例如“为正式场合购买礼服”。该数据集采集自人类实验中人类用户与人类助手之间的交互。在评估智能体时，我们使用人类助手的动作来衡量智能体的成功程度。


In RealWebAssist, a web agent has access to the current instruction, webpage (as a screenshot), and all the past interactions (previous instructions & screenshots of webpages). Since we are focusing on tasks on real-world websites, it is challenging to ensure safety and reproducibility in an interactive evaluation setting. Therefore, we adopt an offline evaluation setting following prior web-based agent benchmarks with real websites (Deng et al. 2024; Cheng et al. 2024). Specifically, for each instruction collected from the human experiment, the agent needs to identify the correct element to interact with by providing a coordinate or a bounding box to click on the webpage. As shown by figure 3, a web agent's action is considered correct if the coordinate or the center of the bounding box they provide falls in the annotated correct regions on the webpage. If there are multiple steps corresponding to one instruction, we evaluate if the web agent's actions for the same instruction are all correct.
在 RealWebAssist 中，网络智能体可以访问当前指令、网页（以截图形式）以及所有过往交互（之前的指令和网页截图）。由于我们专注于真实网站上的任务，在交互式评估设置中确保安全性和可复现性具有挑战性。因此，我们遵循先前基于真实网站的网络智能体基准（Deng et al. 2024; Cheng et al. 2024），采用了离线评估设置。具体而言，对于从人类实验中收集的每条指令，智能体需要通过提供坐标或边界框来点击网页，从而识别出正确的交互元素。如图 3 所示，如果智能体提供的坐标或边界框中心位于网页上标注的正解区域内，则认为其动作正确。如果一条指令对应多个步骤，我们会评估智能体针对该指令的所有动作是否全部正确。


## Evaluation Metrics
## 评估指标


We consider the following evaluation metrics:
我们考虑以下评估指标：


- Task success rate: A task is successful if the web agent can correctly produce actions for all instructions in a task.
- 任务成功率：如果网络智能体能正确执行一个任务中所有指令的动作，则该任务成功。


- Average progress: We measure the progress of a task by the percentage of consecutive instructions the web agent can successfully perform before its first error in the task.
- 平均进度：我们通过网络智能体在任务中出现首次错误前，能够成功执行的连续指令百分比来衡量任务进度。


- Step success rate: We also consider a teacher forcing setting as a simpler, diagnostic evaluation, where the web agent will only need to follow the instruction at a single step of a task assuming all previous instructions have been successfully performed.
- 步骤成功率：我们还考虑将教师强制（teacher forcing）设置作为一种更简单的诊断性评估，即假设之前的所有指令都已成功执行，网络智能体仅需遵循任务中单个步骤的指令。


## Dataset Construction
## 数据集构建


Setup. We recruited 10 participants (4 female, 6 male, mean age = 20 years) from a US university campus, none of whom had prior knowledge of the study's purpose, to construct the dataset. All participants were native or fluent English speakers. Each participant completed a 40-minute real-world web assistance session in which they tackled a series of open-ended tasks designed to encourage diverse strategies. During each session, participants verbally instructed an experimenter, who operated the computer on their behalf, to complete the tasks. We captured screen recordings and used a high-quality USB microphone to record speech as raw data. The user study was approved by an institutional review board.
设置。我们从美国一所大学校园招募了 10 名参与者（4 名女性，6 名男性，平均年龄 20 岁）来构建数据集，所有人事先均不知道研究目的。所有参与者均为英语母语或流利者。每位参与者完成了一场 40 分钟的真实网络协助会话，期间处理了一系列旨在鼓励多样化策略的开放式任务。在每场会话中，参与者口头指示一名实验人员代表他们操作电脑以完成任务。我们捕捉了屏幕录像，并使用高质量 USB 麦克风录制语音作为原始数据。该用户研究已获得机构审查委员会的批准。


User Tasks. To increase the instruction diversity and realism, participants received general web-based tasks requiring active information seeking, sub-goal planning, and comparison among various options. We generated the task list by few-shot prompting GPT-40 with open-ended tasks, followed by manual filtering and editing to ensure task quality and feasibility. These tasks provide only general guidance, ensuring flexibility for personal decision-making. Example tasks include "Purchase an outfit for a formal event" and "Plan a 5-day trip to Japan, booking both flights and hotels". Each user finishes about 10 tasks.
用户任务。为了增加指令的多样性和真实性，参与者接受了通用的网络任务，这些任务需要主动寻找信息、划分子目标计划以及在各种选项之间进行比较。我们通过少样本提示词让 GPT-40 生成开放式任务列表，随后进行人工过滤和编辑，以确保任务质量和可行性。这些任务仅提供一般性指导，确保了个人决策的灵活性。示例任务包括“为正式场合购买一套衣服”和“计划一次为期 5 天的日本旅行，预订机票和酒店”。每位用户完成约 10 个任务。


Emergent User Behavior. In our realistic, open-ended settings, users exhibit rich behaviors that are not present in previous benchmarks. These include, but are not limited to, information seeking, researching and comparing different options, change of mind, and trial-and-error.
涌现的用户行为。在我们的真实开放式设置中，用户表现出丰富的行为，这些行为在以往的基准测试中并不存在。这些行为包括但不限于信息搜寻、研究和比较不同选项、改变主意以及反复试验。


Annotations. We manually labeled RealWebAssist data to ensure high-quality annotations. We first segmented the full recording into individual clips corresponding to each user's instructions. In our benchmark, we disregard user speech unrelated to explicit instructions for the assistant, such as filler words or verbalized thought processes. For each instruction, we provide raw speech, speech transcript, webpage, and the correct regions to click (in the form of one or more bounding boxes). When there were multiple correct answers for the instructions (for instance, "can you close all the current tabs"), we annotated all correct regions with multiple bounding boxes. When the experimenter made a mistake during the data collection sessions, we annotated the correct action intended by the user. If an instruction required multiple steps to complete, we set the instruction at each step as the same instruction. To generate the text instructions, we used an off-the-shelf recognition model, Whisper Large-V3 (Radford et al. 2023), to transcribe users' speech and then manually fixed transcription errors. For all the instructions, we have three annotators verifying all of them, ensuring 100% agreement.
标注。我们手动标注了 RealWebAssist 数据以确保高质量。我们首先将完整录制内容分割成与每条用户指令相对应的独立片段。在基准测试中，我们忽略了与助理明确指令无关的用户语音，例如语气词或口头化的思考过程。对于每条指令，我们提供原始语音、语音转录文本、网页以及正确的点击区域（以一个或多个边界框的形式）。当指令存在多个正确答案时（例如“你能关闭当前所有标签页吗”），我们用多个边界框标注了所有正确区域。当实验人员在数据采集过程中出现错误时，我们标注了用户意图进行的正确操作。如果一条指令需要多个步骤完成，我们将每一步的指令设定为同一条指令。为了生成文本指令，我们使用现成的识别模型 Whisper Large-V3 (Radford et al. 2023) 转录用户语音，并手动修正了转录错误。所有指令均由三名标注员进行验证，确保 100% 的一致性。


Dataset Statistics. RealWebAssist contains 1,885 user instructions across 107 tasks, 66 websites, and 2,524 screen-shots. In addition to the benchmark, we also plan to release the raw data, consisting of over 6 hours of video & audio.
数据集统计。RealWebAssist 包含分布在 107 个任务、66 个网站和 2,524 张屏幕截图中的 1,885 条用户指令。除基准测试外，我们还计划发布包含超过 6 小时视频和音频的原始数据。


## Key Challenges
## 核心挑战


RealWebAssist features multiple challenges as illustrated in Figure 4, including spatial and temporal reasoning needed to understand ambiguous and context-dependent user instructions, planning for multiple steps of actions to reach the goal communicated by an instruction, and learning about user-specific routines. These key challenges provide a more realistic and holistic evaluation of a web agent's reasoning, planning, and learning abilities to assist real-world users on long-horizon tasks. It is worth noting that many of these challenges, in particular, spatial reasoning, temporal reasoning, and routine understanding, are not present in existing web agent benchmarks. Unlike RealWebAssist, prior benchmarks, such as ScreenSpot (Cheng et al. 2024), WebArena (Zhou et al. 2023), and Mind2Web (Deng et al. 2024), only include clear, unambiguous, and non-sequential instructions.
如图 4 所示，RealWebAssist 包含多重挑战，包括理解模糊且依赖上下文的用户指令所需的空间和时间推理、为实现指令传达的目标而进行的多步动作规划，以及学习用户特定的惯例。这些核心挑战为评估网络智能体在长程任务中协助真实用户的推理、规划和学习能力提供了更真实、更全面的衡量标准。值得注意的是，其中许多挑战，特别是空间推理、时间推理和惯例理解，在现有的网络智能体基准测试中并不存在。与 RealWebAssist 不同，之前的基准测试（如 ScreenSpot (Cheng et al. 2024)、WebArena (Zhou et al. 2023) 和 Mind2Web (Deng et al. 2024)）仅包含清晰、明确且非连续的指令。


Spatial Reasoning. When referring to one of the elements on a webpage, real-world users tend to use a concise instruction that can be understood conditioned on spatial context instead of an overly elaborated instruction. For instance, when instructing an assistant to buy a product, users may give short instructions such as "select the cheapest one," instead of describing the desired product in detail. Figure 4A depicts different types of spatial reasoning that rely on diverse spatial contexts, including ranking, spatial relations, and overall website functionalities. It is worth noting that these instructions may sometimes reveal users' preferences (e.g., preferred seating), providing additional information for the web agent to provide potentially more customized assistance in the future.
空间推理。在提及网页上的某个元素时，真实用户倾向于使用基于空间上下文理解的简炼指令，而非过度详述。例如，在指示助理购买产品时，用户可能会给出诸如“选择最便宜的那个”之类的简短指令，而不是详细描述所需产品。图 4A 描绘了依赖于不同空间上下文（包括排序、空间关系和整体网站功能）的各类空间推理。值得注意的是，这些指令有时可能会揭示用户的偏好（例如，偏好的座位），从而为网络智能体在未来提供更个性化的协助提供额外信息。


Temporal Reasoning. In our sequential instruction following tasks, users may instruct an assistant with the history as an assumed temporal context. For example, to understand the intended meaning of "click the last item," the assistant must memorize the items the user has viewed in the past. Figure 4B shows temporal reasoning based on different kinds of temporal context, ranging from short context between two consecutive webpages to long context with the same website to long context across websites. From the temporal context, the assistant needs to memorize crucial elements in the previous webpages, infer and track a user's mind (e.g., change of mind about what to buy) based on the past instructions and webpages, and identify the earlier web-page the user refers to. Such temporal reasoning has not been evaluated in prior web agent benchmarks. However, it is very common in our benchmark due to the nature of human web browsing behavior as well as human instructions guided by pragmatics (Goodman and Frank 2016).
时间推理。在我们的连续指令遵循任务中，用户可能会以历史记录作为隐含的时间上下文来指示助理。例如，为了理解“点击最后一项”的意图，助理必须记住用户过去查看过的项目。图 4B 展示了基于不同时间上下文的时间推理，范围涵盖从两个连续网页之间的短上下文，到同一网站的长上下文，再到跨网站的长上下文。基于时间上下文，助理需要记住先前网页中的关键元素，根据过去的指令和网页推断并追踪用户的想法（例如，改变购买主意），并识别用户引用的早期网页。这种时间推理在之前的网络智能体基准测试中尚未得到评估。然而，由于人类网页浏览行为的特性以及受语用学引导的人类指令 (Goodman and Frank 2016)，它在我们的基准测试中非常普遍。


Multi-step Planning. Many instructions require multiple steps to complete. In these cases, the assistant needs to interpret the goal implied by the instruction and plan a sequence of actions to achieve that goal. This goes beyond grounding the instruction to a single action on the current webpage. Figure 4C shows an example where the agent was asked to repeat the same order on another food delivery website to check if the price would be different. A successful execution of this instruction would require the agent to first understand what the order is to ground the goal on the current website and generate a successful multi-step plan.
多步规划。许多指令需要多个步骤才能完成。在这种情况下，助理需要解读指令隐含的目标，并规划一系列动作来实现该目标。这超越了将指令定位到当前网页上单一动作的范畴。图 4C 展示了一个示例，其中代理被要求在另一个送餐网站上重复同样的订单，以检查价格是否有差异。成功执行此指令需要代理首先理解订单内容以在当前网站定位目标，并生成一个成功的多步计划。


Routine. Since our benchmark allows a user to engage in repeated interactions with an assistant over multiple tasks, we observe that users may define routines understood by the assistant after repeated interactions. As shown in Figure 4D, the user initially gave detailed step-by-step instructions when selecting arrival and departure dates for a flight. In a subsequent task, however, the user simplified them into a single instruction when selecting dates for a hotel room. Such shorter instructions become possible after establishing a routine in the earlier task. Cognitive studies found that procedural abstraction, like these routines, naturally emerges in human cooperative communication through repeated interactions, allowing more efficient communication with partners (McCarthy et al. 2021). The emergence of such routines in our benchmark poses a novel challenge for web agents-learning user-specific procedural abstraction via repeated interactions to achieve human-like adaptive assistance. We hypothesize that this ability could enhance users' perception of the AI assistant, as it understands human cooperative communication.
常规。由于我们的基准测试允许用户在多个任务中与助手进行重复交互，我们观察到用户可能会定义出助手在重复交互后能够理解的常规。如图 4D 所示，用户最初在选择航班的往返日期时给出了详细的分步指令。然而，在随后的任务中，用户在为酒店房间选择日期时，将其简化为了单条指令。在早期任务中建立常规后，这种更简短的指令便成为可能。认知研究发现，像这些常规一样的程序化抽象，会通过重复交互自然地出现在人类的协作交流中，从而提高与伙伴的沟通效率 (McCarthy et al. 2021)。在我们的基准测试中，这种常规的出现对网络代理提出了新的挑战——通过重复交互学习用户特定的程序化抽象，以实现类人的自适应辅助。我们假设这种能力可以增强用户对 AI 助手的感知，因为它理解人类的协作交流。


<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/02/2026_02_18__00_22_27_a182f9.jpg"/>



Figure 4: Key challenges introduced by RealWebAssist: (A) spatial reasoning, (B) temporal reasoning, (C) multi-step planning, and (D) learning user-specific routines.
图 4：RealWebAssist 引入的关键挑战：(A) 空间推理，(B) 时间推理，(C) 多步规划，以及 (D) 学习用户特定的常规。


## Experiments
## 实验


## Baselines
## 基准模型


We evaluated several types of models for web agents commonly evaluated in existing web agent benchmarks that have real-world websites (i.e., offline evaluation). For all the experiments, we use the ground-truth captions for instructions.
我们评估了几类在现有具有真实网站的网代理基准测试中常用的模型（即离线评估）。在所有实验中，我们使用指令的地面真值标题。


GUI Grounding Models. GUI grounding models directly translate an instruction to an action on a webpage. There are
GUI 定位模型。GUI 定位模型直接将指令转换为网页上的动作。存在


two general types of grounding models. First, Set-of-Mark (SoM) (Yang et al. 2023) segments salient elements on a webpage using an off-the-shelf segmentation model (e.g., SAM (Kirillov et al. 2023) and Semantic-SAM (Li et al. 2023)) and prompts a VLM to select a segment mask to identify the clicking area corresponding to the given instruction. Second, VLMs finetuned on datasets with paired instructions and annotated clicking coordinates or bounding boxes. We evaluated UGround-V1 (Gou et al. 2024), OS-Atlas (Wu et al. 2024b), Aria-UI (Yang et al. 2024), GTA-1 (Yang et al. 2025), GUI-Actor (Wu et al. 2024a), and UI-TARS (Qin et al. 2025).
两种通用类型的定位模型。第一种是标志集 (SoM) (Yang et al. 2023)，它使用现成的分割模型（如 SAM (Kirillov et al. 2023) 和 Semantic-SAM (Li et al. 2023)）分割网页上的显著元素，并提示 VLM 选择一个分割掩码，以识别对应于给定指令的点击区域。第二种是在具有配对指令和标注点击坐标或边界框的数据集上进行微调的 VLM。我们评估了 UGround-V1 (Gou et al. 2024)、OS-Atlas (Wu et al. 2024b)、Aria-UI (Yang et al. 2024)、GTA-1 (Yang et al. 2025)、GUI-Actor (Wu et al. 2024a) 和 UI-TARS (Qin et al. 2025)。


VLM/LRM + Grounding. Grounding models are designed or trained to ground a simple instruction to a webpage and thus tend to lack reasoning or planning capabilities. To address this, we leveraged VLMs and LRMs to first translate real user instructions to more understandable ones for grounding models. In particular, a VLM or an LRM needs to reason about the true user intent implied by the instruction and the spatial & temporal context. For instructions that require multiple actions, it needs to generate a plan to complete the instructions. Finally, it needs to generate a straightforward, clear instruction for the grounding model to produce the final action at each step. We evaluated state-of-the-art VLMs (OpenAI 2023; Team 2025; Qwen et al. 2025), as well as state-of-the-art LRMs (Jaech et al. 2024; Team 2025; Anthropic 2025). In the main results, we paired each VLM and LRM with the grounding model that achieved the highest step accuracy (GTA-1). For all VLMs and LRMs, we provide the past 10 steps for context, which we found to be a reasonable fixed context length in our preliminary study, balancing cost and informativeness. We also found that prompting models with screenshots of past webpages could incur a high cost. Therefore, we only prompt the models with the screenshot of the current webpage. For the history, we prompted GPT-40 to generate text-based action history based on consecutive screenshots and the instructions at each step. We then used this text-based history description for the evaluated VLMs and LRMs.
VLM/LRM + 定位。定位模型旨在或经训练用于将简单指令定位到网页，因此往往缺乏推理或规划能力。为了解决这个问题，我们利用 VLM 和 LRM 首先将真实的初级用户指令转换为定位模型更易理解的指令。具体而言，VLM 或 LRM 需要推理出指令以及空间和时间上下文所隐含的真实用户意图。对于需要多个动作的指令，它需要生成一个计划来完成指令。最后，它需要为定位模型生成一个直接、清晰的指令，以便在每一步产生最终动作。我们评估了最先进的 VLM (OpenAI 2023; Team 2025; Qwen et al. 2025) 以及最先进的 LRM (Jaech et al. 2024; Team 2025; Anthropic 2025)。在主要结果中，我们将每个 VLM 和 LRM 与获得最高单步准确率的定位模型 (GTA-1) 配对。对于所有 VLM 和 LRM，我们提供过去 10 个步骤作为上下文，在我们的初步研究中，我们发现这是一个合理的固定上下文长度，平衡了成本和信息量。我们还发现，使用过去网页的截图来提示模型可能会产生很高的成本。因此，我们仅使用当前网页的截图提示模型。对于历史记录，我们提示 GPT-4o 根据连续的截图和每一步的指令生成基于文本的动作历史。然后，我们将此基于文本的历史描述用于评估的 VLM 和 LRM。


Finetuning. To evaluate whether models can learn to better follow real-world user instructions with additional training, we finetuned the best-performing grounding model (GTA-1) following the model's original group relative policy optimization (GRPO) training procedure (Yang et al. 2025) on 9 participants' data and tested it on the held-out participants' instructions. Specifically, we trained the grounding model to produce an action based on the past 10 steps of actions (in text), the current webpage screenshot, and the instruction. We enumerated different train/test splits and reported the averaged performance, either using the finetuned model alone or pairing it with the best VLM or LRM.
微调。为了评估模型是否可以通过额外的训练更好地遵循现实世界的用户指令，我们按照模型原始的群体相对策略优化 (GRPO) 训练程序 (Yang et al. 2025)，在 9 名参与者的数据上微调了表现最好的定位模型 (GTA-1)，并在保留的参与者指令上对其进行了测试。具体来说，我们训练定位模型根据过去 10 个动作步骤（文本形式）、当前网页截图和指令来产生动作。我们列举了不同的训练/测试划分，并报告了平均性能，无论是单独使用微调模型，还是将其与最好的 VLM 或 LRM 配对。


## Results
## 结果


Main results are summarized in Table 3. All models fell short in following real user instructions. The highest task success rate was only 14.0%, and the highest average progress was only 28.7%, a large gap compared to humans (93.4% task success rate). This difference has a 95% confidence interval of [71.3, 87.5], and is highly significant with p-value < 0.0001. Grounding methods by themselves failed to finish most tasks. However, when paired with the best-performing grounding model (GTA-1), instructions generated by VLMs & LRMs significantly improved the performance. LRMs performed marginally better than most VLMs. Across all three metrics, Gemini 2.5 Flash, Gemini 2.5 Pro, and o3 showed the strongest performance. Finetun-ing GTA-1 on real user data marginally improved its performance, but finetuning offered no benefit when GTA-1 was paired with VLMs and LRMs, since the finetuned model is trained to adapt to real users' instructions instead of instructions generated by VLM or LRM.
主要结果总结在表3中。所有模型在遵循真实用户指令方面均表现不佳。最高任务成功率仅为14.0%，最高平均进度仅为28.7%，与人类（93.4%的任务成功率）相比存在巨大差距。该差异的95%置信区间为[71.3, 87.5]，且p值 &lt; 0.0001，具有高度显著性。定位方法本身无法完成大多数任务。然而，当与性能最好的定位模型（GTA-1）配合使用时，由VLM和LRM生成的指令显著提高了性能。LRM的表现略优于大多数VLM。在所有三项指标中，Gemini 2.5 Flash、Gemini 2.5 Pro和o3表现最强。在真实用户数据上微调GTA-1略微提升了其性能，但当GTA-1与VLM和LRM配对时，微调并未带来益处，因为微调后的模型旨在适应真实用户的指令，而非VLM或LRM生成的指令。


## Discussion
## 讨论


Can grounding models understand real-world user instructions? There remains a significant gap in the performance of current direct grounding methods. The best grounding model, GUI-Actor, has a task success rate of only 5.7%. Figure 5 illustrates various failure cases encountered when directly using GTA-1. Unsurprisingly, grounding models fail to interpret instructions requiring reasoning due to their limited reasoning capabilities. However, even for context-free instructions involving straightforward spatial reasoning-tasks where grounding methods should excel-they frequently misinterpret spatial layouts or rankings. For instance, they often incorrectly select elements for instructions such as "click the first one."
定位模型能理解真实世界的用户指令吗？当前的直接定位方法在性能上仍存在显著差距。表现最好的定位模型GUI-Actor的任务成功率仅为5.7%。图5展示了直接使用GTA-1时遇到的各种失败案例。不出所料，定位模型由于推理能力有限，无法解析需要推理的指令。然而，即使是涉及简单空间推理的无上下文指令——即定位方法本应擅长的任务——它们也经常误解空间布局或排序。例如，对于“点击第一个”之类的指令，它们经常错误地选择元素。


How can VLMs & LRMs help? VLMs or LRMs can convert the original user instructions into more direct and explicit descriptions that a grounding model can more easily understand. This is made possible by their reasoning capacities. For instance, in Figure 5A, the grounding model (GTA-1) on its own fails to select the first tab: it selects the first element instead of the first tab. However, it succeeds after o3 rewrites the instruction to refer to the title. As shown in Figure 5B, grounding models may sometimes still fail due to inherent limitations even when VLMs/LRMs generate clearer instructions. Nonetheless, incorporating VLMs or LRMs significantly improves overall performance.
VLM和LRM如何提供帮助？VLM或LRM可以将原始用户指令转换为更直接、更明确的描述，使定位模型更容易理解。这得益于它们的推理能力。例如，在图5A中，定位模型（GTA-1）自身无法选择第一个标签页：它选择了第一个元素而非第一个标签页。然而，在o3将指令重写为指向标题后，它取得了成功。如图5B所示，即使VLM/LRM生成了更清晰的指令，定位模型有时仍可能因固有局限性而失败。尽管如此，引入VLM或LRM显著提升了整体性能。


What are the limitations of VLMs & LRMs? While VLMs and LRMs help, the highest task success rate is still only 14.0%. Beyond errors from grounding models (e.g., Figure 5B), they continue to struggle with complex temporal reasoning. In Figure 5C, the user previously asked to open the first two search results in new tabs. When later instructed to "look at the first one we just opened," o3 failed to identify which element "the first one" referred to--instead of the first newly opened tab, it pointed to the first search result. We further analyze the error distribution between reasoning errors (the VLM/LRM mistranslates the instruction and refers to the wrong element) and grounding errors (the rewritten instruction is correct, but the grounding model still fails to click the right element). For the best model (o3 + GTA-1), 43.3% of errors are grounding errors and 56.7% are reasoning errors. This suggests that current VLMs and LRMs still lack the reasoning and planning abilities needed to robustly perform sequential instruction-following tasks.
VLM和LRM的局限性是什么？虽然VLM和LRM有所帮助，但最高任务成功率仍仅为14.0%。除了来自定位模型的错误（例如图5B）之外，它们在复杂的时序推理方面依然吃力。在图5C中，用户之前要求在新标签页中打开前两个搜索结果。当随后被指示“查看我们刚刚打开的第一个”时，o3未能识别出“第一个”指的是哪个元素——它指向了第一个搜索结果，而非第一个新打开的标签页。我们进一步分析了推理错误（VLM/LRM误译指令并指向错误元素）与定位错误（重写的指令正确，但定位模型仍未能点击正确元素）之间的错误分布。对于表现最好的模型（o3 + GTA-1），43.3%的错误是定位错误，56.7%是推理错误。这表明当前的VLM和LRM仍缺乏稳健执行序列指令遵循任务所需的推理和规划能力。


Does learning from real-world user data help? Fine-tuning GTA-1 marginally improved average progress and step accuracy but yielded no additional benefit when paired with VLMs and LRMs. These results show that the fine-tuned model better understands real user instructions, yet it still fails to generalize to instructions generated by VLMs and LRMs. The experiments suggest that finetuning grounding models on a small set of real user instructions provides minimal benefit, and collecting large-scale real user instructions remains a significant challenge.
从真实世界用户数据中学习是否有帮助？微调GTA-1略微提高了平均进度和步骤准确度，但与VLM和LRM配对时并未产生额外收益。这些结果表明，微调后的模型能更好地理解真实用户指令，但仍无法泛化到VLM和LRM生成的指令。实验表明，在少量真实用户指令上微调定位模型收益极小，而收集大规模真实用户指令仍是一个重大挑战。


<table><tr><td>Category</td><td>Model</td><td>Task Success</td><td>Progress</td><td>Step Accuracy</td></tr><tr><td>Human</td><td>Human Operator</td><td>93.4</td><td>96.4</td><td>99.2</td></tr><tr><td rowspan="7">Grounding</td><td>Set-of-Mark</td><td>0.0</td><td>2.7</td><td>29.8</td></tr><tr><td>OS-Atlas</td><td>0.0</td><td>3.8</td><td>26.6</td></tr><tr><td>Aria-UI</td><td>0.0</td><td>2.4</td><td>32.8</td></tr><tr><td>UGround-V1</td><td>0.0</td><td>6.2</td><td>47.7</td></tr><tr><td>UI-TARS</td><td>2.8</td><td>13.1</td><td>53.8</td></tr><tr><td>GTA-1</td><td>3.7</td><td>17.7</td><td>61.5</td></tr><tr><td>GUI-Actor</td><td>5.7</td><td>14.7</td><td>61.4</td></tr><tr><td rowspan="3">VLM + Grounding</td><td>GPT-40 + GTA-1</td><td>8.4</td><td>23.5</td><td>72.7</td></tr><tr><td>Owen 2.5 72B + GTA-1</td><td>9.3</td><td>24.3</td><td>69.0</td></tr><tr><td>Gemini 2.5 Flash + GTA-1</td><td>11.2</td><td>26.9</td><td>75.4</td></tr><tr><td rowspan="5">LRM + Grounding</td><td>o1 + GTA-1</td><td>7.5</td><td>17.7</td><td>68.2</td></tr><tr><td>Gemini 2.5 Pro + GTA-1</td><td>8.4</td><td>23.5</td><td>74.5</td></tr><tr><td>o4-mini + GTA-1</td><td>10.3</td><td>21.7</td><td>67.1</td></tr><tr><td>Claude 3.7 Sonnet + GTA-1</td><td>12.1</td><td>26.7</td><td>68.8</td></tr><tr><td>o3 + GTA-1</td><td>14.0</td><td>28.7</td><td>76.7</td></tr><tr><td rowspan="3">Finetuned</td><td>GTA-1-F</td><td>3.7 (+0.0)</td><td>19.7 (+2.0)</td><td>64.3 (+2.8)</td></tr><tr><td>Gemini 2.5 Flash + GTA-1-F</td><td>11.2 (+0.0)</td><td>26.9 (+0.0)</td><td>75.4 (+0.0)</td></tr><tr><td>o3 + GTA-1-F</td><td>14.0 (+0.0)</td><td>28.7 (+0.0)</td><td>76.7 (+0.0)</td></tr></table>
<table><tbody><tr><td>类别</td><td>模型</td><td>任务成功率</td><td>进度</td><td>步骤准确率</td></tr><tr><td>人类</td><td>人类操作员</td><td>93.4</td><td>96.4</td><td>99.2</td></tr><tr><td rowspan="7">接地 (Grounding)</td><td>标记集 (Set-of-Mark)</td><td>0.0</td><td>2.7</td><td>29.8</td></tr><tr><td>OS-Atlas</td><td>0.0</td><td>3.8</td><td>26.6</td></tr><tr><td>Aria-UI</td><td>0.0</td><td>2.4</td><td>32.8</td></tr><tr><td>UGround-V1</td><td>0.0</td><td>6.2</td><td>47.7</td></tr><tr><td>UI-TARS</td><td>2.8</td><td>13.1</td><td>53.8</td></tr><tr><td>GTA-1</td><td>3.7</td><td>17.7</td><td>61.5</td></tr><tr><td>GUI-Actor</td><td>5.7</td><td>14.7</td><td>61.4</td></tr><tr><td rowspan="3">VLM + Grounding</td><td>GPT-40 + GTA-1</td><td>8.4</td><td>23.5</td><td>72.7</td></tr><tr><td>Owen 2.5 72B + GTA-1</td><td>9.3</td><td>24.3</td><td>69.0</td></tr><tr><td>Gemini 2.5 Flash + GTA-1</td><td>11.2</td><td>26.9</td><td>75.4</td></tr><tr><td rowspan="5">LRM + Grounding</td><td>o1 + GTA-1</td><td>7.5</td><td>17.7</td><td>68.2</td></tr><tr><td>Gemini 2.5 Pro + GTA-1</td><td>8.4</td><td>23.5</td><td>74.5</td></tr><tr><td>o4-mini + GTA-1</td><td>10.3</td><td>21.7</td><td>67.1</td></tr><tr><td>Claude 3.7 Sonnet + GTA-1</td><td>12.1</td><td>26.7</td><td>68.8</td></tr><tr><td>o3 + GTA-1</td><td>14.0</td><td>28.7</td><td>76.7</td></tr><tr><td rowspan="3">微调 (Finetuned)</td><td>GTA-1-F</td><td>3.7 (+0.0)</td><td>19.7 (+2.0)</td><td>64.3 (+2.8)</td></tr><tr><td>Gemini 2.5 Flash + GTA-1-F</td><td>11.2 (+0.0)</td><td>26.9 (+0.0)</td><td>75.4 (+0.0)</td></tr><tr><td>o3 + GTA-1-F</td><td>14.0 (+0.0)</td><td>28.7 (+0.0)</td><td>76.7 (+0.0)</td></tr></tbody></table>


Table 2: Model Performance including task success rate, average progress, and step accuracy. All results are in %. The best performance of pretrained models and finetuned models is highlighted in bold. GTA-1-F indicates the finetuned GTA-1. Plus sign indicates the improvement compared to using the raw model for the same set of instructions.
表 2：模型性能，包括任务成功率、平均进度和步骤准确率。所有结果均以 % 表示。预训练模型和微调模型中的最佳性能以加粗显示。GTA-1-F 表示微调后的 GTA-1。加号表示相对于对同一组指令使用原始模型的改进。


<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/02/2026_02_18__00_22_27_3d0562.jpg"/>



Figure 5: Qualitative results. The captions show instructions generated by o3 (the best LRM). (A) Error corrected by using o3 to convert instructions. (B) Failure caused by GTA-1 when o3 reasons correctly. (C) Reasoning failure caused by o3.
图 5：定性结果。标题显示了由 o3（最佳 LRM）生成的指令。(A) 通过使用 o3 转换指令纠正的错误。(B) 当 o3 推理正确时由 GTA-1 导致的失败。(C) 由 o3 导致的推理失败。


Limitations. RealWebAssist represents an important first step towards evaluating web agents on long-horizon, real-user tasks. However, it has several limitations. The first is participant scale and diversity. Collecting real-user data is expensive and time-consuming. The number of participants is comparable to prior works that use expert annotators (Lù, Kasner, and Reddy 2024). However, we intend to increase user diversity in future versions of the benchmark. We will also open-source our data collection tools for community expansion of the dataset. Second, like prior benchmarks on real-world websites (Deng et al. 2024; Cheng et al. 2024), we constrain our evaluation to an offline setting to ensure reproducibility and safety. This is complementary to benchmarks that focus on interactive evaluation in sandbox environments (e.g., WebArena). We believe that web agents should be evaluated on both types of benchmarks to fully assess their capabilities. Lastly, the current setting does not allow dialogue between a user and the AI assistant, which we will explore in future work.
局限性。RealWebAssist 代表了在长程、真实用户任务上评估 Web 智能体迈出的重要第一步。然而，它存在若干局限性。首先是参与者的规模和多样性。收集真实用户数据既昂贵又耗时。参与者数量与之前使用专家标注者的工作相当（Lù, Kasner, and Reddy 2024）。不过，我们打算在基准测试的未来版本中增加用户多样性。我们还将开源我们的数据收集工具，以便社区扩展该数据集。其次，与以往针对真实世界网站的基准测试一样（Deng et al. 2024; Cheng et al. 2024），我们将评估限制在离线设置中，以确保可复现性和安全性。这与关注沙盒环境中交互式评估的基准测试（如 WebArena）相辅相成。我们认为 Web 智能体应在两类基准测试上进行评估，以全面衡量其能力。最后，目前的设置不允许用户与 AI 助手之间进行对话，我们将在未来的工作中对此进行探索。


## Conclusion
## 结论


In this paper, we present RealWebAssist, the first benchmark for evaluating web agents' ability to provide long-horizon web assistance with real-world users via sequential instruction-following. Our benchmark poses novel challenges, including spatial and temporal reasoning, planning, and adapting to user-specific routines. We conducted a comprehensive evaluation and analysis on multiple state-of-the-art GUI grounding models, VLMs, and LRMs, revealing critical limitations of them. We have also shown the limited benefit of finetuning models on real user data. Our benchmark, along with the well-annotated user instruction dataset, provides resources and diagnostic tools for further research on real-world web assistance. In future work, we plan to expand our human study to include more participants from various backgrounds, examine web assistance in interactive settings, and incorporate chat between users and web agents.
在本文中，我们提出了 RealWebAssist，这是首个通过顺序指令遵循来评估 Web 智能体为真实世界用户提供长程 Web 辅助能力的基准测试。我们的基准测试提出了新的挑战，包括空间和时间推理、规划以及适应用户特定的习惯。我们对多种先进的 GUI 定位模型、VLM 和 LRM 进行了全面的评估和分析，揭示了它们的关键局限性。我们也展示了在真实用户数据上微调模型的收益有限。我们的基准测试以及标注良好的用户指令数据集，为进一步研究真实世界 Web 辅助提供了资源和诊断工具。在未来的工作中，我们计划扩大人工研究规模，以纳入来自不同背景的更多参与者，在交互式设置中考察 Web 辅助，并整合用户与 Web 智能体之间的聊天。


## Acknowledgements
## 致谢


This work was supported by a research grant from Amazon. We thank Janice Chen for helpful discussions.
本研究由 Amazon 的研究资助支持。感谢 Janice Chen 参与有益的讨论。


## References
## 参考文献


Anthropic. 2025. Claude 3.7 Sonnet and Claude Code. https: //www.anthropic.com/news/claude-3-7-sonnet. Accessed: 2025-03-17.
Anthropic. 2025. Claude 3.7 Sonnet and Claude Code. https: //www.anthropic.com/news/claude-3-7-sonnet. Accessed: 2025-03-17.


Cai, T.; Wang, X.; Ma, T.; Chen, X.; and Zhou, D. 2024. Large Language Models as Tool Makers. arXiv:2305.17126.
Cai, T.; Wang, X.; Ma, T.; Chen, X.; and Zhou, D. 2024. Large Language Models as Tool Makers. arXiv:2305.17126.


Cheng, K.; Sun, Q.; Chu, Y.; Xu, F.; Li, Y.; Zhang, J.; and Wu, Z. 2024. Seeclick: Harnessing gui grounding for advanced visual gui agents. arXiv preprint arXiv:2401.10935.
Cheng, K.; Sun, Q.; Chu, Y.; Xu, F.; Li, Y.; Zhang, J.; and Wu, Z. 2024. Seeclick: Harnessing gui grounding for advanced visual gui agents. arXiv preprint arXiv:2401.10935.


Deng, X.; Gu, Y.; Zheng, B.; Chen, S.; Stevens, S.; Wang, B.; Sun, H.; and Su, Y. 2024. Mind2web: Towards a generalist agent for the web. Advances in Neural Information Processing Systems, 36.
Deng, X.; Gu, Y.; Zheng, B.; Chen, S.; Stevens, S.; Wang, B.; Sun, H.; and Su, Y. 2024. Mind2web: Towards a generalist agent for the web. Advances in Neural Information Processing Systems, 36.


Fried, D.; Tomlin, N.; Hu, J.; Patel, R.; and Nematzadeh, A. 2023. Pragmatics in Language Grounding: Phenomena, Tasks, and Modeling Approaches. arXiv:2211.08371.
Fried, D.; Tomlin, N.; Hu, J.; Patel, R.; and Nematzadeh, A. 2023. Pragmatics in Language Grounding: Phenomena, Tasks, and Modeling Approaches. arXiv:2211.08371.


Goodman, N. D.; and Frank, M. C. 2016. Pragmatic language interpretation as probabilistic inference. Trends in cognitive sciences, 20(11): 818-829.
Goodman, N. D.; and Frank, M. C. 2016. Pragmatic language interpretation as probabilistic inference. Trends in cognitive sciences, 20(11): 818-829.


Gou, B.; Wang, R.; Zheng, B.; Xie, Y.; Chang, C.; Shu, Y.; Sun, H.; and Su, Y. 2024. Navigating the digital world as humans do: Universal visual grounding for gui agents. arXiv preprint arXiv:2410.05243.
Gou, B.; Wang, R.; Zheng, B.; Xie, Y.; Chang, C.; Shu, Y.; Sun, H.; and Su, Y. 2024. Navigating the digital world as humans do: Universal visual grounding for gui agents. arXiv preprint arXiv:2410.05243.


Gur, I.; Furuta, H.; Huang, A.; Safdari, M.; Matsuo, Y.; Eck, D.; and Faust, A. 2024. A Real-World WebAgent with Planning, Long Context Understanding, and Program Synthesis. arXiv:2307.12856.
Gur, I.; Furuta, H.; Huang, A.; Safdari, M.; Matsuo, Y.; Eck, D.; and Faust, A. 2024. A Real-World WebAgent with Planning, Long Context Understanding, and Program Synthesis. arXiv:2307.12856.


He, H.; Yao, W.; Ma, K.; Yu, W.; Dai, Y.; Zhang, H.; Lan, Z.; and Yu, D. 2024. WebVoyager: Building an End-to-End Web Agent with Large Multimodal Models. arXiv:2401.13919.
He, H.; Yao, W.; Ma, K.; Yu, W.; Dai, Y.; Zhang, H.; Lan, Z.; 和 Yu, D. 2024. WebVoyager: 使用大型多模态模型构建端到端网络智能体. arXiv:2401.13919.


Hong, W.; Wang, W.; Lv, Q.; Xu, J.; Yu, W.; Ji, J.; Wang, Y.; Wang, Z.; Zhang, Y.; Li, J.; Xu, B.; Dong, Y.; Ding, M.; and Tang, J. 2024. CogAgent: A Visual Language Model for GUI Agents. arXiv:2312.08914.
Hong, W.; Wang, W.; Lv, Q.; Xu, J.; Yu, W.; Ji, J.; Wang, Y.; Wang, Z.; Zhang, Y.; Li, J.; Xu, B.; Dong, Y.; Ding, M.; 和 Tang, J. 2024. CogAgent: 用于 GUI 智能体的视觉语言模型. arXiv:2312.08914.


Humphreys, P. C.; Raposo, D.; Pohlen, T.; Thornton, G.; Chhaparia, R.; Muldal, A.; Abramson, J.; Georgiev, P.; Santoro, A.; and Lillicrap, T. 2022. A data-driven approach for learning to control computers. In International Conference on Machine Learning, 9466-9482. PMLR.
Humphreys, P. C.; Raposo, D.; Pohlen, T.; Thornton, G.; Chhaparia, R.; Muldal, A.; Abramson, J.; Georgiev, P.; Santoro, A.; 和 Lillicrap, T. 2022. 一种用于学习控制计算机的数据驱动方法. In International Conference on Machine Learning, 9466-9482. PMLR.


Jaech, A.; Kalai, A.; Lerer, A.; Richardson, A.; El-Kishky, A.; Low, A.; Helyar, A.; Madry, A.; Beutel, A.; Carney, A.; et al. 2024. Openai o1 system card. arXiv preprint arXiv:2412.16720.
Jaech, A.; Kalai, A.; Lerer, A.; Richardson, A.; El-Kishky, A.; Low, A.; Helyar, A.; Madry, A.; Beutel, A.; Carney, A.; et al. 2024. OpenAI o1 系统卡片. arXiv preprint arXiv:2412.16720.


Jang, L.; Li, Y.; Zhao, D.; Ding, C.; Lin, J.; Liang, P. P.; Bon-atti, R.; and Koishida, K. 2024. Videowebarena: Evaluating long context multimodal agents with video understanding web tasks. arXiv preprint arXiv:2410.19100.
Jang, L.; Li, Y.; Zhao, D.; Ding, C.; Lin, J.; Liang, P. P.; Bon-atti, R.; 和 Koishida, K. 2024. Videowebarena: 通过视频理解网络任务评估长上下文多模态智能体. arXiv preprint arXiv:2410.19100.


Kim, M.; Bursztyn, V.; Koh, E.; Guo, S.; and Hwang, S.- w. 2024. Rada: Retrieval-augmented web agent planning with llms. In Findings of the Association for Computational Linguistics ACL 2024, 13511-13525.
Kim, M.; Bursztyn, V.; Koh, E.; Guo, S.; 和 Hwang, S.- w. 2024. Rada: 使用大语言模型进行检索增强的网络智能体规划. In Findings of the Association for Computational Linguistics ACL 2024, 13511-13525.


Kirillov, A.; Mintun, E.; Ravi, N.; Mao, H.; Rolland, C.; Gustafson, L.; Xiao, T.; Whitehead, S.; Berg, A. C.; Lo, W.- Y.; Dollár, P.; and Girshick, R. 2023. Segment Anything. arXiv:2304.02643.
Kirillov, A.; Mintun, E.; Ravi, N.; Mao, H.; Rolland, C.; Gustafson, L.; Xiao, T.; Whitehead, S.; Berg, A. C.; Lo, W.- Y.; Dollár, P.; 和 Girshick, R. 2023. Segment Anything (分割一切). arXiv:2304.02643.


Li, F.; Zhang, H.; Sun, P.; Zou, X.; Liu, S.; Yang, J.; Li, C.; Zhang, L.; and Gao, J. 2023. Semantic-SAM: Segment and Recognize Anything at Any Granularity. arXiv preprint arXiv:2307.04767.
Li, F.; Zhang, H.; Sun, P.; Zou, X.; Liu, S.; Yang, J.; Li, C.; Zhang, L.; 和 Gao, J. 2023. Semantic-SAM: 在任何粒度上分割并识别一切. arXiv preprint arXiv:2307.04767.


Liu, E. Z.; Guu, K.; Pasupat, P.; Shi, T.; and Liang, P. 2018. Reinforcement learning on web interfaces using workflow-guided exploration. arXiv preprint arXiv:1802.08802.
Liu, E. Z.; Guu, K.; Pasupat, P.; Shi, T.; 和 Liang, P. 2018. 使用工作流引导探索在网络界面上进行强化学习. arXiv preprint arXiv:1802.08802.


Liu, Z.; Yao, W.; Zhang, J.; Xue, L.; Heinecke, S.; Murthy, R.; Feng, Y.; Chen, Z.; Niebles, J. C.; Arpit, D.; et al. 2023. Bolaa: Benchmarking and orchestrating llm-augmented autonomous agents. arXiv preprint arXiv:2308.05960.
Liu, Z.; Yao, W.; Zhang, J.; Xue, L.; Heinecke, S.; Murthy, R.; Feng, Y.; Chen, Z.; Niebles, J. C.; Arpit, D.; et al. 2023. Bolaa: 基准测试与编排大语言模型增强的自主智能体. arXiv preprint arXiv:2308.05960.


Lù, X. H.; Kasner, Z.; and Reddy, S. 2024. Weblinx: Real-world website navigation with multi-turn dialogue. arXiv preprint arXiv:2402.05930.
Lù, X. H.; Kasner, Z.; 和 Reddy, S. 2024. Weblinx: 基于多轮对话的真实世界网站导航. arXiv preprint arXiv:2402.05930.


Majumder, B. P.; Mishra, B. D.; Jansen, P.; Tafjord, O.; Tan-don, N.; Zhang, L.; Callison-Burch, C.; and Clark, P. 2023. CLIN: A Continually Learning Language Agent for Rapid Task Adaptation and Generalization. arXiv:2310.10134.
Majumder, B. P.; Mishra, B. D.; Jansen, P.; Tafjord, O.; Tan-don, N.; Zhang, L.; Callison-Burch, C.; 和 Clark, P. 2023. CLIN: 用于快速任务适应与泛化的持续学习语言智能体. arXiv:2310.10134.


McCarthy, W. P.; Hawkins, R. D.; Wang, H.; Holdaway, C.; and Fan, J. E. 2021. Learning to communicate about shared procedural abstractions. arXiv preprint arXiv:2107.00077.
McCarthy, W. P.; Hawkins, R. D.; Wang, H.; Holdaway, C.; 和 Fan, J. E. 2021. 学习交流共享的过程抽象. arXiv preprint arXiv:2107.00077.


Nakano, R.; Hilton, J.; Balaji, S.; Wu, J.; Ouyang, L.; Kim, C.; Hesse, C.; Jain, S.; Kosaraju, V.; Saunders, W.; Jiang, X.; Cobbe, K.; Eloundou, T.; Krueger, G.; Button, K.; Knight, M.; Chess, B.; and Schulman, J. 2022. WebGPT: Browser-assisted question-answering with human feedback. arXiv:2112.09332.
Nakano, R.; Hilton, J.; Balaji, S.; Wu, J.; Ouyang, L.; Kim, C.; Hesse, C.; Jain, S.; Kosaraju, V.; Saunders, W.; Jiang, X.; Cobbe, K.; Eloundou, T.; Krueger, G.; Button, K.; Knight, M.; Chess, B.; 以及 Schulman, J. 2022. WebGPT: 结合人类反馈的浏览器辅助问答。arXiv:2112.09332。


Nogueira, R.; and Cho, K. 2016. End-to-end goal-driven web navigation. Advances in neural information processing systems, 29.
Nogueira, R.; 以及 Cho, K. 2016. 端到端目标驱动的网页导航。Advances in neural information processing systems, 29。


OpenAI. 2023. GPT-4 Technical Report. ArXiv, abs/2303.08774.
OpenAI. 2023. GPT-4 技术报告。ArXiv, abs/2303.08774。


Pan, J.; Zhang, Y.; Tomlin, N.; Zhou, Y.; Levine, S.; and Suhr, A. 2024. Autonomous Evaluation and Refinement of Digital Agents. arXiv:2404.06474.
Pan, J.; Zhang, Y.; Tomlin, N.; Zhou, Y.; Levine, S.; 以及 Suhr, A. 2024. 数字化智能体的自主评估与优化。arXiv:2404.06474。


Pang, R. Y.; Yuan, W.; Cho, K.; He, H.; Sukhbaatar, S.; and Weston, J. 2024. Iterative Reasoning Preference Optimization. arXiv:2404.19733.
Pang, R. Y.; Yuan, W.; Cho, K.; He, H.; Sukhbaatar, S.; 以及 Weston, J. 2024. 迭代推理偏好优化。arXiv:2404.19733。


Putta, P.; Mills, E.; Garg, N.; Motwani, S.; Finn, C.; Garg, D.; and Rafailov, R. 2024. Agent q: Advanced reasoning and learning for autonomous ai agents. arXiv preprint arXiv:2408.07199.
Putta, P.; Mills, E.; Garg, N.; Motwani, S.; Finn, C.; Garg, D.; 以及 Rafailov, R. 2024. Agent Q：针对自主 AI 智能体的高级推理与学习。arXiv preprint arXiv:2408.07199。


Qin, Y.; Ye, Y.; Fang, J.; Wang, H.; Liang, S.; Tian, S.; Zhang, J.; Li, J.; Li, Y.; Huang, S.; et al. 2025. UI-TARS: Pioneering Automated GUI Interaction with Native Agents. arXiv preprint arXiv:2501.12326.
Qin, Y.; Ye, Y.; Fang, J.; Wang, H.; Liang, S.; Tian, S.; Zhang, J.; Li, J.; Li, Y.; Huang, S.; 等. 2025. UI-TARS：利用原生智能体开拓自动化 GUI 交互。arXiv preprint arXiv:2501.12326。


Qwen; ;; Yang, A.; Yang, B.; Zhang, B.; Hui, B.; Zheng, B.; Yu, B.; Li, C.; Liu, D.; Huang, F.; Wei, H.; Lin, H.; Yang, J.; Tu, J.; Zhang, J.; Yang, J.; Yang, J.; Zhou, J.; Lin, J.; Dang, K.; Lu, K.; Bao, K.; Yang, K.; Yu, L.; Li, M.; Xue, M.; Zhang, P.; Zhu, Q.; Men, R.; Lin, R.; Li, T.; Tang, T.; Xia, T.; Ren, X.; Ren, X.; Fan, Y.; Su, Y.; Zhang, Y.; Wan, Y.; Liu, Y.; Cui, Z.; Zhang, Z.; and Qiu, Z. 2025. Qwen2.5 Technical Report. arXiv:2412.15115.
Qwen; ;; Yang, A.; Yang, B.; Zhang, B.; Hui, B.; Zheng, B.; Yu, B.; Li, C.; Liu, D.; Huang, F.; Wei, H.; Lin, H.; Yang, J.; Tu, J.; Zhang, J.; Yang, J.; Yang, J.; Zhou, J.; Lin, J.; Dang, K.; Lu, K.; Bao, K.; Yang, K.; Yu, L.; Li, M.; Xue, M.; Zhang, P.; Zhu, Q.; Men, R.; Lin, R.; Li, T.; Tang, T.; Xia, T.; Ren, X.; Ren, X.; Fan, Y.; Su, Y.; Zhang, Y.; Wan, Y.; Liu, Y.; Cui, Z.; Zhang, Z.; 以及 Qiu, Z. 2025. Qwen2.5 技术报告。arXiv:2412.15115。


Radford, A.; Kim, J. W.; Xu, T.; Brockman, G.; McLeavey, C.; and Sutskever, I. 2023. Robust speech recognition via large-scale weak supervision. In International conference on machine learning, 28492-28518. PMLR.
Radford, A.; Kim, J. W.; Xu, T.; Brockman, G.; McLeavey, C.; 以及 Sutskever, I. 2023. 通过大规模弱监督实现鲁棒语音识别。In International conference on machine learning, 28492-28518. PMLR。


Reddy, C. K.; Beyrami, E.; Pool, J.; Cutler, R.; Srinivasan, S.; and Gehrke, J. 2019. A scalable noisy speech dataset and online subjective test framework. arXiv preprint arXiv:1909.08050.
Reddy, C. K.; Beyrami, E.; Pool, J.; Cutler, R.; Srinivasan, S.; 以及 Gehrke, J. 2019. 一种可扩展的嘈杂语音数据集及在线主观测试框架。arXiv preprint arXiv:1909.08050。


Shi, T.; Karpathy, A.; Fan, L.; Hernandez, J.; and Liang, P. 2017. World of bits: An open-domain platform for web-based agents. In International Conference on Machine Learning, 3135-3144. PMLR.
Shi, T.; Karpathy, A.; Fan, L.; Hernandez, J.; 以及 Liang, P. 2017. World of Bits：一个基于网页的智能体开放域平台。In International Conference on Machine Learning, 3135-3144. PMLR。


Song, Y.; Thai, K.; Pham, C. M.; Chang, Y.; Nadaf, M.; and Iyyer, M. 2025. Bearcubs: A benchmark for computer-using web agents. arXiv preprint arXiv:2503.07919.
Song, Y.; Thai, K.; Pham, C. M.; Chang, Y.; Nadaf, M.; 以及 Iyyer, M. 2025. Bearcubs：计算机端网页智能体基准测试。arXiv preprint arXiv:2503.07919。


Team. 2025. Gemini 2.5: Pushing the Frontier with Advanced Reasoning, Multimodality, Long Context, and Next Generation Agentic Capabilities. arXiv:2507.06261.
Team. 2025. Gemini 2.5：通过高级推理、多模态、长上下文及下一代智能体能力突破边界。arXiv:2507.06261。


Wang, Z. Z.; Mao, J.; Fried, D.; and Neubig, G. 2024. Agent workflow memory. arXiv preprint arXiv:2409.07429.
Wang, Z. Z.; Mao, J.; Fried, D.; 以及 Neubig, G. 2024. 智能体工作流记忆。arXiv preprint arXiv:2409.07429。


Wu, Q.; Cheng, K.; Yang, R.; Zhang, C.; Yang, J.; Jiang, H.; Mu, J.; Peng, B.; Qiao, B.; Tan, R.; et al. 2025. GUI-Actor: Coordinate-Free Visual Grounding for GUI Agents. arXiv preprint arXiv:2506.03143.
Wu, Q.; Cheng, K.; Yang, R.; Zhang, C.; Yang, J.; Jiang, H.; Mu, J.; Peng, B.; Qiao, B.; Tan, R.; et al. 2025. GUI-Actor: 无坐标 GUI 智能体视觉定位。arXiv preprint arXiv:2506.03143.


Wu, Z.; Han, C.; Ding, Z.; Weng, Z.; Liu, Z.; Yao, S.; Yu, T.; and Kong, L. 2024a. OS-Copilot: Towards Generalist Computer Agents with Self-Improvement. arXiv:2402.07456.
Wu, Z.; Han, C.; Ding, Z.; Weng, Z.; Liu, Z.; Yao, S.; Yu, T.; and Kong, L. 2024a. OS-Copilot: 迈向具有自我改进能力的通用计算机智能体。arXiv:2402.07456.


Wu, Z.; Wu, Z.; Xu, F.; Wang, Y.; Sun, Q.; Jia, C.; Cheng, K.; Ding, Z.; Chen, L.; Liang, P. P.; et al. 2024b. Os-atlas: A foundation action model for generalist gui agents. arXiv preprint arXiv:2410.23218.
Wu, Z.; Wu, Z.; Xu, F.; Wang, Y.; Sun, Q.; Jia, C.; Cheng, K.; Ding, Z.; Chen, L.; Liang, P. P.; et al. 2024b. OS-Atlas: 通用 GUI 智能体基础动作模型。arXiv preprint arXiv:2410.23218.


Xu, Y.; Wang, Z.; Wang, J.; Lu, D.; Xie, T.; Saha, A.; Sahoo, D.; Yu, T.; and Xiong, C. 2024. Aguvis: Unified Pure Vision Agents for Autonomous GUI Interaction. arXiv:2412.04454.
Xu, Y.; Wang, Z.; Wang, J.; Lu, D.; Xie, T.; Saha, A.; Sahoo, D.; Yu, T.; and Xiong, C. 2024. Aguvis: 用于自主 GUI 交互的统一纯视觉智能体。arXiv:2412.04454.


Xu, Y.; Wang, Z.; Wang, J.; Lu, D.; Xie, T.; Saha, A.; Sahoo, D.; Yu, T.; and Xiong, C. 2025. Aguvis: Unified Pure Vision Agents for Autonomous GUI Interaction. arXiv:2412.04454.
Xu, Y.; Wang, Z.; Wang, J.; Lu, D.; Xie, T.; Saha, A.; Sahoo, D.; Yu, T.; and Xiong, C. 2025. Aguvis: 用于自主 GUI 交互的统一纯视觉智能体。arXiv:2412.04454.


Yang, J.; Zhang, H.; Li, F.; Zou, X.; Li, C.; and Gao, J. 2023. Set-of-Mark Prompting Unleashes Extraordinary Visual Grounding in GPT-4V. arXiv preprint arXiv:2310.11441.
Yang, J.; Zhang, H.; Li, F.; Zou, X.; Li, C.; and Gao, J. 2023. Set-of-Mark 提示词激发 GPT-4V 卓越的视觉定位能力。arXiv preprint arXiv:2310.11441.


Yang, Y.; Li, D.; Dai, Y.; Yang, Y.; Luo, Z.; Zhao, Z.; Hu, Z.; Huang, J.; Saha, A.; Chen, Z.; et al. 2025. GTA1: GUI Test-time Scaling Agent. arXiv preprint arXiv:2507.05791.
Yang, Y.; Li, D.; Dai, Y.; Yang, Y.; Luo, Z.; Zhao, Z.; Hu, Z.; Huang, J.; Saha, A.; Chen, Z.; et al. 2025. GTA1: GUI 测试时推理扩展智能体。arXiv preprint arXiv:2507.05791.


Yang, Y.; Wang, Y.; Li, D.; Luo, Z.; Chen, B.; Huang, C.; and Li, J. 2024. Aria-UI: Visual Grounding for GUI Instructions. arXiv preprint arXiv:2412.16256.
Yang, Y.; Wang, Y.; Li, D.; Luo, Z.; Chen, B.; Huang, C.; and Li, J. 2024. Aria-UI: GUI 指令的视觉定位。arXiv preprint arXiv:2412.16256.


Yao, S.; Chen, H.; Yang, J.; and Narasimhan, K. 2022. Webshop: Towards scalable real-world web interaction with grounded language agents. Advances in Neural Information Processing Systems, 35: 20744-20757.
Yao, S.; Chen, H.; Yang, J.; and Narasimhan, K. 2022. Webshop: 致力于通过定位语言智能体实现可扩展的真实世界网络交互。Advances in Neural Information Processing Systems, 35: 20744-20757.


Yao, S.; Zhao, J.; Yu, D.; Du, N.; Shafran, I.; Narasimhan, K.; and Cao, Y. 2023. ReAct: Synergizing Reasoning and Acting in Language Models. arXiv:2210.03629.
Yao, S.; Zhao, J.; Yu, D.; Du, N.; Shafran, I.; Narasimhan, K.; and Cao, Y. 2023. ReAct: 在语言模型中协同推理与行动。arXiv:2210.03629.


Ying, L.; Liu, J. X.; Aarya, S.; Fang, Y.; Tellex, S.; Tenenbaum, J. B.; and Shu, T. 2024. SIFToM: Robust Spoken Instruction Following through Theory of Mind. arXiv:2409.10849.
Ying, L.; Liu, J. X.; Aarya, S.; Fang, Y.; Tellex, S.; Tenenbaum, J. B.; and Shu, T. 2024. SIFToM: 通过心智理论实现鲁棒的口头指令遵循。arXiv:2409.10849.


Yuan, X.; Zhang, J.; Li, K.; Cai, Z.; Yao, L.; Chen, J.; Wang, E.; Hou, Q.; Chen, J.; Jiang, P.-T.; and Li, B. 2025. Enhancing Visual Grounding for GUI Agents via Self-Evolutionary Reinforcement Learning. arXiv:2505.12370.
Yuan, X.; Zhang, J.; Li, K.; Cai, Z.; Yao, L.; Chen, J.; Wang, E.; Hou, Q.; Chen, J.; Jiang, P.-T.; and Li, B. 2025. 通过自我进化强化学习增强 GUI 智能体的视觉定位。arXiv:2505.12370.


Zhang, C.; Yang, Z.; Liu, J.; Han, Y.; Chen, X.; Huang, Z.; Fu, B.; and Yu, G. 2023. AppAgent: Multimodal Agents as Smartphone Users. arXiv:2312.13771.
Zhang, C.; Yang, Z.; Liu, J.; Han, Y.; Chen, X.; Huang, Z.; Fu, B.; and Yu, G. 2023. AppAgent: 作为智能手机用户的多模态智能体。arXiv:2312.13771.


Zhang, Z.; and Zhang, A. 2024. You Only Look at Screens: Multimodal Chain-of-Action Agents. arXiv:2309.11436.
Zhang, Z.; and Zhang, A. 2024. You Only Look at Screens: 多模态动作链智能体。arXiv:2309.11436.


Zheng, B.; Gou, B.; Kil, J.; Sun, H.; and Su, Y. 2024a. Gpt- 4v (ision) is a generalist web agent, if grounded. arXiv preprint arXiv:2401.01614.
Zheng, B.; Gou, B.; Kil, J.; Sun, H.; and Su, Y. 2024a. 只要具备定位能力，GPT-4V(ision) 就是通用网络智能体。arXiv preprint arXiv:2401.01614.


Zheng, L.; Wang, R.; Wang, X.; and An, B. 2024b. Synapse: Trajectory-as-Exemplar Prompting with Memory for Computer Control. arXiv:2306.07863.
Zheng, L.; Wang, R.; Wang, X.; and An, B. 2024b. Synapse: Trajectory-as-Exemplar Prompting with Memory for Computer Control. arXiv:2306.07863.


Zhou, A.; Yan, K.; Shlapentokh-Rothman, M.; Wang, H.; and Wang, Y.-X. 2024. Language Agent Tree Search Unifies Reasoning Acting and Planning in Language Models. arXiv:2310.04406.
Zhou, A.; Yan, K.; Shlapentokh-Rothman, M.; Wang, H.; and Wang, Y.-X. 2024. Language Agent Tree Search Unifies Reasoning Acting and Planning in Language Models. arXiv:2310.04406.


Zhou, S.; Xu, F. F.; Zhu, H.; Zhou, X.; Lo, R.; Sridhar, A.; Cheng, X.; Ou, T.; Bisk, Y.; Fried, D.; et al. 2023. Webarena: A realistic web environment for building autonomous agents. arXiv preprint arXiv:2307.13854.
Zhou, S.; Xu, F. F.; Zhu, H.; Zhou, X.; Lo, R.; Sridhar, A.; Cheng, X.; Ou, T.; Bisk, Y.; Fried, D.; et al. 2023. Webarena: A realistic web environment for building autonomous agents. arXiv preprint arXiv:2307.13854.


## Appendix
## 附录


More experiment results
更多实验结果


Full VLM & LRM + Grounding results
完整的 VLM &amp; LRM + Grounding 结果


For the best three grounding models, GTA-1 (Yang et al. 2025), GUI-Actor (Wu et al. 2025) and UI-TARS (Qin et al. 2025), we test their pairing with all the VLMs and LRMs. Table 3 shows the full results. All the evaluation experiments are run on a single A100 GPU for 20 - 40 minutes. Finetun-ing GTA-1 model takes 4 hours on 4 A100 GPUs.
针对表现最好的三个定位模型 GTA-1 (Yang et al. 2025)、GUI-Actor (Wu et al. 2025) 和 UI-TARS (Qin et al. 2025)，我们测试了它们与所有 VLM 和 LRM 的配对情况。表 3 展示了完整结果。所有评估实验均在单块 A100 GPU 上运行 20 - 40 分钟。在 4 块 A100 GPU 上微调 GTA-1 模型需要 4 小时。


Experiment with different context lengths
不同上下文长度的实验


<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/02/2026_02_18__00_22_27_dd198b.jpg"/>



Figure 6: Effect of context length on Gemini 2.5 Flash + GTA-1.
图 6：上下文长度对 Gemini 2.5 Flash + GTA-1 的影响。


We evaluated the best-performing VLM (Gemini 2.5 Flash) + GTA-1 with varying history context lengths, from no history to 20 steps. An ideal assistant should be able to leverage different kinds of historical context based on different instructions, ranging from no history to multi-task history context (e.g., for routine learning). As shown in Figure 6, increasing context length also does not necessarily lead to better performance. Gemini 2.5 Flash + GTA-1 achieved the highest task success rate with a context length of 10, and increasing the context length further led to poorer performance. This suggest the limitation of VLM in effectively utilizing historical context for reasoning.
我们评估了表现最好的 VLM (Gemini 2.5 Flash) + GTA-1 在不同历史上下文长度下的表现，范围从无历史记录到 20 步。一个理想的助手应当能够根据不同的指令利用各类历史上下文，从无历史到多任务历史上下文（例如用于常规学习）。如图 6 所示，增加上下文长度并不一定会带来更好的性能。Gemini 2.5 Flash + GTA-1 在上下文长度为 10 时达到了最高的任务成功率，而进一步增加上下文长度会导致性能下降。这表明 VLM 在有效利用历史上下文进行推理方面存在局限性。


## Effect of Speech Recognition Errors
## 语音识别错误的影响


All baseline experiments use the ground truth transcripts of user speech instructions as input to ensure that performance is not affected by errors in speech-to-text transcription. However, in real-world settings, instructions are often given via speech. To reflect this, we evaluated the effect of speech recognition on the agent's performance by using the transcripts generated from a state-of-the-art automatic speech recognition (ASR) model, Whisper Large-V3 (Radford et al. 2023). Additionally, since users may not always be in quiet, controlled environments using a high-quality microphone like in our user experiment setup, we simulated noisy environments by injecting background noise with noise files from the Microsoft Scalable Noisy Speech Dataset (MS-SNSD) dataset (Reddy et al. 2019), following (Ying et al. 2024). The noise files include people talking in the background and keyboard typing sounds. As shown in Table 4, using speech recognition resulted in a 1.9% drop in task success rate, and having noisy speech resulted in a further 1.9% drop. In contrast, the word error rate (WER) of the ASR results increased from 1.4% (original speech) to 28.1% (noisy speech), a much larger performance drop compared to the final task performance. This result suggests that reasoning the true meanings of speech instructions by leveraging context can help mitigate errors from ASR.
所有基线实验均使用用户语音指令的地面真值文本作为输入，以确保性能不受语音转文字识别错误的影响。然而，在现实场景中，指令通常通过语音发出。为了反映这一点，我们使用最先进的自动语音识别 (ASR) 模型 Whisper Large-V3 (Radford et al. 2023) 生成的转录文本，评估了语音识别对智能体性能的影响。此外，由于用户并不总是在像我们用户实验设置那样安静、受控且使用高质量麦克风的环境中，我们参考 (Ying et al. 2024)，通过注入来自 Microsoft 可扩展噪声语音数据集 (MS-SNSD) (Reddy et al. 2019) 的噪声文件来模拟噪声环境。噪声文件包括背景中的人声对话和键盘输入声。如表 4 所示，使用语音识别导致任务成功率下降了 1.9%，而加入噪声语音则进一步下降了 1.9%。相比之下，ASR 结果的词错误率 (WER) 从 1.4%（原始语音）上升到 28.1%（噪声语音），性能降幅远大于最终的任务表现。这一结果表明，通过利用上下文推理语音指令的真实含义，有助于缓解 ASR 带来的错误。


## Dataset Construction Details
## 数据集构建细节


Video Segmenting. As shown in the video example, the interactive sessions are highly dynamic, and spoken instructions do not always align cleanly with specific screens or timesteps. Automatically segmenting instructions and matching them to corresponding webpages and actions using heuristics would risk significantly degrading data quality. Therefore, we manually segment the live sessions using video editing software to construct the final RealWe-bAssist dataset. All participants provided consent to have their speech recorded and included in this dataset.
视频分段。如视频示例所示，交互过程是高度动态的，且口头指令并不总是能与特定的屏幕或时间步精确对齐。使用启发式方法自动分割指令并匹配相应的网页和动作会有显著降低数据质量的风险。因此，我们使用视频编辑软件手动分割实时会话，以构建最终的 RealWebAssist 数据集。所有参与者均同意录制其语音并将其包含在本数据集中。


Bounding Box Labeling. As shown in Figure 7, certain instructions like "close all the tabs" may correspond to multiple valid actions, since closing any of the tabs first would be reasonable. Therefore, we add bounding boxes to all of the elements that would be correct. The bounding boxes are drawn manually using a Python tool built with tkinter, and the clickable regions are determined by a visual inspection of the webpage.
边界框标注。如图 7 所示，某些指令（如“关闭所有标签页”）可能对应多个有效动作，因为先关闭其中任何一个标签页都是合理的。因此，我们为所有正确的元素都添加了边界框。边界框是使用基于 tkinter 构建的 Python 工具手动绘制的，可点击区域通过对网页的视觉检查来确定。


## More Dataset Details
## 更多数据集细节


## Evaluation detail
## 评估详情


User instructions in RealWebAssist require different operations on the webpage, including clicking, scrolling and typing. We believe that action types other than clicking is trivial (for typing actions, the benchmark includes the step of finding the correct place to type instead of the actual typing process), so we only evaluate click-type actions with annotated bounding boxes are scored; instructions like "scroll" remain in the history but are not counted in our metrics. Of the 1,885 instructions, 1,412 are scored, yielding 1,714 evaluated action steps (one screenshot per step). Tasks average 17.6 evaluated steps.
RealWebAssist 中的用户指令需要对网页进行不同的操作，包括点击、滚动和输入。我们认为点击以外的操作类型较为琐碎（对于输入操作，基准测试包含寻找正确输入位置的步骤，而非实际输入过程），因此我们仅对带有标注边界框的点击类动作进行评分；诸如“滚动”之类的指令保留在历史记录中，但不计入指标。在 1,885 条指令中，有 1,412 条参与评分，产生了 1,714 个评估动作步骤（每步一张截图）。任务平均包含 17.6 个评估步骤。


<table><tr><td rowspan="3">VLM + GTA-1</td><td>GPT-4o + GTA-1</td><td>8.4</td><td>23.5</td><td>72.7</td></tr><tr><td>Qwen 2.5 72B + GTA-1</td><td>9.3</td><td>24.3</td><td>69.0</td></tr><tr><td>Gemini 2.5 Flash + GTA-1</td><td>11.2</td><td>26.9</td><td>75.4</td></tr><tr><td rowspan="5">LRM + GTA-1</td><td>Claude 3.7 Sonnet + GTA-1</td><td>12.1</td><td>26.7</td><td>68.8</td></tr><tr><td>Gemini 2.5 Pro + GTA-1</td><td>8.4</td><td>23.5</td><td>74.5</td></tr><tr><td>o1 + GTA-1</td><td>7.5</td><td>21.1</td><td>73.1</td></tr><tr><td>o3 + GTA-1</td><td>14.0</td><td>28.7</td><td>76.7</td></tr><tr><td>o4-mini + GTA-1</td><td>10.3</td><td>21.7</td><td>67.1</td></tr><tr><td rowspan="3">VLM + <br> GUI-Actor</td><td>GPT-40 + GUI-Actor</td><td>6.5</td><td>18.0</td><td>67.0</td></tr><tr><td>Qwen 2.5 72B + GUI-Actor</td><td>9.3</td><td>21.4</td><td>64.9</td></tr><tr><td>Gemini 2.5 Flash + GUI-Actor</td><td>10.3</td><td>25.6</td><td>73.1</td></tr><tr><td rowspan="5">LRM + GUI-Actor</td><td>Claude 3.7 Sonnet+ GUI-Actor</td><td>7.5</td><td>18.5</td><td>63.9</td></tr><tr><td>Gemini 2.5 Pro + GUI-Actor</td><td>9.3</td><td>24.0</td><td>73.2</td></tr><tr><td>o1 + GUI-Actor</td><td>7.5</td><td>17.7</td><td>68.2</td></tr><tr><td>o3 + GUI-Actor</td><td>12.1</td><td>27.4</td><td>74.0</td></tr><tr><td>o4-mini + GUI-Actor</td><td>8.4</td><td>20.0</td><td>65.1</td></tr><tr><td rowspan="3">VLM + UI-TARS</td><td>GPT-40 + UI-TARS</td><td>6.5</td><td>20.8</td><td>67.3</td></tr><tr><td>Qwen 2.5 72B + UI-TARS</td><td>7.5</td><td>21.8</td><td>63.2</td></tr><tr><td>Gemini 2.5 Flash + UI-TARS</td><td>9.3</td><td>24.1</td><td>70.2</td></tr><tr><td rowspan="5">LRM + UI-TARS</td><td>Claude 3.7 Sonnet + UI-TARS</td><td>9.3</td><td>17.5</td><td>61.5</td></tr><tr><td>Gemini 2.5 Pro + UI-TARS</td><td>7.5</td><td>23.4</td><td>71.6</td></tr><tr><td>o1 + UI-TARS</td><td>6.5</td><td>18.5</td><td>66.0</td></tr><tr><td>o3 + UI-TARS</td><td>12.1</td><td>27.2</td><td>72.4</td></tr><tr><td>o4-mini + UI-TARS</td><td>7.5</td><td>19.4</td><td>62.5</td></tr></table>
<table><tbody><tr><td rowspan="3">VLM + GTA-1</td><td>GPT-4o + GTA-1</td><td>8.4</td><td>23.5</td><td>72.7</td></tr><tr><td>Qwen 2.5 72B + GTA-1</td><td>9.3</td><td>24.3</td><td>69.0</td></tr><tr><td>Gemini 2.5 Flash + GTA-1</td><td>11.2</td><td>26.9</td><td>75.4</td></tr><tr><td rowspan="5">LRM + GTA-1</td><td>Claude 3.7 Sonnet + GTA-1</td><td>12.1</td><td>26.7</td><td>68.8</td></tr><tr><td>Gemini 2.5 Pro + GTA-1</td><td>8.4</td><td>23.5</td><td>74.5</td></tr><tr><td>o1 + GTA-1</td><td>7.5</td><td>21.1</td><td>73.1</td></tr><tr><td>o3 + GTA-1</td><td>14.0</td><td>28.7</td><td>76.7</td></tr><tr><td>o4-mini + GTA-1</td><td>10.3</td><td>21.7</td><td>67.1</td></tr><tr><td rowspan="3">VLM + <br/> GUI-Actor</td><td>GPT-40 + GUI-Actor</td><td>6.5</td><td>18.0</td><td>67.0</td></tr><tr><td>Qwen 2.5 72B + GUI-Actor</td><td>9.3</td><td>21.4</td><td>64.9</td></tr><tr><td>Gemini 2.5 Flash + GUI-Actor</td><td>10.3</td><td>25.6</td><td>73.1</td></tr><tr><td rowspan="5">LRM + GUI-Actor</td><td>Claude 3.7 Sonnet+ GUI-Actor</td><td>7.5</td><td>18.5</td><td>63.9</td></tr><tr><td>Gemini 2.5 Pro + GUI-Actor</td><td>9.3</td><td>24.0</td><td>73.2</td></tr><tr><td>o1 + GUI-Actor</td><td>7.5</td><td>17.7</td><td>68.2</td></tr><tr><td>o3 + GUI-Actor</td><td>12.1</td><td>27.4</td><td>74.0</td></tr><tr><td>o4-mini + GUI-Actor</td><td>8.4</td><td>20.0</td><td>65.1</td></tr><tr><td rowspan="3">VLM + UI-TARS</td><td>GPT-40 + UI-TARS</td><td>6.5</td><td>20.8</td><td>67.3</td></tr><tr><td>Qwen 2.5 72B + UI-TARS</td><td>7.5</td><td>21.8</td><td>63.2</td></tr><tr><td>Gemini 2.5 Flash + UI-TARS</td><td>9.3</td><td>24.1</td><td>70.2</td></tr><tr><td rowspan="5">LRM + UI-TARS</td><td>Claude 3.7 Sonnet + UI-TARS</td><td>9.3</td><td>17.5</td><td>61.5</td></tr><tr><td>Gemini 2.5 Pro + UI-TARS</td><td>7.5</td><td>23.4</td><td>71.6</td></tr><tr><td>o1 + UI-TARS</td><td>6.5</td><td>18.5</td><td>66.0</td></tr><tr><td>o3 + UI-TARS</td><td>12.1</td><td>27.2</td><td>72.4</td></tr><tr><td>o4-mini + UI-TARS</td><td>7.5</td><td>19.4</td><td>62.5</td></tr></tbody></table>


Table 3: Model Performance for pairing GTA-1, GUI-Actor and UI-TARS with all LRMs & VLMs, including task success rate, average progress, and step accuracy. All results are in %.
表 3：GTA-1、GUI-Actor 及 UI-TARS 搭配所有 LRM 与 VLM 的模型性能，包括任务成功率、平均进度和步骤准确度。所有结果均以 % 计。


<table><tr><td>Input Transcript</td><td>Task Success</td><td>Progress</td><td>Step Accuracy</td></tr><tr><td>Ground Truth</td><td>10.3</td><td>21.7</td><td>66.4</td></tr><tr><td>Whisper Large-V3</td><td>8.4</td><td>20.9</td><td>65.5</td></tr><tr><td>Whisper Large-V3 (Noise)</td><td>6.5</td><td>20.6</td><td>63.4</td></tr></table>
<table><tbody><tr><td>输入转录</td><td>任务成功率</td><td>进度</td><td>步骤准确率</td></tr><tr><td>地面真值</td><td>10.3</td><td>21.7</td><td>66.4</td></tr><tr><td>Whisper Large-V3</td><td>8.4</td><td>20.9</td><td>65.5</td></tr><tr><td>Whisper Large-V3 (噪声)</td><td>6.5</td><td>20.6</td><td>63.4</td></tr></tbody></table>


Table 4: Performance of GPT-40 + UGround-V1 using (1) ground-truth transcripts, (2) transcripts generated from original user speech by Whisper Large-V3, and (3) transcripts generated from noisy speech by Whisper Large-V3.
表 4：GPT-40 + UGround-V1 在以下情况的性能：(1) 真值转录，(2) 由 Whisper Large-V3 从原始用户语音生成的转录，以及 (3) 由 Whisper Large-V3 从噪声语音生成的转录。


## User behaviors
## 用户行为


Figure 8 shows diverse user behaviors in RealWebAssist not present in previous benchmarks. We include a zip file of the live recordings (including audio) from which the examples are taken.
图 8 展示了 RealWebAssist 中以前的基准测试所不具备的多样化用户行为。我们附带了一个包含示例取自的现场录音（含音频）的 zip 文件。


Information seeking As Figure 8A shows, the user is seeking information from different aspects, like images and ratings, before they make the purchase decision.
信息寻求 如图 8A 所示，用户在做出购买决定之前，会从图片和评分等不同方面寻求信息。


Comparing different options Figure 8B shows the process of the user viewing two candidates and finally make the decision between them.
比较不同选项 图 8B 展示了用户查看两个备选项并最终在它们之间做出决定的过程。


Changing minds In Figure 8C, the user is searching for some immersive dining experience. They are checking different restaurants and frequently change their minds when they see more options.
改变主意 在图 8C 中，用户正在寻找某种沉浸式用餐体验。他们在查看不同的餐厅，并随着看到更多选项而频繁改变主意。


Trial-and-error As Figure 8D shows, the user has several unsuccessful attempts when searching for men's fashion week. They refer to previous searchs or initiate new ones to look for what they want.
反复尝试 如图 8D 所示，用户在搜索男装周时经历了多次失败的尝试。他们参考之前的搜索或发起新的搜索来寻找想要的内容。


These diverse behaviors increase the complexity of the web assistance: instead of clearly defined-goals, the user themselves are also actively collecting knowledge to make decisions, which requires web assistant to follow the user's mind and act accordingly.
这些多样的行为增加了网络辅助的复杂性：用户不再有明确定义的目标，而是在积极收集知识以做出决定，这要求网络助手紧随用户思路并做出相应反应。


<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/02/2026_02_18__00_22_27_5ca995.jpg"/>



Figure 7: Example of annotated bounding boxes for an instruction. The red boxes represent the correct bounding boxes. The user gave the instruction "Close all the tabs". For evaluation purposes, closing any of the tabs first is considered correct at each step,so all the x marks are labeled as correct at each step.
图 7：指令的标注边界框示例。红框代表正确的边界框。用户给出了“关闭所有标签页”的指令。出于评估目的，在每个步骤中先关闭任何一个标签页都被视为正确，因此所有 x 标记在每个步骤中都被标注为正确。


<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/02/2026_02_18__00_22_27_daf508.jpg"/>



Figure 8: Example of rich user behaviors in RealWebAssist.
图 8：RealWebAssist 中丰富的用户行为示例。


Task # Description
任务 # 描述


Buy a gift for each of my three friends with a budget of \$100
在 100 美元预算内为我的三位朋友各买一份礼物


Find and buy a birthday gift for a friend who loves tech, within a \$50 budget.
为一位热爱科技的朋友寻找并购买一份 50 美元以内的生日礼物。


3 Purchase a cute water bottle for everyday use, under \$15
3 购买一个日常使用的可爱水杯，15 美元以下


4 Compare different laptops and buy one with the best review
4 比较不同的笔记本电脑并购买一款好评最多的


5 Purchase three home workout items under \$75 and compare their reviews before buying.
5 购买三件 75 美元以下的家庭健身器材，并在购买前对比其评价。


6 Find and order a customized gift (e.g., engraved or personalized) for a friend's graduation under \$60.
6 为朋友的毕业典礼寻找并订购一份 60 美元以内的定制礼物（如刻字或个性化）。


7 Order a complete warm and durable winter outfit (jacket, gloves, and boots) under \$200.
7 订购一套 200 美元以内的保暖且耐穿的冬季全套装备（夹克、手套和靴子）。


8 Get two sets of reusable grocery bags under \$20 total, checking for durability and eco-friendliness.
8 购买两套总价 20 美元以内的重复使用购物袋，检查其耐用性和环保性。


9 Buy two wall paintings for a family house, one for a 13-year old boy, one for a 6-year old girl
9 为家庭住宅购买两幅壁画，一幅给 13 岁男孩，一幅给 6 岁女孩


0 Purchase a set of colorful coffee mugs under \$20 with fun designs
0 购买一套 20 美元以内的设计有趣的彩色咖啡杯


11 Buy a small easy-care indoor plant under \$15 and schedule delivery within three days
11 购买一盆 15 美元以内的易打理室内小植物，并安排在三天内送达


12 Get a colorful umbrella for under \$30, making sure it's big enough for two people
12 购买一把 30 美元以内的彩色雨伞，确保大到足以容纳两人


13 Buy a set of scented candles under \$25, ensuring they have good reviews for long-lasting fragrance.
13 购买一套 25 美元以内的香氛蜡烛，确保其长效留香评价良好。


14 Find and purchase a durable phone case under \$20 for an iPhone 14 Pro Max.
14 为 iPhone 14 Pro Max 寻找并购买一款 20 美元以内的耐用手机壳。


5 Order a cozy throw blanket under \$30, checking for softness and warmth.
5 订购一条 30 美元以内的舒适毛毯，检查其柔软度和保暖性。


Buy a set of three face masks (reusable & breathable) under \$15.
购买一套 15 美元以内的三只装面罩（可重复使用且透气）。


7 Get a wireless Bluetooth speaker under \$40 with good bass and waterproofing.
7 购买一个 40 美元以内的无线蓝牙扬声器，需具备良好低音和防水功能。


Order a set of noise-canceling earplugs under \$15, ensuring they're comfortable for sleep.
订购一套 15 美元以内的降噪耳塞，确保睡眠佩戴舒适。


19 Find and buy a compact travel pillow and eye mask set under \$30.
19 寻找并购买一套 30 美元以内的便携式旅行枕和眼罩套装。


20 Purchase a set of six kitchen towels under \$20 with high absorbency.
20 购买一套 20 美元以内的六条装高吸水性厨房毛巾。


21 Buy an adjustable desk lamp under \$35 with multiple brightness settings.
21 购买一盏 35 美元以内的可调节且带多种亮度设置的台灯。


22 Order a pack of 12 gel pens under \$15 in assorted colors with smooth writing.
22 订购一盒 15 美元以下、颜色丰富且书写流畅的 12 支装中性笔。


23 Purchase a waterproof picnic blanket under \$40, ensuring it's easy to fold and carry.
23 购买一条 40 美元以下、易折叠携带的防水野餐垫。


24 Buy a cute yet professional notebook under \$20 for journaling or work.
24 购买一本 20 美元以下、可爱且专业的笔记本，用于日记或工作。


Find and purchase a comfortable memory foam seat cushion under \$35 for long sitting hours.
寻找并购买一个 35 美元以下、适合久坐的舒适记忆棉座垫。


26 Order a set of reusable silicone food storage bags under \$25.
26 订购一套 25 美元以下的重复使用硅胶食品保鲜袋。


27 Buy a pair of comfy indoor slippers under \$30 with high reviews for warmth and durability.
27 购买一双 30 美元以下、保暖耐穿且好评如潮的舒适室内拖鞋。


28 Purchase a portable mini humidifier under \$40 with USB charging.
28 购买一个 40 美元以下、支持 USB 充电的便携式微型加湿器。


29 Order a stylish travel makeup bag under \$25, ensuring it has multiple compartments.
29 订购一个 25 美元以下、带多个隔层的时尚旅行化妆包。


30 Find and order a surprise gift box for a friend who enjoys skincare, under \$50.
30 为喜欢护肤的朋友寻找并订购一个 50 美元以下的惊喜礼盒。


31 Compare wireless earbuds and purchase the best-reviewed pair under \$100.
31 比较无线耳机并购买一对 100 美元以下、好评率最高的耳机。


32 Order a budget-friendly yet stylish smartwatch under \$75, ensuring good battery life.
32 订购一款 75 美元以下、长续航且平价时尚的智能手表。


33 Find and order a high-quality mechanical keyboard under \$120, comparing typing feel and reviews
33 寻找并订购一把 120 美元以下的高品质机械键盘，对比打字手感与评价。


34 Find and buy a useful desk gadget under \$40 for a friend who works from home
34 为居家办公的朋友寻找并购买一件 40 美元以下的实用办公桌小物。


35 Plan flights for a trip from US to Europe (at least two different countries) for 3 days, comparing different airlines to find the best deal.
35 规划一次从美国到欧洲（至少两个不同国家）的 3 天旅程，对比不同航空公司以寻找最优方案。


36 Plan a 5-day trip to Japan, booking both flights and hotels, taking into account customer reviews.
36 规划一次 5 天的日本之行，参考客户评价预订机票和酒店。


37 Book a hotel for a weekend trip for a good price near the beach within the country, making sure you can cancel the trip at any time
37 在国内海滨附近以优惠价格预订周末旅行酒店，确保可随时取消。


38 Plan a spontaneous weekend trip to a destination with cheap last-minute flights and good hotel deals, for hotel make sure it's comfortable enough.
38 计划一次说走就走的周末旅行，选择有廉价最后一刻机票和优质酒店优惠的目的地，确保酒店足够舒适。


39 Book a luxury hotel for a weekend at a city in the west US, pay attention to different services offered
39 预订美国西部城市的一周末奢华酒店，关注所提供的各类服务。


40 Plan a three-stop European trip in a single week, with flights and hotel for each place
40 计划一周内的欧洲三站式旅行，包含每处的机票和酒店。


41 Book hotel for a family tour of four to a kid-friendly destination, with a hotel offering family amenities and breakfast included.
41 为四人家庭游预订前往适合儿童目的地的酒店，酒店需提供家庭设施并含早餐。


42 Arrange a road trip across the US, booking rental cars and a mix of motels and boutique hotels along the route.
42 安排一次横跨美国的公路旅行，预订租赁汽车以及沿途的汽车旅馆与精品酒店。


43 Book a romantic beach getaway in Hawaii for two people, make sure it's close to beach and have sea view
43 为两人预订夏威夷浪漫海滨度假之旅，确保靠近海滩并拥有海景。


Task # Description
任务 # 描述


44 Plan a family Disney Cruise, securing flights to Port Canaveral and a hotel near the theme parks before sailing.
44 计划一次家庭迪士尼邮轮之旅，确保飞往卡纳维拉尔港的机票以及启航前主题公园附近的酒店。


45 Arrange a wine country getaway, booking flights to Napa Valley, a rental car, and a vineyard hotel with wine-tasting experiences.
45 安排一次葡萄酒乡度假，预订飞往纳帕谷的机票、租赁汽车以及提供品酒体验的庄园酒店。


46 Find flights and a convertible rental car for a coastal drive in Hawaii, staying in beachfront resorts along the way.
46 为夏威夷沿海自驾寻找机票和敞篷租赁车，沿途入住海滨度假村。


7 Choose flights to a popular ski destination and secure a lodge or hotel under \$150/night.
7 选择飞往热门滑雪胜地的航班，并确保入住每晚150美元以下的木屋或酒店。


48 Book last-minute flights and a centrally located hotel in a major US city, focusing on deals under \$100/night with great city landscape view.
48 预订美国主要城市的最后一刻机票和中心地段酒店，重点关注每晚100美元以下且拥有绝佳城市景观的优惠。


49 Secure round-trip flights to a scenic South American city and book a comfortable hotel near local attractions.
49 确保往返南美风景优美城市的机票，并预订当地景点附近的舒适酒店。


50 Pick flights from a major US airport to a warm city in Canada, with a hotel under \$100/night in the downtown area.
50 选择从美国主要机场飞往加拿大温暖城市的航班，并入住市中心每晚100美元以下的酒店。


51 Schedule flights and a boutique hotel stay in a city rich in history, aiming for under \$100/night in a central location.
51 安排飞往历史名城的航班并入住精品酒店，目标是中心位置且每晚100美元以下。


52 Arrange direct flights to a popular theme park region, booking a nearby hotel or hotel with easy transportation
52 安排飞往热门主题公园地区的直飞航班，预订附近的酒店或交通便利的酒店。


53 Schedule flights for a quick visit to a popular national park, booking a nearby lodge or hotel with scenic views.
53 安排前往热门国家公园速览的航班，预订风景优美的附近旅馆或酒店。


54 Book round-trip flights to a major Middle Eastern city and reserve a modern hotel near historic sites for under \$100/night
54 预订前往中东大城市的往返航班，并以每晚低于 $100 的价格预订历史遗迹附近的现代化酒店


55 Secure flights from the US to a tropical island, choosing a resort that offers water sports
55 预订从美国飞往热带岛屿的航班，选择一家提供水上运动的度假村


56 Find flights and a resort for a tropical vacation in Cancun, Mexico, focusing on all-inclusive options for relaxation
56 寻找前往墨西哥坎昆热带度假的航班和度假村，专注于全包式的放松选择


57 Book flights to Cairo for a 5-day trip, then pick a hotel with a direct view of the Pyramids and free breakfast included
57 预订前往开罗的 5 日游航班，选择一家能直观金字塔并含免费早餐的酒店


58 Book a solo retreat to Kyoto, Japan, selecting a traditional ryokan stay with an onsen and authentic Japanese breakfast.
58 预订去日本京都的单人静修之旅，选择一家带温泉和地道日式早餐的传统旅馆。


59 Buy tickets for 2 people to an NBA Basketball game next weekend.
59 购买下周末两人的 NBA 篮球赛门票。


60 Find and book tickets for a concert by a top artist in the nearest major city within the next three months.
60 搜索并预订未来三个月内最近大城市顶级艺术家的演唱会门票。


61 Search for a last-minute concert ticket and find the best available seat.
61 搜索最后时刻的演唱会门票，并找到最佳可用座位。


62 Book 3 tickets for a rivalry match between two major sports teams
62 预订 3 张两大著名运动队之间宿敌对决的比赛门票


63 Book 3 tickets for a unique or unusual event, such as a drag show, wrestling match, or haunted experience
63 预订 3 张独特或不寻常活动的门票，例如变装秀、摔跤比赛或闹鬼体验


64 Purchase four tickets for a Broadway musical happening next month, aiming for orchestra seats if possible.
64 购买四张下月百老汇音乐剧门票，尽可能选择池座。


65 Buy tickets for a family of 4 with 2 kids to a MLB game
65 为一家四口（含两名儿童）购买一张 MLB 比赛门票


66 Find and book tickets to a popular stand-up comedy show in a western big city for the upcoming weekend, prioritizing seats near the front.
66 搜索并预订本周末西部大城市的热门单口喜剧秀门票，优先选择靠前座位。


67 Locate discounted tickets for a live theater performance in California this weekend
67 寻找本周末加州现场戏剧表演的折扣票


68 Search for an NFL game next month and buy two tickets in a mid-priced seating section for some eastern teams
68 搜索下个月的 NFL 比赛，并为某些东部球队购买两张中等价位区域的门票


69 Identify and reserve tickets for a children's matinee performance at a local venue, comparing any available family packages or group discounts.
查询并预订当地场馆的儿童日场演出门票，并对比现有的家庭套票或团体折扣。


70 Secure seats for a must-see hockey match, comparing "Best Seat" options.
锁定一场必看曲棍球比赛的座位，并对比“最佳位置”选项。


71 Find tickets for a classical music or orchestra concert in the nearest major city next month, aiming for seats with a good view of the stage.
寻找下个月最近大城市举办的古典乐或管弦乐演唱会门票，目标是视野良好的座位。


72 Buy tickets for two people to an English Premier League soccer match in London city center next weekend.
购买下周末两张位于伦敦市中心的英超足球比赛门票。


73 Find and purchase tickets to a major electronic music festival in Las Vegas within the next two months.
查找并购买未来两个月内在拉斯维加斯举办的大型电子音乐节门票。


74 Book seats for a stand-up comedy show in downtown Chicago next month, make sure the location is in city center.
预订下个月芝加哥市中心的一场单口喜剧演出座位，确保地点位于市中心。


75 Search for tickets to a top-tier cricket match in Sydney next month, aiming for seats that offer a good view of the pitch
搜索下个月在悉尼举行的顶级板球比赛门票，目标是球场视野良好的座位。


76 Locate a family-friendly musical performance near your city for next month.
查找下个月在你所在城市附近举办的适合家庭观看的音乐演出。


<table><tr><td></td><td>Task # Description</td></tr><tr><td></td><td>Purchase two tickets to an upcoming rugby match in Dublin next month, making sure seats <br> are in a central section and remain under.</td></tr><tr><td></td><td>Find a highly rated ballet or opera production in Paris within the next two months, choose the <br> seat in the second floor if available</td></tr><tr><td></td><td>79 Find tickets to a major fashion event, such as a runway show or fashion week experience.</td></tr><tr><td></td><td>80 Look for tickets to a themed immersive dining experience (e.g., murder mystery dinner, <br> fantasy-inspired restaurant)</td></tr><tr><td></td><td>81 Book tickets for UEFA soccer game between two Spanish teams for the next week</td></tr><tr><td></td><td>82 Book a ticket for a rooftop movie screening or outdoor film festival in a major city.</td></tr><tr><td></td><td>Find tickets for an esports event and compare standard vs. premium seating options.</td></tr><tr><td></td><td>84 Book a ticket for a "silent disco" event in a city of your choice.</td></tr><tr><td></td><td>85 secure two tickets to a major MLB game in a well-known ballpark anywhere in the U.S. next <br> month, opting for seats along the first baseline.</td></tr><tr><td></td><td>5 Find and book tickets for a large-scale country music festival occurring in the southern U.S. <br> within the next two months, focusing on general admission passes.</td></tr><tr><td></td><td>87 Purchase seats for a top-tier college football rivalry game taking place within the next six <br> weeks, ensuring you can view the marching band's performance easily.</td></tr><tr><td></td><td>88 Reserve tickets to a major NHL match in the next two months, choosing seats close to the <br> ice.</td></tr><tr><td></td><td>89 Book passes for a nationally touring art exhibition or immersive art experience within the <br> next two months, ensuring weekend availability.</td></tr><tr><td></td><td>0 Secure seats for a top-rated Broadway musical in New York City, making sure the date aligns <br> with a Saturday evening performance.</td></tr><tr><td></td><td>1 Reserve a spot for a special museum or cultural center night event (e.g., "Night at the Mu- <br> seum" or themed after-hours) in a major U.S. city within the next two months.</td></tr><tr><td></td><td>92 Find the best deal on a new smartphone (latest model iPhone or Samsung)</td></tr><tr><td></td><td>93 Find the best dinner deal for two using food delivery apps</td></tr><tr><td></td><td>94 Purchase an outfit for a formal event within a \$150 budget</td></tr><tr><td></td><td>95 Buy a high-quality gaming chair for under \$250</td></tr><tr><td></td><td>96 Find and book the best available concert tickets for a top artist in your city</td></tr><tr><td></td><td>97 Book tickets for a live theater performance and find a pre-show dinner reservation</td></tr><tr><td></td><td>98 Plan a sports game outing for two within a \$150 budget</td></tr><tr><td></td><td>99 Plan a weekend getaway for two within a \$500 budget</td></tr><tr><td></td><td>00 Organize a one-day itinerary for a solo traveler in a major city</td></tr><tr><td></td><td>Compare car rental options for a 5-day road trip</td></tr><tr><td></td><td>02 Find and book a local escape room challenge for a group of four</td></tr><tr><td></td><td>3 Plan a movie night with discounted tickets and snacks</td></tr><tr><td></td><td>04 Find a highly-rated sushi restaurant and order a meal for delivery</td></tr><tr><td></td><td>Plan a surprise birthday dinner at a fine dining restaurant</td></tr><tr><td></td><td>06 Order a late-night snack under \$15 for delivery</td></tr><tr><td></td><td>Book a luxury hotel staycation for a weekend</td></tr></table>
<table><tbody><tr><td></td><td>任务 # 描述</td></tr><tr><td></td><td>购买两张下个月在都柏林举行的橄榄球比赛门票，确保座位<br/>位于中间区域且价格在预算内。</td></tr><tr><td></td><td>在未来两个月内找一场巴黎的高评分芭蕾舞或歌剧演出，如有票<br/>请选择二层座位</td></tr><tr><td></td><td>79 寻找大型时尚活动的门票，例如时装秀或时装周体验。</td></tr><tr><td></td><td>80 寻找主题沉浸式用餐体验的门票（如谋杀之谜晚餐、<br/>奇幻风格餐厅）</td></tr><tr><td></td><td>81 预订下周两支西班牙球队之间的欧足联足球比赛门票</td></tr><tr><td></td><td>82 预订一张大城市天台电影放映或户外电影节的门票。</td></tr><tr><td></td><td>寻找一场电子竞技赛事的门票，并对比标准座与高级座选项。</td></tr><tr><td></td><td>84 预订一张你所选城市的“静音迪斯科”活动门票。</td></tr><tr><td></td><td>85 下个月在美国境内著名球场预订两张大联盟棒球赛门票，<br/>选择沿一垒线的座位。</td></tr><tr><td></td><td>5 寻找并预订未来两个月内在美国南部举行的大型乡村音乐节门票，<br/>重点关注普通入场票。</td></tr><tr><td></td><td>87 购买未来六周内顶级大学橄榄球德比战的座位，<br/>确保可以轻松观看行进乐队的表演。</td></tr><tr><td></td><td>88 预订未来两个月内一场大型 NHL 比赛的门票，选择靠近<br/>冰面的座位。</td></tr><tr><td></td><td>89 预订未来两个月内全美巡回艺术展或沉浸式艺术体验的门票，<br/>确保在周末。</td></tr><tr><td></td><td>0 预订一张纽约市顶级百老汇音乐剧的门票，确保日期<br/>符合周六晚上的演出。</td></tr><tr><td></td><td>1 预订未来两个月内美国大城市特殊博物馆或文化中心夜晚活动的名额<br/>（如“博物馆奇妙夜”或主题加时活动）。</td></tr><tr><td></td><td>92 寻找新款智能手机（最新款 iPhone 或三星）的最佳优惠</td></tr><tr><td></td><td>93 使用外卖 App 寻找最佳的双人晚餐优惠</td></tr><tr><td></td><td>94 在 150 美元预算内购买一套参加正式活动的服装</td></tr><tr><td></td><td>95 以低于 250 美元的价格购买一把高质量电竞椅</td></tr><tr><td></td><td>96 寻找并预订你所在城市顶级艺人的最佳演唱会门票</td></tr><tr><td></td><td>97 预订一场现场剧院演出门票并预约赛前晚餐</td></tr><tr><td></td><td>98 在 150 美元预算内规划一次双人体育比赛观赛活动</td></tr><tr><td></td><td>99 在 500 美元预算内规划一次双人周末旅行</td></tr><tr><td></td><td>00 为大城市的一名独自旅行者安排一日行程</td></tr><tr><td></td><td>对比为期 5 天公路旅行的租车选项</td></tr><tr><td></td><td>02 寻找并预订一个本地四人组的密室逃脱挑战</td></tr><tr><td></td><td>3 使用折扣门票和零食规划一次电影之夜</td></tr><tr><td></td><td>04 寻找一家高评分寿司店并点外卖</td></tr><tr><td></td><td>在高级餐厅策划一场生日惊喜晚餐</td></tr><tr><td></td><td>06 订一份 15 美元以下的深夜外卖小吃</td></tr><tr><td></td><td>预订周末的豪华酒店宅度假</td></tr></tbody></table>


## Full List of Websites
## 网站完整列表


<table><tr><td>Name</td><td>URL</td><td>Task Type</td></tr><tr><td>ACL Festival</td><td>aclfestival.com</td><td>Entertainment</td></tr><tr><td>Amazon</td><td>amazon.com</td><td>Shopping</td></tr><tr><td>Ammoora</td><td>ammoora.com</td><td>Entertainment</td></tr><tr><td>Apple</td><td>apple.com</td><td>Shopping</td></tr><tr><td>Artechouse</td><td>artechouse.com</td><td>Entertainment</td></tr><tr><td>Atom Tickets</td><td>atomtickets.com</td><td>Entertainment</td></tr><tr><td>Best Buy</td><td>bestbuy.com</td><td>Shopping</td></tr><tr><td>Adidas Arena</td><td>billetterie.adidasarena.com</td><td>Entertainment</td></tr><tr><td>Broadway</td><td>broadway.com</td><td>Entertainment</td></tr><tr><td>Charm City Clue Room</td><td>charmcityclueroom.com</td><td>Entertainment</td></tr><tr><td>City Pass</td><td>citypass.com</td><td>Travel Planning</td></tr></table>
<table><tbody><tr><td>名称</td><td>URL</td><td>任务类型</td></tr><tr><td>ACL 音乐节</td><td>aclfestival.com</td><td>娱乐</td></tr><tr><td>亚马逊</td><td>amazon.com</td><td>购物</td></tr><tr><td>Ammoora</td><td>ammoora.com</td><td>娱乐</td></tr><tr><td>苹果</td><td>apple.com</td><td>购物</td></tr><tr><td>Artechouse</td><td>artechouse.com</td><td>娱乐</td></tr><tr><td>Atom Tickets</td><td>atomtickets.com</td><td>娱乐</td></tr><tr><td>百思买</td><td>bestbuy.com</td><td>购物</td></tr><tr><td>阿迪达斯竞技场</td><td>billetterie.adidasarena.com</td><td>娱乐</td></tr><tr><td>百老汇</td><td>broadway.com</td><td>娱乐</td></tr><tr><td>Charm City 密室逃脱</td><td>charmcityclueroom.com</td><td>娱乐</td></tr><tr><td>城市通票</td><td>citypass.com</td><td>旅行规划</td></tr></tbody></table>


<table><tr><td>Name</td><td>URL</td><td>Task Type</td></tr><tr><td>CN Tower</td><td>cntower.ca</td><td>Travel Planning</td></tr><tr><td>Colorado Tourism</td><td>colorado.com</td><td>Travel Planning</td></tr><tr><td>Corsair</td><td>corsair.com</td><td>Shopping</td></tr><tr><td>Coupon Follow</td><td>couponfollow.com</td><td>Shopping</td></tr><tr><td>Crave 4D</td><td>crave4d.com</td><td>Entertainment</td></tr><tr><td>Dine Immersive</td><td>dineimmersive.com</td><td>Food</td></tr><tr><td>Disney Cruise</td><td>disneycruise.disney.go.com</td><td>Travel Planning</td></tr><tr><td>DoorDash</td><td>doordash.com</td><td>Food</td></tr><tr><td>Drone and DSLR</td><td>droneandslr.com</td><td>Shopping</td></tr><tr><td>Enterprise</td><td>enterprise.com</td><td>Travel Planning</td></tr><tr><td>ESCharts</td><td>escharts.com</td><td>Entertainment</td></tr><tr><td>ETIX</td><td>etix.com</td><td>Entertainment</td></tr><tr><td>Eventbrite</td><td>eventbrite.com</td><td>Entertainment</td></tr><tr><td>Expedia</td><td>expedia.com</td><td>Travel Planning</td></tr><tr><td>Fashion Week Online</td><td>fashionweekonline.com</td><td>Entertainment</td></tr><tr><td>Fever Up</td><td>feverup.com</td><td>Entertainment</td></tr><tr><td>Google</td><td>google.com</td><td>Travel Planning</td></tr><tr><td>Google Maps</td><td>google.com/maps</td><td>Travel Planning</td></tr><tr><td>Live Nation</td><td>livenation.com</td><td>Entertainment</td></tr><tr><td>Library of Congress</td><td>loc.gov</td><td>Travel Planning</td></tr><tr><td>LoL Esports</td><td>lolesports.com</td><td>Entertainment</td></tr><tr><td>MLB</td><td>mlb.com</td><td>Entertainment</td></tr><tr><td>MLB Tickets</td><td>mlb.tickets.com</td><td>Entertainment</td></tr><tr><td>NYICFF</td><td>nyicff.org</td><td>Entertainment</td></tr><tr><td>OpenTable</td><td>opentable.com</td><td>Food</td></tr><tr><td>Postmates</td><td>postmates.com</td><td>Food</td></tr><tr><td>Rakuten</td><td>rakuten.com</td><td>Shopping</td></tr><tr><td>Reddit</td><td>reddit.com</td><td>Entertainment</td></tr><tr><td>Retail Me Not</td><td>retailmenot.com</td><td>Shopping</td></tr><tr><td>Road Trip USA</td><td>roadtripusa.com</td><td>Travel Planning</td></tr><tr><td>Samsung</td><td>samsung.com</td><td>Shopping</td></tr><tr><td>San Lorenzo DC</td><td>sanlorenzodc.com</td><td>Food</td></tr><tr><td>Screen Daily</td><td>screendaily.com</td><td>Entertainment</td></tr><tr><td>Secret Baltimore</td><td>secretbaltimore.com</td><td>Travel Planning</td></tr><tr><td>Secret Lab</td><td>secretlab.co</td><td>Shopping</td></tr><tr><td>Smithsonian Sleepovers</td><td>smithsoniansleepovers.org</td><td>Entertainment</td></tr><tr><td>StubHub</td><td>stubhub.com</td><td>Entertainment</td></tr><tr><td>The Bureau Fashion Week</td><td>thebureaufashionweek.com</td><td>Entertainment</td></tr><tr><td>The Meltdown</td><td>themeltdown.com</td><td>Entertainment</td></tr><tr><td>The UFL</td><td>theufl.com</td><td>Entertainment</td></tr><tr><td>Ticketmaster</td><td>ticketmaster.com</td><td>Entertainment</td></tr><tr><td>Ticketmaster France</td><td>ticketmaster.fr</td><td>Entertainment</td></tr><tr><td>Ticket Web</td><td>ticketweb.com</td><td>Entertainment</td></tr><tr><td>TickPick</td><td>tickpick.com</td><td>Entertainment</td></tr><tr><td>TripAdvisor</td><td>tripadvisor.com</td><td>Travel Planning</td></tr><tr><td>Two Step Inn</td><td>twostepinn.com</td><td>Entertainment</td></tr><tr><td>Two Step Inn Frontgate</td><td>twostepinn.frontgatetickets.com</td><td>Entertainment</td></tr><tr><td>Uber</td><td>uber.com</td><td>Travel Planning</td></tr><tr><td>Uber Eats</td><td>ubereats.com</td><td>Food</td></tr><tr><td>Viator</td><td>viator.com</td><td>Travel Planning</td></tr><tr><td>Vivid Seats</td><td>vividseats.com</td><td>Entertainment</td></tr><tr><td>Washington Tourism</td><td>washington.org</td><td>Travel Planning</td></tr><tr><td>Yelp</td><td>yelp.com</td><td>Food</td></tr><tr><td>Zara</td><td>zara.com</td><td>Shopping</td></tr></table>
<table><tbody><tr><td>名称</td><td>URL</td><td>任务类型</td></tr><tr><td>加拿大国家电视塔</td><td>cntower.ca</td><td>旅行规划</td></tr><tr><td>科罗拉多州旅游局</td><td>colorado.com</td><td>旅行规划</td></tr><tr><td>美商海盗船</td><td>corsair.com</td><td>购物</td></tr><tr><td>Coupon Follow</td><td>couponfollow.com</td><td>购物</td></tr><tr><td>Crave 4D</td><td>crave4d.com</td><td>娱乐</td></tr><tr><td>Dine Immersive</td><td>dineimmersive.com</td><td>美食</td></tr><tr><td>迪士尼邮轮</td><td>disneycruise.disney.go.com</td><td>旅行规划</td></tr><tr><td>DoorDash</td><td>doordash.com</td><td>美食</td></tr><tr><td>Drone and DSLR</td><td>droneandslr.com</td><td>购物</td></tr><tr><td>企业租车</td><td>enterprise.com</td><td>旅行规划</td></tr><tr><td>ESCharts</td><td>escharts.com</td><td>娱乐</td></tr><tr><td>ETIX</td><td>etix.com</td><td>娱乐</td></tr><tr><td>Eventbrite</td><td>eventbrite.com</td><td>娱乐</td></tr><tr><td>Expedia</td><td>expedia.com</td><td>旅行规划</td></tr><tr><td>在线时装周</td><td>fashionweekonline.com</td><td>娱乐</td></tr><tr><td>Fever Up</td><td>feverup.com</td><td>娱乐</td></tr><tr><td>Google</td><td>google.com</td><td>旅行规划</td></tr><tr><td>谷歌地图</td><td>google.com/maps</td><td>旅行规划</td></tr><tr><td>Live Nation</td><td>livenation.com</td><td>娱乐</td></tr><tr><td>美国国会图书馆</td><td>loc.gov</td><td>旅行规划</td></tr><tr><td>英雄联盟电竞</td><td>lolesports.com</td><td>娱乐</td></tr><tr><td>MLB</td><td>mlb.com</td><td>娱乐</td></tr><tr><td>MLB 门票</td><td>mlb.tickets.com</td><td>娱乐</td></tr><tr><td>纽约国际儿童电影节</td><td>nyicff.org</td><td>娱乐</td></tr><tr><td>OpenTable</td><td>opentable.com</td><td>美食</td></tr><tr><td>Postmates</td><td>postmates.com</td><td>美食</td></tr><tr><td>乐天</td><td>rakuten.com</td><td>购物</td></tr><tr><td>Reddit</td><td>reddit.com</td><td>娱乐</td></tr><tr><td>Retail Me Not</td><td>retailmenot.com</td><td>购物</td></tr><tr><td>美国公路旅行</td><td>roadtripusa.com</td><td>旅行规划</td></tr><tr><td>三星</td><td>samsung.com</td><td>购物</td></tr><tr><td>San Lorenzo DC</td><td>sanlorenzodc.com</td><td>美食</td></tr><tr><td>Screen Daily</td><td>screendaily.com</td><td>娱乐</td></tr><tr><td>神秘巴尔的摩</td><td>secretbaltimore.com</td><td>旅行规划</td></tr><tr><td>Secret Lab</td><td>secretlab.co</td><td>购物</td></tr><tr><td>史密森尼博物馆奇妙夜</td><td>smithsoniansleepovers.org</td><td>娱乐</td></tr><tr><td>StubHub</td><td>stubhub.com</td><td>娱乐</td></tr><tr><td>局内人时装周</td><td>thebureaufashionweek.com</td><td>娱乐</td></tr><tr><td>The Meltdown</td><td>themeltdown.com</td><td>娱乐</td></tr><tr><td>UFL 联盟</td><td>theufl.com</td><td>娱乐</td></tr><tr><td>Ticketmaster</td><td>ticketmaster.com</td><td>娱乐</td></tr><tr><td>Ticketmaster 法国</td><td>ticketmaster.fr</td><td>娱乐</td></tr><tr><td>Ticket Web</td><td>ticketweb.com</td><td>娱乐</td></tr><tr><td>TickPick</td><td>tickpick.com</td><td>娱乐</td></tr><tr><td>猫途鹰</td><td>tripadvisor.com</td><td>旅行规划</td></tr><tr><td>Two Step Inn 音乐节</td><td>twostepinn.com</td><td>娱乐</td></tr><tr><td>Two Step Inn Frontgate 票务</td><td>twostepinn.frontgatetickets.com</td><td>娱乐</td></tr><tr><td>优步</td><td>uber.com</td><td>旅行规划</td></tr><tr><td>优食</td><td>ubereats.com</td><td>美食</td></tr><tr><td>Viator</td><td>viator.com</td><td>旅行规划</td></tr><tr><td>Vivid Seats</td><td>vividseats.com</td><td>娱乐</td></tr><tr><td>华盛顿旅游局</td><td>washington.org</td><td>旅行规划</td></tr><tr><td>Yelp</td><td>yelp.com</td><td>美食</td></tr><tr><td>飒拉</td><td>zara.com</td><td>购物</td></tr></tbody></table>


## Word Frequency
## 词频


Figure 9 compares the most frequent instruction words in RealWebAssist with those from two common benchmarks, WebLINX and WebArena. The vocabulary used in RealWe-bAssist is more informal, as the dataset comes from natural spoken instructions. The tone is also more informal and conversational compared to WebLINX and WebArena.
图 9 将 RealWebAssist 中最常出现的指令词与两个常用基准测试 WebLINX 和 WebArena 进行了对比。由于数据集源自自然口语指令，RealWebAssist 使用的词汇更为非正式。与 WebLINX 和 WebArena 相比，其语气也更具非正式感和对话感。


<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/02/2026_02_18__00_22_27_9e3223.jpg"/>



Figure 9: Word Cloud of the most frequent words in Re-alWebAssist v.s. common benchmarks WebLINX and We-bArena.
图 9：RealWebAssist 与常用基准测试 WebLINX 及 WebArena 的高频词词云图对比。


## Instructions for the participants
## 参与者指南


Thank you for participating in our study! You'll be guiding another person who is controlling the computer on your behalf. Imagine you are helping a friend navigate a website remotely, giving step-by-step instructions to complete a task. Feel free to interpret the task as you see fit. Here are some guidelines to keep in mind:
感谢您参与我们的研究！您将引导另一位代您操作电脑的人员。请想象您正在远程帮助一位朋友浏览网站，通过逐步指令引导其完成任务。您可以根据自己的理解来解读任务。请记住以下准则：


- Give instructions as naturally as possible, just like you would in real life.
- 尽可能自然地给出指令，就像在现实生活中一样。


- You don't have to be overly precise-say what feels natural.
- 无需追求过度精准——说出您觉得自然的话即可。


- You can only give one instruction at a time. After the operator follows your instruction, wait for them to complete it before giving the next step.
- 每次只能给出一个指令。在操作员执行您的指令后，请等待其完成后再进行下一步。


- Keep your instructions clear and concise, but don't stress too much about exact wording—just say what comes to mind!
- 保持指令清晰简洁，但不必过于纠结措辞——想到什么就说什么！


- You are allowed to instruct the operator to use Google to search for things.
- 您可以指示操作员使用 Google 搜索信息。


## Video Example
## 视频示例


A sample raw recording can be viewed via the link below (audio included)
可通过下方链接查看原始录制样本（含音频）


https://youtu.be/CcyIt9tr5qo
