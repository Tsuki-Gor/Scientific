<!-- Meanless: JOURNAL OF LATEX CLASS FILES, DECEMBER 2024<br>SURECTICALLYS -->

# Large Language Model-Brained GUI Agents: A Survey

Chaoyun Zhang, Shilin He, Jiaxu Qian, Bowen Li, Liqun Li, Si Qin, Yu Kang, Minghua Ma, Guyue Liu, Qingwei Lin, Saravan Rajmohan, Dongmei Zhang, Qi Zhang

Abstract-Graphical User Interfaces (GUIs) have long been central to human-computer interaction, providing an intuitive and visually-driven way to access and interact with digital systems. Traditionally, automating GUI interactions relied on script-based or rule-based approaches, which, while effective for fixed workflows, lacked the flexibility and adaptability required for dynamic, real-world applications. The advent of Large Language Models (LLMs), particularly multimodal models, has ushered in a new era of GUI automation. They have demonstrated exceptional capabilities in natural language understanding, code generation, task generalization, and visual processing. This has paved the way for a new generation of "LLM-brained" GUI agents capable of interpreting complex GUI elements and autonomously executing actions based on natural language instructions. These agents represent a paradigm shift, enabling users to perform intricate, multi-step tasks through simple conversational commands. Their applications span across web navigation, mobile app interactions, and desktop automation, offering a transformative user experience that revolutionizes how individuals interact with software. This emerging field is rapidly advancing, with significant progress in both research and industry.

To provide a structured understanding of this trend, this paper presents a comprehensive survey of LLM-brained GUI agents, exploring their historical evolution, core components, and advanced techniques. We address critical research questions such as existing GUI agent frameworks, the collection and utilization of data for training specialized GUI agents, the development of large action models tailored for GUI tasks, and the evaluation metrics and benchmarks necessary to assess their effectiveness. Additionally, we examine emerging applications powered by these agents. Through a detailed analysis, this survey identifies key research gaps and outlines a roadmap for future advancements in the field. By consolidating foundational knowledge and state-of-the-art developments, this work aims to guide both researchers and practitioners in overcoming challenges and unlocking the full potential of LLM-brained GUI agents. We anticipate that this survey will serve both as a practical cookbook for constructing LLM-powered GUI agents, and as a definitive reference for advancing research in this rapidly evolving domain.

The collection of papers reviewed in this survey will be hosted and regularly updated on the GitHub repository: https://github.com/vyokky/LLM-Brained-GUI-Agents-Survey Additionally, a searchable webpage is available at https://aka.ms/gui-agent for easier access and exploration.

Index Terms-Large Language Model, Graphical User Interface, AI Agent, Automation, Human-Computer Interaction

## 1 INTRODUCTION

Graphical User Interfaces (GUIs) have been a cornerstone of human-computer interaction, fundamentally transforming how users navigate and operate within digital systems [1]. Designed to make computing more intuitive and accessible, GUIs replaced command-line interfaces (CLIs) [2] with visually driven, user-friendly environments. Through the use of icons, buttons, windows, and menus, GUIs empowered a broader range of users to interact with computers using simple actions such as clicks, typing, and gestures. This shift democratized access to computing, allowing even non-technical users to effectively engage with complex systems. However, GUIs often sacrifice efficiency for usability, particularly in workflows requiring repetitive or multi-step interactions, where CLIs can remain more streamlined [3].

While GUIs revolutionized usability, their design, primarily tailored for human visual interaction, poses significant challenges for automation. The diversity, dynamism, and platform-specific nature of GUI layouts make it difficult to develop flexible and intelligent automation tools capable of adapting to various environments. Early efforts to automate GUI interactions predominantly relied on script-based or rule-based methods [4], [5]. Although effective for predefined workflows, these methods were inherently narrow in scope, focusing primarily on tasks such as software testing and robotic process automation (RPA) [6]. Their rigidity required frequent manual updates to accommodate new tasks, changes in GUI layouts, or evolving workflows, limiting their scalability and versatility. Moreover, these approaches lacked the sophistication needed to support dynamic, human-like interactions, thereby constraining their applicability in complex or unpredictable scenarios.

The rise of Large Language Models (LLMs) [8, 9], especially those augmented with multimodal capabilities [10], has emerged as a game changer for GUI automation, redefining the the way agents interact with graphical user interfaces. Beginning with models like ChatGPT [11], LLMs have demonstrated extraordinary proficiency in natural language understanding, code generation, and generalization across diverse tasks [8], 12-14. The integration of visual language models (VLMs) has further extended these capabilities, enabling these models to process visual data, such as the intricate layouts of GUIs [15]. This evolution bridges the gap between linguistic and visual comprehension, empowering intelligent agents to interact with GUIs in a more human-like and adaptive manner. By leveraging these advancements, LLMs and VLMs offer transformative potential, enabling agents to navigate complex digital environments, execute tasks dynamically, and revolutionize the field of GUI automation.

---

<!-- Footnote -->

Version: v8 (major update on May 2, 2025)

Chaoyun Zhang, Shilin He, Jiaxu Qian, Liqun Li, Si Qin, Yu Kang, Minghua Ma, Qingwei Lin, Saravan Rajmohan, Dongmei Zhang and Qi Zhang are with Microsoft. e-mail: \{chaoyun.zhang, shilin.he, v-jiaxuqian, liqun.li, si.gin, yu.kang, minghuama, qlin, saravan.rajmohan, dongmeiz, zhang.qi\}@microsoft.com.

Bowen Li is with Shanghai Artificial Intelligence Laboratory, China. e-mail: libowen.ne@gmail.com.

Guyue Liu is with Peking University, China. e-mail: guyue.liu@gmail.com.For any inquiries or discussions, please contact Chaoyun Zhang and Shilin He.

1. By LLMs, we refer to the general concept of foundation models capable of accepting various input modalities (e.g., visual language models (VLMs), multimodal LLMs (MLLMs)) while producing output exclusively in textual sequences [7].

<!-- Footnote -->

---


<!-- Meanless: JOURNAL OF LATEX CLASS FILES, DECEMBER 2024<br>2 -->

<!-- Media -->

<!-- figureText: User Request<br>GUI Agent<br>Word<br>Photos<br>Web Browser<br>Eiten discussing thoroughly wit<br>：<br>my co-workers, we are<br>convinced that ASSISTANT 2<br>extract<br>Summarize<br>Observe<br><icon/><br>Multiagent LLMs<br>Creation<br>通工厂<br>... ...<br>Recap: What is LLM-based Agent? - -<br>Auto-Agent Profiling<br>② 画<br>Send<br>3.<br>... $2{t}^{2} - 1$<br>Teams<br>Click<br>PowerPoint<br>Adobe Acrobat -->

<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2025/11/2025_11_04__12_39_04_fa085c.jpg"/>

Fig. 1: Illustration of the high-level concept of an LLM-powered GUI agent. The agent receives a user's natural language request and orchestrates actions seamlessly across multiple applications. It extracts information from Word documents, observes content in Photos, summarizes web pages in the browser, reads PDFs in Adobe Acrobat, and creates slides in PowerPoint before sending them through Teams.

<!-- Media -->

### 1.1 Motivation for LLM-Brained GUI agents

With an LLM serving as its "brain", LLM-powered GUI automation introduces a new class of intelligent agents capable of interpreting a user's natural language requests, analyzing GUI screens and their elements, and autonomously executing appropriate actions. Importantly, these capabilities are achieved without reliance on complex, platform-specific scripts or predefined workflows. These agents, referred to as "LLM-brained GUI agents", can be formally defined as:

Intelligent agents that operate within GUI environments, leveraging LLMs as their core inference and cognitive engine to generate, plan, and execute actions in a flexible and adaptive manner.

This paradigm represents a transformative leap in GUI automation, fostering dynamic, human-like interactions across diverse platforms. It enables the creation of intelligent, adaptive systems that can reason, make decisions in real-time, and respond flexibly to evolving tasks and environments. We illustrate this high-level concept in Figure 1

Traditional GUI automation are often limited by predefined rules or narrowly focused on specific tasks, constraining their ability to adapt to dynamic environments and diverse applications. In contrast, LLM-powered GUI agents bring a paradigm shift by integrating natural language understanding, visual recognition, and decision-making into a unified framework. This enables them to generalize across a wide range of use cases, transforming task automation and significantly enhancing the intuitiveness and efficiency of human-computer interaction. Moreover, unlike the emerging trend of pure Application Programming Interface (API)-based agents—which depend on APIs that may not always be exposed or accessible-GUI agents leverage the universal nature of graphical interfaces. GUIs offer a general mechanism to control most software applications, enabling agents to operate in a nonintrusive manner without requiring internal API access. This capability not only broadens the applicability of GUI agents but also empowers external developers to build advanced functionality on top of existing software across diverse platforms and ecosystems. Together, these innovations position GUI agents as a versatile and transformative technology for the future of intelligent automation.


<!-- Meanless: JOURNAL OF IATEX CLASS FILES, DECEMBER 2024<br>3 -->

This new paradigm enables users to control general software systems with conversational commands [16]. By reducing the cognitive load of multi-step GUI operations, LLM-powered agents make complex systems accessible to non-technical users and streamline workflows across diverse domains. Notable examples include SeeAct [17] for web navigation, AppAgent [18] for mobile interactions, and UFO [19] for Windows OS applications. These agents resemble a "virtual assistant" [20] akin to J.A.R.V.I.S. from Iron Man-an intuitive, adaptive system capable of understanding user goals and autonomously performing actions across applications. The futuristic concept of an AI-powered operating system that executes cross-application tasks with fluidity and precision is rapidly becoming a reality [21], [22].

Real-world applications of LLM-powered GUI agents are already emerging. For example, Microsoft Power Automate utilizes LLMs to streamline low-code/no-code automation ${}^{2}$ . allowing users to design workflows across Microsoft applications with minimal technical expertise. Integrated AI assistants in productivity software,like Microsoft Copilot ${}^{3}$ are bridging the gap between natural language instructions and operations on application. Additionally, LLM-powered agents show promise for enhancing accessibility [23], potentially allowing visually impaired users to navigate GUIs more effectively by converting natural language commands into executable steps. These developments underscore the timeliness and transformative potential of LLM-powered GUI agents across diverse applications.

The convergence of LLMs and GUI automation addresses longstanding challenges in human-computer interaction and introduces new opportunities for intelligent GUI control [24]. This integration has catalyzed a surge in research activity, spanning application frameworks [19], data collection [25], model optimization [15], and evaluation benchmarks [26]. Despite these advancements, key challenges and limitations persist, and many foundational questions remain unexplored. However, a systematic review of this rapidly evolving area is notably absent, leaving a critical gap in understanding.

### 1.2 Scope of the Survey

To address this gap, this paper provides a pioneering, comprehensive survey of LLM-brained GUI agents. We cover the historical evolution of GUI agents, provide a step-by-step guide to building these agents, summarize essential and advanced techniques, review notable tools and research related to frameworks, data and models, showcase representative applications, and outline future directions. Specifically, this survey aims to answer the following research questions (RQs):

1) RQ1: What is the historical development trajectory of LLM-powered GUI agents? (Section 4)

2) RQ2: What are the essential components and advanced technologies that form the foundation of LLM-brained GUI agents? (Section 5)

3) RQ3: What are the principal frameworks for LLM GUI agents, and what are their defining characteristics? (Section 6)

4) RQ4: What are the existing datasets, and how can comprehensive datasets be collected to train optimized LLMs for GUI agents? (Section 7)

5) RQ5: How can the collected data be used to train purpose-built Large Action Models (LAMs) for GUI agents, and what are the current leading models in the field? (Section 8)

6) RQ6: What metrics and benchmarks are used to evaluate the capability and performance of GUI agents? (Section 9)

7) RQ7: What are the most significant real-world applications of LLM-powered GUI agents, and how have they been adapted for practical use? (Section 10)

8) RQ8: What are the major challenges, limitations, and future research directions for developing robust and intelligent GUI agents? (Section 11)

Through these questions, this survey aims to provide a comprehensive overview of the current state of the field, offer a guide for building LLM-brained GUI agents, identify key research gaps, and propose directions for future work. This survey is one of the pioneers to systematically examine the domain of LLM-brained GUI agents, integrating perspectives from LLM advancements, GUI automation, and human-computer interaction.

### 1.3 Survey Structure

The survey is organized as follows, with a structural illustration provided in Figure 2. Section 2 reviews related survey and review literature on LLM agents and GUI automation. Section 3 provides preliminary background on LLMs, LLM agents, and GUI automation. Section 2 traces the evolution of LLM-powered GUI agents. Section 5 introduces key components and advanced technologies within LLM-powered GUI agents, serving as a comprehensive guide. Section 6 presents representative frameworks for LLM-powered GUI agents. Section 7 discusses dataset collection and related data-centric research for optimizing LLMs in GUI agent. Section 8 covers foundational and optimized models for GUI agents. Section 9 outlines evaluation metrics and benchmarks. Section 10 explores real-world applications and use cases. Finally, Section 11 examines current limitations, challenges, and potential future directions, and section 12 conclude this survey. For clarity, a list of abbreviations is provided in Table 1

## 2 RELATED WORK

The integration of LLMs with GUI agents is an emerging and rapidly growing field of research. Several related surveys and tutorials provide foundational insights and guidance. We provide a brief review of existing overview articles on GUI automation and LLM agents, as these topics closely relate to and inform our research focus. To begin, we provide an overview of representative surveys and books on GUI automation, LLM agents, and their integration, as summarized in Table 2 These works either directly tackle one or two core areas in GUI automation and LLM-driven agents, or provide valuable insights that, while not directly addressing the topic, contribute indirectly to advancing the field. GUI agents, application UI screenshots are equally essential, serving as key inputs for reliable task comprehension and execution.

---

<!-- Footnote -->

2. https://www.microsoft.com/en-us/power-platform/blog/

power-automate/revolutionize-the-way-you-work-with-automation-and-ai/ 3. https://copilot.microsoft.com/

<!-- Footnote -->

---


<!-- Media -->

<!-- figureText: 1.1 Motivation for LLM-Brained GUI Agents<br>1 Introduction<br>1.2 Scope of the Survey<br>1.3 Survey Structure<br>2.1 Surveys on LLM Agents<br>2 Related Work<br>2.2 Survey on GUI Automation<br>3.1 Large Language Models: Foundations and Capabilities<br>3 Background<br>2 LLM Agents: From Language to Action<br>3.3 GUI Automation: Tools, Techniques, and Challenges<br>4.1 Early Automation Systems<br>4.1.1 Random-Based Automation<br>4.1.3 Rule-Based Automation<br>4.1.3 Script-Based Automation<br>4.1.4 Tools and Software<br>4 Evolution and Progression of LLM-Brained GUI Agents<br>4.2.1 Machine Learning and Computer Vision<br>2 The Shift Towards Intelligent Agents<br>4.2.2 Natural Language Processing<br>4.2.3 Reinforcement Learning<br>4.3.1 Web Domain<br>4.3.2 Mobile Devices<br>4.3 The Advent of LLM-Brained GUI Agents<br>4.3.3 Computer Systems<br>4.4 GUI Agent vs. API-Based Agent<br>5.1 Architecture and Workflow in a Nutshell<br>5.2 Operating Environment<br>5.2.1 Platform<br>5.2.2 Environment State Perception<br>5.2.3 Environment Feedback<br>5.3 Prompt Engineering<br>5.4.1 Planning<br>5.4 Model Inference<br>5.4.2 Action Inference<br>5 LLM-Brained GUI Agents: Foundations and Design<br>5.4.3 Complementary Outputs<br>5.5.1 UI Operations<br>5.5.2 Native API Calls<br>5.5 Actions Execution<br>5.5.3 AI Tools<br>5.5.4 Summary<br>5.6.1 Short-Term Memory<br>5.6 Memory<br>5.6.2 Long-Term Memory<br>5.7 Advanced Enhancements<br>5.7.1 Computer Vision-Based GUI Grounding<br>5.8 From Foundations to Innovations: A Roadmap<br>5.7.2 Multi-Agent Framework<br>6.1 Web GUI Agents<br>5.7.3 Self-Reflection<br>Large Language Model-Brained GUI Agents<br>6.2 Mobile GUI Agents<br>5.7.4 Self-Evolution<br>5.7.5 Reinforcement Learning<br>6 LLM-Brained GUI Agent Framework<br>6.3 Computer GUI Agents<br>5.7.6 Summary & Takeaways<br>6.4 Cross-Platform GUI Agents<br>6.5 Takeaways<br>7.1.1 Data Composition and Sources<br>7.1 Data Collection<br>7.1.2 Collection Pipeline<br>7.2 Web Agent Data<br>7 Data for Optimizing LLM-Brained GUI Agents<br>7.3 Mobile Agent Data<br>7.4 Computer Agent Data<br>7.5 Cross-Platform Agent Data<br>7.6 Takeaways<br>3.1.1 Close-Source Models<br>8.1 Foundation Models<br>2 Open-Source Models<br>8.2 Large Action Models<br>8.3 LAMs for Web GUI Agents<br>8 Models for Optimizing LUM-Brained GUI Agents<br>8.4 LAMs for Mobile GUI Agents<br>8.5 LAMs for Computer GUI Agents<br>8.6 Cross-Platform Large Action Models<br>8.7 Takeaways<br>9.1 Evaluation Metrics<br>9.2 Evaluation Measurements<br>9.3 Evaluation Platforms<br>9.4 Web Agent Benchmark<br>9 Evaluation for LLM-Brained GUI Agents<br>9.5 Mobile Agent Benchmark<br>9.6 Computer Agent Benchmark<br>10.1.1 General Testing<br>9.7 Cross-Platform Benchmark<br>10.1.2 Text Input Generation<br>9.8 Takeaways<br>10.1.3 Bug Replay<br>10.1 GUI Testing<br>10.1.4 Verification<br>10 Applications of LLM-Brained GUI Agents<br>10.2 Virtual Assistants<br>10.2.1 Research<br>10.3 Takeaways<br>10.2.2 Open-Source Projects<br>11.1 Privacy Concerns<br>10.2.3 Production<br>11.2 Latency, Performance,<br>and Resource Constraints<br>11.3 Safety and Reliability<br>11 Limitations, Challenges and Future Roadmap<br>11.4 Human-Agent Interaction<br>11.5 Customization and Personalization<br>12 Conclusion<br>11.6 Ethical and Regulatory Challenges<br>11.7 Scalability and Generalization<br>11.8 Summary -->

<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2025/11/2025_11_04__12_39_04_d0dc4d.jpg"/>

Fig. 2: The structure of the survey on LLM-brained GUI agents.

<!-- Media -->


<!-- Meanless: 5<br>JOURNAL OF LATEX CLASS FILES, DECEMBER 2024 -->

<!-- Media -->

TABLE 1: List of abbreviations in alphabetical order.

<table><tr><td>Acronym</td><td>Explanation</td></tr><tr><td>AI</td><td>Artificial Intelligence</td></tr><tr><td>AITW</td><td>Android in the Wild</td></tr><tr><td>AITZ</td><td>Android in The Zoo</td></tr><tr><td>API</td><td>Application Programming Interface</td></tr><tr><td>CLI</td><td>Command-Line Interface</td></tr><tr><td>CLIP</td><td>Contrastive Language-Image Pre-Training</td></tr><tr><td>CoT</td><td>Chain-of-Thought</td></tr><tr><td>CSS</td><td>Cascading Style Sheets</td></tr><tr><td>CUA</td><td>Computer-Using Agent</td></tr><tr><td>CuP</td><td>Completion under Policy</td></tr><tr><td>CV</td><td>Computer Vision</td></tr><tr><td>DOM</td><td>Document Object Model</td></tr><tr><td>DPO</td><td>Direct Preference Optimization</td></tr><tr><td>GCC</td><td>General Computer Control</td></tr><tr><td>GPT</td><td>Generative Pre-trained Transformers</td></tr><tr><td>GUI</td><td>Graphical User Interface</td></tr><tr><td>HCI</td><td>Human-Computer Interaction</td></tr><tr><td>HTML</td><td>Hypertext Markup Language</td></tr><tr><td>ICL</td><td>In-Context Learning</td></tr><tr><td>IoU</td><td>Intersection over Union</td></tr><tr><td>LAM</td><td>Large Action Model</td></tr><tr><td>LLM</td><td>Large Language Model</td></tr><tr><td>LSTM</td><td>Long Short-Term Memory</td></tr><tr><td>LTM</td><td>Long-Term Memory</td></tr><tr><td>MCTS</td><td>Monte Carlo Tree Search</td></tr><tr><td>MoE</td><td>Mixture of Experts</td></tr><tr><td>MDP</td><td>Markov Decision Process</td></tr><tr><td>MLLM</td><td>Multimodal Large Language Model</td></tr><tr><td>OCR</td><td>Optical Character Recognition</td></tr><tr><td>OS</td><td>Operation System</td></tr><tr><td>RAG</td><td>Retrieval-Augmented Generation</td></tr><tr><td>ReAct</td><td>Reasoning and Acting</td></tr><tr><td>RL</td><td>Reinforcement Learning</td></tr><tr><td>RLHF</td><td>Reinforcement Learning from Human Feedback</td></tr><tr><td>RNN</td><td>Recurrent Neural Network</td></tr><tr><td>RPA</td><td>Robotic Process Automation</td></tr><tr><td>UI</td><td>User Interface</td></tr><tr><td>UX</td><td>User Experience</td></tr><tr><td>VAB</td><td>VisualAgentBench</td></tr><tr><td>VLM</td><td>Visual Language Models</td></tr><tr><td>ViT</td><td>Vision Transformer</td></tr><tr><td>VQA</td><td>Visual Question Answering</td></tr><tr><td>SAM</td><td>Segment Anything Model</td></tr><tr><td>SoM</td><td>Set-of-Mark</td></tr><tr><td>STM</td><td>Short-Trem Memory</td></tr></table>

<!-- Media -->

### 2.1 Survey on GUI Automation

GUI automation has a long history and wide applications in industry, especially in GUI testing [27]-[29] and RPA [6] for task automation [42].

Said et al., [30] provide an overview of GUI testing for mobile applications, covering objectives, approaches, and challenges within this domain. Focusing on Android applications, Li [31] narrows the scope further, while Oksanen et al., 32 explore automatic testing techniques for Windows GUI applications, a key platform for agent operations. Similarly, Moura et al., 72 review GUI testing for web applications, which involves diverse tools, inputs, and methodologies. Deshmukh et al., 33 discuss automated GUI testing for enhancing user experience, an area where LLMs also bring new capabilities. A cornerstone of modern GUI testing is computer vision (CV), which is used to interpret UI elements and identify actionable controls [34]. Yu et al., [35] survey the application of CV in mobile GUI testing, highlighting both its significance and associated challenges. In LLM-powered

On the other hand, RPA, which focuses on automating repetitive human tasks, also relies heavily on GUI automation for relevant processes. Syed et al., [36] review this field and highlight contemporary RPA themes, identifying key challenges for future research. Chakraborti et al., 37 emphasize the importance of shifting from traditional, script-based RPA toward more intelligent, adaptive paradigms, offering a systematic overview of advancements in this direction. Given RPA's extensive industrial applications, Enriquez et al., [38] and Ribeiro et al., 39 survey the field from an industrial perspective, underscoring its significance and providing a comprehensive overview of RPA methods, development trends, and practical challenges.

Both GUI testing [40] and RPA [41] continue to face significant challenges in achieving greater intelligence and robustness. LLM-powered GUI agents are poised to play a transformative role in these fields, providing enhanced capabilities and adding substantial value to address these persistent issues.

### 2.2 Surveys on LLM Agents

The advent of LLMs has significantly enhanced the capabilities of intelligent agents [43], enabling them to tackle complex tasks previously out of reach, particularly those involving natural language understanding and code generation [44]. This advancement has spurred substantial research into LLM-based agents designed for a wide array of applications [45].

Both Xie et al., [46] and Wang et al., [47] offer comprehensive surveys on LLM-powered agents, covering essential background information, detailed component breakdowns, taxonomies, and various applications. These surveys serve as valuable references for a foundational understanding of LLM-driven agents, laying the groundwork for further exploration into LLM-based GUI agents. Xie et al., [59] provide an extensive overview of multimodal agents, which can process images, videos, and audio in addition to text. This multimodal capability significantly broadens the scope beyond traditional text-based agents [60]. Notably, most GUI agents fall under this category, as they rely on image inputs, such as screenshots, to interpret and interact with graphical interfaces effectively. Multi-agent frameworks are frequently employed in the design of GUI agents to enhance their capabilities and scalability. Surveys by Guo et al., [48] and Han et al., [49] provide comprehensive overviews of the current landscape, challenges, and future directions in this area. Sun et al., [50] provide an overview of recent methods that leverage reinforcement learning to strengthen multi-agent LLM systems, opening new pathways for enhancing their capabilities and adaptability. These surveys offer valuable insights and guidance for designing effective multi-agent systems within GUI agent frameworks.

In the realm of digital environments, Wu et al., [61] presents a survey on LLM agents operating in mobile environments, covering key aspects of mobile GUI agents. In a boarder scope, Wang et al., [62] present a survey on the integration of foundation models with GUI agents. Another survey by Gao et al., provides an overview of autonomous


<!-- Meanless: 6 -->

JOURNAL OF LATEX CLASS FILES, DECEMBER 2024 agents operating across various digital platforms [63], highlighting their capabilities, challenges, and applications. All these surveys highlighting emerging trends in this area.

<!-- Media -->

TABLE 2: Summary of representative surveys and books on GUI automation and LLM agents. A √symbol indicates that a publication explicitly addresses a given domain, while an $○$ symbol signifies that the publication does not focus on the area but offers relevant insights. Publications covering both GUI automation and LLM agents are highlighted for emphasis.

<table><tr><td rowspan="2">Survey</td><td rowspan="2">One Sentence Summary</td><td colspan="3">Scope</td></tr><tr><td>GUI Automation</td><td>LLM Agent</td><td>LLM Agent + GUI Automation</td></tr><tr><td>Li et al., 27]</td><td>A book on how to develop an automated GUI testing tool.</td><td>✓</td><td></td><td></td></tr><tr><td>Rodríguez et al., 28</td><td>A survey on automated GUI testing in 30 years.</td><td>✓</td><td></td><td></td></tr><tr><td>Arnatovich et al., 29</td><td>A survey on automated techniques for mobile functional GUI testing.</td><td>✓</td><td></td><td></td></tr><tr><td>Ivančić et al., [6]</td><td>A literature review on RPA.</td><td>✓</td><td></td><td></td></tr><tr><td>Said et al., 30</td><td>An overview on mobile GUI testing.</td><td>✓</td><td></td><td></td></tr><tr><td>Li 31</td><td>An survey on Android GUI testing.</td><td>✓</td><td></td><td></td></tr><tr><td>Oksanen et al., 32</td><td>GUI testing on Windows OS.</td><td>✓</td><td></td><td></td></tr><tr><td>Deshmukh et al., 33</td><td>A survey on GUI testing for improving user experience.</td><td>✓</td><td></td><td></td></tr><tr><td>Bajammal et al., 34</td><td>A survey on the use of computer vision for software engineering.</td><td>✓</td><td></td><td></td></tr><tr><td>Yu et al., 35</td><td>A survey on using computer for mobile app GUI testing.</td><td>✓</td><td></td><td></td></tr><tr><td>Syed et al., 36</td><td>A review of contemporary themes and challenges in RPA.</td><td>✓</td><td></td><td></td></tr><tr><td>Chakraborti et al., 37</td><td>A review of emerging trends of intelligent process automation.</td><td>✓</td><td></td><td></td></tr><tr><td>Enriquez et al., 38</td><td>A scientific and industrial systematic mapping study of RPA.</td><td>✓</td><td></td><td></td></tr><tr><td>Ribeiro et al., 39</td><td>A review of combining AI and RPA in industry 4.0.</td><td>✓</td><td></td><td></td></tr><tr><td>Nass et al., 40</td><td>Discuss the chanllenges of GUI testing.</td><td>✓</td><td></td><td></td></tr><tr><td>Agostinelli et al., 41</td><td>Discuss the research challenges of intelligent RPA.</td><td>✓</td><td></td><td></td></tr><tr><td>Wali et al., 42]</td><td>A review on task automation with intelligent agents.</td><td>✓</td><td></td><td></td></tr><tr><td>Zhao et al., 8</td><td>A comprehensive survey of LLMs.</td><td></td><td>✓</td><td></td></tr><tr><td>Zhao et al., 44</td><td>A survey of LLM-based agents.</td><td></td><td>✓</td><td></td></tr><tr><td>Cheng et al., 44</td><td>An overview of LLM-based AI agent.</td><td></td><td>✓</td><td></td></tr><tr><td>Li et al., 45</td><td>A survey on personal LLM agents on their capability, efficiency and security.</td><td></td><td>✓</td><td></td></tr><tr><td>Xie et al., [46]</td><td>A comprehensive survey of LLM-based agents.</td><td></td><td>✓</td><td></td></tr><tr><td>Wang et al.,. 47</td><td>A survey on LLM-based autonomous agents.</td><td></td><td>✓</td><td></td></tr><tr><td>Guo et al., 48</td><td>A survey of mult-agent LLM frameworks.</td><td></td><td>✓</td><td></td></tr><tr><td>Han et al., 49</td><td>A survey on LLM multi-agent systems, with their challenges and open problems.</td><td></td><td>✓</td><td></td></tr><tr><td>Sun et al., 50</td><td>A survey on LLM-based multi-agent reinforcement learning.</td><td></td><td>✓</td><td></td></tr><tr><td>Huang et al., 51]</td><td>A survey on planning in LLM agents.</td><td></td><td>✓</td><td></td></tr><tr><td>Aghzal et al., 52</td><td>A survey on automated planning in LLMs.</td><td></td><td>✓</td><td></td></tr><tr><td>Zheng et al., 53</td><td>Discuss the roadmap of lifelong learning in LLM agents.</td><td></td><td>✓</td><td></td></tr><tr><td>Zhang et al., 54</td><td>A survey on the memory of LLM-based agents.</td><td></td><td>✓</td><td></td></tr><tr><td>Shen 13</td><td>A survey of the tool usage in LLM agents.</td><td></td><td>✓</td><td></td></tr><tr><td>Chang et al., 55</td><td>A survey on evaluation of LLMs.</td><td></td><td>✓</td><td></td></tr><tr><td>Li et al., 56</td><td>A survey on benchmarks multimodal applications.</td><td></td><td>✓</td><td></td></tr><tr><td>Li et al., 57</td><td>A survey on benchmarking evaluations, applications, and challenges of visual LLMs.</td><td></td><td>✓</td><td></td></tr><tr><td>Huang and Zhang 58</td><td>A survey on evaluation of multimodal LLMs.</td><td></td><td>✓</td><td></td></tr><tr><td>Xie et al., 69</td><td>A survey on LLM based multimodal agent.</td><td></td><td>✓</td><td>①</td></tr><tr><td>Durante et al., 60</td><td>A survey of multimodal interaction with AI agents.</td><td></td><td>✓</td><td>0</td></tr><tr><td>Wu et al., 611</td><td>A survey of foundations and trend on multimodal mobile agents.</td><td></td><td>✓</td><td>✓</td></tr><tr><td>Wang et al., 62</td><td>A survey on the integration of foundation models with GUI agents.</td><td></td><td>✓</td><td>✓</td></tr><tr><td>Gao et al., 63</td><td>A survey on autonomous agents across digital platforms.</td><td></td><td>✓</td><td>✓</td></tr><tr><td>Dang et al., 64</td><td>A survey on GUI agents.</td><td></td><td>✓</td><td>✓</td></tr><tr><td>Liu et al., 65</td><td>A survey on GUI agent on phone automation.</td><td></td><td>✓</td><td>✓</td></tr><tr><td>Hu et al., 66</td><td>A survey on MLLM based agents for OS.</td><td></td><td>✓</td><td>✓</td></tr><tr><td>Shi et al., 167</td><td>A survey of building trustworthy GUI agents.</td><td></td><td>✓</td><td>✓</td></tr><tr><td>Ning et al., 68</td><td>A survey of agents for Web automation.</td><td></td><td>✓</td><td>✓</td></tr><tr><td>Tang et al., 69</td><td>A survey of GUI agents powered by (multimodal) LLMs.</td><td></td><td>✓</td><td>✓</td></tr><tr><td>Li and Huang et al., 70</td><td>A summary of GUI agents powered by foundation models and enhanced through reinforcement learning</td><td></td><td>✓</td><td>✓</td></tr><tr><td>Sager et al., 71</td><td>A review of AI agent for computer use.</td><td>O</td><td>✓</td><td>✓</td></tr><tr><td>Our work</td><td>A comprehensive survey on LLM-brained GUI agents, on their foundations, technologies, <br> frameworks, data, models, applications, challenges and future roadmap.</td><td>O</td><td>✓</td><td>✓</td></tr></table>

<!-- Media -->

Regarding individual components within LLM agents, several surveys provide detailed insights that are especially relevant for GUI agents. Huang et al., [51] examine planning mechanisms in LLM agents, which are essential for executing long-term tasks-a frequent requirement in GUI automation. Zhang et al., [54] explore memory mechanisms, which allow agents to store critical historical information, aiding in knowledge retention and decision-making. Additionally, Shen [13] surveys the use of tools by LLMs (such as APIs and code) to interact effectively with their environments, grounding actions in ways that produce tangible impacts. Further, Chang et al., [55] provide a comprehensive survey on evaluation methods for LLMs, which is crucial for ensuring the robustness and safety of GUI agents. Two additional surveys, [56] and [58], provide comprehensive overviews of benchmarks and evaluation methods specifically tailored to multimodal LLMs. The evaluation also facilitates a feedback loop, allowing agents to improve iteratively based on assessment results. Together, these surveys serve as valuable resources, offering guidance on essential components of LLM agents and forming a foundational basis for LLM-based GUI agents.

Compared to existing surveys, our work offers a significantly more comprehensive and up-to-date overview of the LLM-powered GUI agent landscape. We curate and synthesize over 500 references, covering a wide range of topics including foundation models, data sources, system frameworks, benchmarks, evaluation methodologies, and practical deployments. While prior surveys often concentrate on narrower aspects on selected platform (e.g., web, mobile), our survey takes a holistic perspective that spans the full development and deployment lifecycle. Beyond narrative summaries, we also provide consolidated reference tables for each subdomain, enabling readers to quickly categorize and locate relevant works across platforms and research themes—serving as a practical handbook for both researchers and practitioners. Furthermore, we incorporate foundational background material and propose evaluation taxonomies that make the survey accessible to newcomers, addressing gaps in prior work that often assume a high degree of prior familiarity.


<!-- Meanless: JOURNAL OF IATEX CLASS FILES, DECEMBER 2024<br>7 -->

## 3 BACKGROUND

The development of LLM-brained GUI agents is grounded in three major advancements: (i) large language models (LLMs) [8], which bring advanced capabilities in natural language understanding and code generation, forming the core intelligence of these agents; (ii) accompanying agent architectures and tools [47] that extend LLM capabilities, bridging the gap between language models and physical environments to enable tangible impacts; and (iii) GUI automation [73], which has cultivated a robust set of tools, models, and methodologies essential for GUI agent functionality. Each of these components has played a critical role in the emergence of LLM-powered GUI agents. In the following subsections, we provide a brief overview of these areas to set the stage for our discussion.

### 3.1 Large Language Models: Foundations and Capabili- ties

The study of language models has a long and rich history [74], beginning with early statistical language models [75] and smaller neural network architectures [76]. Building on these foundational concepts, recent advancements have focused on transformer-based LLMs, such as the Generative Pre-trained Transformers (GPTs) [77]. These models are pretrained on extensive text corpora and feature significantly larger model sizes, validating scaling laws and demonstrating exceptional capabilities across a wide range of natural language tasks. Beyond their sheer size, these LLMs exhibit enhanced language understanding and generation abilities, as well as emergent properties that are absent in smaller-scale language models 78].

Early neural language models, based on architectures like recurrent neural networks (RNNs) [79] and long short-term memory networks (LSTMs) [80], were limited in both performance and generalization. The introduction of the Transformer model, built on the attention mechanism [81], marked a transformative milestone, establishing the foundational architecture now prevalent across almost all subsequent LLMs. This development led to variations in model structures, including encoder-only models (e.g., BERT [82], RoBERTa [83], ALBERT [84]), decoder-only models (e.g., GPT-1 [85], GPT-2 [86]), and encoder-decoder models (e.g., T5 [87], BART [88]). In 2022, ChatGPT [11] based on GPT-3.5 [89] launched as a groundbreaking LLM, fundamentally shifting perceptions of what language models can achieve. Since then, numerous advanced LLMs have emerged, including GPT-4 [90], LLaMA-3 [91], and Gemini [92], propelling the field into rapid growth. Today's LLMs are highly versatile, with many of them are capable of processing multimodal data and performing a range of tasks, from question answering to code generation, making them indispensable tools in various applications [93]-[96].

The emergence of LLMs has also introduced significant advanced properties that invigorate their applications, making previously challenging tasks, such as natural language-driven GUI agents feasible. These advancements include:

1) Few-Shot Learning [77]: Also referred to as in-context learning [97], LLMs can acquire new tasks from a small set of demonstrated examples presented in the prompt during inference, eliminating the need for retraining. This capability is crucial for enabling GUI agents to generalize across different environments with minimal effort.

2) Instruction Following [98]: After undergoing instruction tuning, LLMs exhibit a remarkable ability to follow instructions for novel tasks, demonstrating strong generalization skills [89]. This allows LLMs to effectively comprehend user requests directed at GUI agents and to follow predefined objectives accurately.

3) Long-Term Reasoning [99]: LLMs possess the ability to plan and solve complex tasks by breaking them down into manageable steps, often employing techniques like chain-of-thought (CoT) reasoning [100], [101]. This capability is essential for GUI agents, as many tasks require multiple steps and a robust planning framework.

4) Code Generation and Tool Utilization [102]: LLMs excel in generating code and utilizing various tools, such as APIs [13]. This expertise is vital, as code and tools form the essential toolkit for GUI agents to interact with their environments.

5) Multimodal Comprehension [10]: Advanced LLMs can integrate additional data modalities, such as images, into their training processes, evolving into multimodal models. This ability is particularly important for GUI agents, which must interpret GUI screenshots presented as images in order to function effectively [103].

To further enhance the specialization of LLMs for GUI agents, researchers often fine-tune these models with domain-specific data, such as user requests, GUI screenshots, and action sequences, thereby increasing their customization and effectiveness. In Section 8, we delve into these advanced, tailored models for GUI agents, discussing their unique adaptations and improved capabilities for interacting with graphical interfaces.

### 3.2 LLM Agents: From Language to Action

Traditional AI agents have often focused on enhancing specific capabilities, such as symbolic reasoning or excelling in particular tasks like Go or Chess. In contrast, the emergence of LLMs has transformed AI agents by providing them with a natural language interface, enabling human-like decision-making capabilities, and equipping them to perform a wide variety of tasks and take tangible actions in diverse environments [12], [47], [104], [105]. In LLM agents, if LLMs form the "brain" of a GUI agent, then its accompanying components serve as its "eyes and hands", enabling the LLM to perceive the environment's status and translate its textual output into actionable steps that generate tangible effects [46]. These components transform LLMs from passive information sources into interactive agents that execute tasks on behalf of users, which redefine the role of LLMs from purely text-generative models to systems capable of driving actions and achieving specific goals.

In the context of GUI agents, the agent typically perceives the GUI status through screenshots and widget trees [106], then performs actions to mimic user operations (e.g., mouse clicks, keyboard inputs, touch gestures on phones) within the environment. Since tasks can be long-term, effective planning and task decomposition are often required, posing unique challenges. Consequently, an LLM-powered GUI agent usually possess multimodal capabilities [59], a robust planning system [51], a memory mechanism to analyze historical interactions [54], and a specialized toolkit to interact with its environment [27]. We will discuss these tailored designs for GUI agents in detail in Section 5


<!-- Meanless: JOURNAL OF IATEX CLASS FILES, DECEMBER 2024<br>8 -->

### 3.3 GUI Automation: Tools, Techniques, and Challenges

GUI automation has been a critical area of research and application since the early days of GUIs in computing. Initially developed to improve software testing efficiency, GUI automation focused on simulating user actions, such as clicks, text input, and navigation, across graphical applications to validate functionality [30]. Early GUI automation tools were designed to execute repetitive test cases on static interfaces [28]. These approaches streamlined quality assurance processes, ensuring consistency and reducing manual testing time. As the demand for digital solutions has grown, GUI automation has expanded beyond testing to other applications, including RPA [6] and Human-Computer Interaction (HCI) [107]. RPA leverages GUI automation to replicate human actions in business workflows, automating routine tasks to improve operational efficiency. Similarly, HCI research employs GUI automation to simulate user behaviors, enabling usability assessments and interaction studies. In both cases, automation has significantly enhanced productivity and user experience by minimizing repetitive tasks and enabling greater system adaptability [108], [109].

Traditional GUI automation methods have primarily depended on scripting and rule-based frameworks [4, [110]. Scripting-based automation utilizes languages such as Python, Java, and JavaScript to control GUI elements programmatically. These scripts simulate a user's actions on the interface, often using tools like Selenium [111] for web-based automation or Autolt [112] and SikuliX [113] for desktop applications. Rule-based approaches, meanwhile, operate based on predefined heuristics, using rules to detect and interact with specific GUI elements based on properties such as location, color, and text labels [4]. While effective for predictable, static workflows [114], these methods struggle to adapt to the variability of modern GUIs, where dynamic content, responsive layouts, and user-driven changes make it challenging to maintain rigid, rule-based automation [115].

CV has become essential for interpreting the visual aspects of GUIs [35], [116], [117], enabling automation tools to recognize and interact with on-screen elements even as layouts and designs change. CV techniques allow GUI automation systems to detect and classify on-screen elements, such as buttons, icons, and text fields, by analyzing screenshots and identifying regions of interest [118-120]. Optical Character Recognition (OCR) further enhances this capability by extracting text content from images, making it possible for automation systems to interpret labels, error messages, and form instructions accurately [121]. Object detection models add robustness, allowing automation agents to locate GUI elements even when the visual layout shifts [103]. By incorporating CV, GUI automation systems achieve greater resilience and adaptability in dynamic environments.

Despite advances, traditional GUI automation methods fall short in handling the complexity and variability of contemporary interfaces. Today's applications often feature dynamic, adaptive elements that cannot be reliably automated through rigid scripting or rule-based methods alone [122], [123]. Modern interfaces increasingly require contextual awareness [124], such as processing on-screen text, interpreting user intent, and recognizing visual cues. These demands reveal the limitations of existing automation frameworks and the need for more flexible solutions capable of real-time adaptation and context-sensitive responses.

LLMs offer a promising solution to these challenges. With their capacity to comprehend natural language, interpret context, and generate adaptive scripts, LLMs can enable more intelligent, versatile GUI automation [125]. Their ability to process complex instructions and learn from context allows them to bridge the gap between static, rule-based methods and the dynamic needs of contemporary GUIs [126]. By integrating LLMs with GUI agents, these systems gain the ability to generate scripts on-the-fly based on the current state of the interface, providing a level of adaptability and sophistication that traditional methods cannot achieve. The combination of LLMs and GUI agents paves the way for an advanced, user-centered automation paradigm, capable of responding flexibly to user requests and interacting seamlessly with complex, evolving interfaces.

## 4 EVOLUTION AND PROGRESSION OF LLM- BRAINED GUI AGENTS

"Rome wasn't built in a day." The development of LLM-brained GUI agents has been a gradual journey, grounded in decades of research and technical progress. Beginning with simple GUI testing scripts and rule-based automation frameworks, the field has evolved significantly through the integration of machine learning techniques, creating more intelligent and adaptive systems. The introduction of LLMs, especially multimodal models, has transformed GUI automation by enabling natural language interactions and fundamentally reshaping how users interact with software applications.

As illustrated in Figure 3 prior to 2023 and the emergence of LLMs, work on GUI agents was limited in both scope and capability. Since then, the proliferation of LLM-based approaches has fostered numerous notable developments across platforms including web, mobile, and desktop environments. This surge is ongoing and continues to drive innovation in the field. This section takes you on a journey tracing the evolution of GUI agents, emphasizing key milestones that have brought the field to its present state.

### 4.1 Early Automation Systems

In the initial stages of GUI automation, researchers relied on random-based, rule-based, and script-based strategies. While foundational, these methods had notable limitations in terms of flexibility and adaptability.


<!-- Meanless: JOURNAL OF IATEX CLASS FILES, DECEMBER 2024<br>9 -->

<!-- Media -->

<!-- figureText: ⑤ OpenAI<br>Operator<br>2025+<br>Infibuerance<br>UI-TARS<br>Tencent<br>WebVoyager<br>Web<br>Mobile-Agent<br>Mobile<br>SeeAct<br>Sep-Dec<br>Computer<br>CoCo-Agent<br>NaviQAte<br>Cross<br>DUAL-VCR<br>Steward<br>UFOT<br>Hybrid Agent<br>CogAgent<br>OS-Copilot<br>2024<br>®<br>WMA<br>Auto-GUI<br>Cradle<br>amazon<br>AgentOccam<br>G<br>AutoWebGLM<br>May-Aug<br>NNetnav<br>Tencen<br>LASER<br>國<br>MMAC-Copilot<br>MobA<br>Zero-shot Agent<br>SeeClick<br>SIMULAI<br>Agent S<br>OpenAgents<br>AutoGLM<br>WoB<br>MM-Navigator<br>LiMAC<br>WGE<br>Tencen<br>AppAgent<br>Jan-Apr<br>GUI Narrator<br>TinyClick<br>8<br>OSCAR<br>Qweb<br>WebArena<br>Lenovo<br>Mobile-Experts<br>DOM-Q-NET<br>WebAgent<br>emergence<br>Agent-E<br>A<br>Claude Computer Use<br>FLIN<br>AutoDroid<br>一<br>Search-Agent<br>Sep-Dec<br>WebGPT<br>Agent Q<br>WebShop<br>DroidBot-GPT<br>Openwebagent<br>May-Aug<br>WebPilot<br>: 2023<br>Jan-Apr<br>2023 -->

<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2025/11/2025_11_04__12_39_04_37ba24.jpg"/>

Fig. 3: An overview of GUI agents evolution over years.

<!-- Media -->

#### 4.1.1 Random-Based Automation

Random-based automation uses random sequences of actions within the GUI without relying on specific algorithms or structured models using monkey test [127]. This approach was widely used in GUI testing to uncover potential issues by exploring unpredictable input sequences [128]. While effective at identifying edge cases and bugs, random-based methods were often inefficient due to a high number of redundant or irrelevant trials.

#### 4.1.2 Rule-Based Automation

Rule-based automation applies predefined rules and logic to automate tasks. In 2001, Memon et al., 129 introduced a planning approach that generated GUI test cases by transforming initial states to goal states through a series of predefined operators. Hellmann et al., [4] (2011) demonstrated the potential of rule-based approaches in exploratory testing, enhancing bug detection. In the RPA domain, SmartRPA [130] (2020) used rule-based processing to automate routine tasks, illustrating the utility of rules for streamlining structured processes.

#### 4.1.3 Script-Based Automation

Script-based automation relies on detailed scripts to manage GUI interactions. Tools like jRapture [5] (2000) record and replay Java-based GUI sequences using Java binaries and the JVM, enabling consistent execution by precisely reproducing input sequences. Similarly, DART [131] (2003) automated the GUI testing lifecycle, from structural analysis to test case generation and execution, offering a comprehensive framework for regression testing.

#### 4.1.4 Tools and Software

A range of software tools were developed for GUI testing and business process automation during this period. Mi-crosoft Power Automate [132] (2019) provides a low-code/no-code environment for creating automated workflows within Microsoft applications. Selenium [133] (2004) supports cross-browser web testing, while Appium [134 (2012) facilitates mobile UI automation. Commercial tools like TestComplete [135] (1999), Katalon Studio [136] (2015), and Ranorex [137] (2007) allow users to create automated tests with cross-platform capabilities.

Although these early systems were effective for automating specific, predefined workflows, they lacked flexibility and required manual scripting or rule-based logic. Nonetheless, they established the foundations of GUI automation, upon which more intelligent systems were built.

### 4.2 The Shift Towards Intelligent Agents

The incorporation of machine learning marked a major shift towards more adaptable and capable GUI agents. Early milestones in this phase included advancements in machine learning, natural language processing, computer vision, and reinforcement learning applied to GUI tasks.

#### 4.2.1 Machine Learning and Computer Vision

RoScript [110] (2020) was a pioneering system that introduced a non-intrusive robotic testing system for touchscreen applications, expanding GUI automation to diverse platforms. AppFlow [138] (2018) used machine learning to recognize common screens and UI components, enabling modular testing for broad categories of applications. Progress in computer vision also enabled significant advances in GUI testing, with frameworks [117] (2010) automating visual interaction tasks. Humanoid [139] (2019) uses a deep neural network model trained on human interaction traces within the Android system to learn how users select actions based on an app's GUI. This model is then utilized to guide test input generation, resulting in improved coverage and more human-like interaction patterns during testing. Similarly, Deep GUI 140 (2021) applies deep learning techniques to filter out irrelevant parts of the screen, thereby enhancing black-box testing effectiveness in GUI testing by focusing only on significant elements. These approaches demonstrate the potential of deep learning to make GUI testing more efficient and intuitive by aligning it closely with actual user behavior.


<!-- Meanless: JOURNAL OF IATEX CLASS FILES, DECEMBER 2024<br>10 -->

Widget detection, as demonstrated by White et al., 103 (2019), leverages computer vision to accurately identify UI elements, serving as a supporting technique that enables more intelligent and responsive UI automation. By detecting and categorizing interface components, this approach enhances the agent's ability to interact effectively with complex and dynamic GUIs [141].

#### 4.2.2 Natural Language Processing

Natural language processing capabilities introduced a new dimension to GUI automation. Systems like RUSS [142] (2021) and FLIN [143] (2020) allowed users to control GUIs through natural language commands, bridging human language and machine actions. Datasets, such as those in [144] (2020), further advanced the field by mapping natural language instructions to mobile UI actions, opening up broader applications in GUI control. However, these approaches are limited to handling simple natural commands and are not equipped to manage long-term tasks.

#### 4.2.3 Reinforcement Learning

The development of environments like World of Bits (WoB) 145 (2017) enabled the training of web-based agents using reinforcement learning (RL). Workflow-guided exploration 146 (2018) improved RL efficiency and task performance. DQT [147] (2024) applied deep reinforcement learning to automate Android GUI testing by preserving widget structures and semantics, while AndroidEnv [148] (2021) offered realistic simulations for agent training on Android. WebShop [149] (2022) illustrated the potential for large-scale web interaction, underscoring the growing sophistication of RL-driven GUI automation.

While these machine learning-based approaches were more adaptable than earlier rule-based systems [150], [151], they still struggled to generalize across diverse, unforeseen tasks. Their dependence on predefined workflows and limited adaptability required retraining or customization for new environments, and natural language control was still limited.

### 4.3 The Advent of LLM-Brained GUI Agents

The introduction of LLMs, particularly multimodal models like GPT-40 [93] (2023), has radically transformed GUI automation by allowing intuitive interactions through natural language. Unlike previous approaches that required integration of separate modules, LLMs provide an end-to-end solution for GUI automation, offering advanced capabilities in natural language understanding, visual recognition, and reasoning.

LLMs present several unique advantages for GUI agents, including natural language understanding, multimodal processing, planning, and generalization. These features make LLMs and GUI agents a powerful combination. While there were earlier explorations, 2023 marked a pivotal year for LLM-powered GUI agents, with significant developments across various platforms such as web, mobile, and desktop applications.

#### 4.3.1 Web Domain

The initial application of LLMs in GUI automation was within the web domain, with early studies establishing benchmark datasets and environments [145], [149]. A key milestone was WebAgent [152] (2023), which, alongside WebGUM [153] (2023), pioneered real-world web navigation using LLMs. These advancements paved the way for further developments [17], [154], [155], utilizing more specialized LLMs to enhance web-based interactions.

#### 4.3.2 Mobile Devices

The integration of LLMs into mobile devices began with AutoDroid [156] (2023), which combined LLMs with domain-specific knowledge for smartphone automation. Additional contributions like MM-Navigator [157] (2023), AppAgent [18] (2023), and Mobile-Agent [158] (2023) enabled refined control over smartphone applications. Research has continued to improve accuracy for mobile GUI automation through model fine-tuning [159], [160] (2024).

#### 4.3.3 Computer Systems

For desktop applications, UFO [19] (2024) was one of the first systems to leverage GPT-4 with visual capabilities to fulfill user commands in Windows environments. Cradle [161] (2024) extended these capabilities to software applications and games, while Wu et al., [162] (2024) provided interaction across diverse desktop applications, including web browsers, code terminals, and multimedia tools.

#### 4.3.4 Industry Models

In industry, the Claude 3.5 Sonnet model [163] (2024) introduced a "computer use" feature capable of interacting with desktop environments through UI operations [164]. This signifies the growing recognition of LLM-powered GUI agents as a valuable application in industry, with stakeholders increasingly investing in this technology.

OpenAI quickly followed up by releasing Operator 165] in 2025, a Computer-Using Agent (CUA) similar to Claude, achieving state-of-the-art performance across various benchmarks. This development underscores the industry's recognition of the value of GUI agents and its growing investment in the field. As interest continues to surge, GUI agent research and development are expected to become increasingly competitive, marking the beginning of a rapidly evolving landscape.

Undoubtedly, LLMs have introduced new paradigms and increased the intelligence of GUI agents in ways that were previously unattainable. As the field continues to evolve, we anticipate a wave of commercialization, leading to transformative changes in user interaction with GUI applications.

### 4.4 GUI Agent vs. API-Based Agent

In the field of LLM-powered agents operating within digital environments, the action space can be broadly categorized into two types:

1) GUI Agents, which primarily rely on GUI operations (e.g., clicks, keystrokes) to complete tasks.

2) API-Based Agents, which utilize system or application-native APIs to fulfill objectives. We show the principle of both agent types in Figure 4. Each type has distinct advantages, and a deeper understanding of these approaches is critical for designing effective agents.


<!-- Meanless: JOURNAL OF IATEX CLASS FILES, DECEMBER 2024<br>11 -->

<!-- Media -->

<!-- figureText: API Information<br>User Query<br>Description: ...<br>↗<br>Args: ...<br><icon/><br>GUI Observation<br>Return: ...<br>Examples: ...<br>API 2<br>Description: ...<br>GUI Agent<br>Args: ...<br>Return: ...<br>API Agent<br>Examples: ...<br>API 6<br>API n<br>CLICK (X=..., Y=...)<br>Description: ...<br>Args: ...<br>argi=...<br>CLICK (X=... Y=...)<br>Return: ...<br>CLICK(X=··· , Y=...) -->

<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2025/11/2025_11_04__12_39_04_d5d8de.jpg"/>

Fig. 4: The comparison between API agent vs. GUI agent.

<!-- Media -->

GUI operations provide a universal control interface that can operate across diverse applications using the same action primitives. This makes GUI agents highly generalizable, as they can interact with a wide range of software environments without requiring application-specific adaptations. However, GUI-based interactions are inherently more complex; even simple tasks may require multiple sequential steps, which can increase both the decision-making cost for the agent and the computational resources required for long-term, multi-step workflows. Another key aspect is the transparency of actions in GUI agents. Since GUI agents interact with applications in the same way a human would, by clicking, typing, and navigating through the interface, their actions are inherently more observable and interpretable to users. This transparency fosters better trust and comprehension in agent-computer interactions.

In contrast, API-based agents offer a more efficient and direct approach to task completion. By leveraging native APIs, tasks can often be fulfilled with a single, precise call, significantly reducing execution time and complexity. However, these native APIs are often private or restricted to specific applications, limiting accessibility and generalizability. This makes API-based agents less versatile in scenarios where API access is unavailable or insufficient. In addition, API-based agents operate behind the scenes, executing tasks through direct system calls, which, while often more efficient and reliable, can make their operations less visible and harder to debug for end users.

The most effective digital agents are likely to operate in a hybrid manner, combining the strengths of both approaches. Such agents can utilize GUI operations to achieve broad compatibility across software while exploiting native APIs where available to maximize efficiency and effectiveness. These hybrid agents strike a balance between generalization and task optimization, making them a critical focus area in this survey. For a more comprehensive comparison between GUI agents and API agents, please refer to [166].

## 5 LLM-BRAINED GUI AGENTS: FOUNDATIONS AND DESIGN

In essence, LLM-brained GUI agents are designed to process user instructions or requests given in natural language, interpret the current state of the GUI through screenshots or UI element trees, and execute actions that simulate human interaction across various software interfaces [19]. These agents harness the sophisticated natural language understanding, reasoning, and generative capabilities of LLMs to accurately comprehend user intent, assess the GUI context, and autonomously engage with applications across diverse environments, thereby enabling the completion of complex, multi-step tasks. This integration allows them to seamlessly interpret and respond to user requests, bringing adaptability and intelligence to GUI automation.

As a specialized type of LLM agent, most current GUI agents adopt a similar foundational framework, integrating core components such as planning, memory, tool usage, and advanced enhancements like multi-agent collaboration, among others [47]. However, each component must be tailored to meet the specific objectives of GUI agents to ensure adaptability and functionality across various application environments.

In the following sections, we provide an in-depth overview of each component, offering a practical guide and tutorial on building an LLM-powered GUI agent from the ground up. This comprehensive breakdown serves as a cookbook for creating effective and intelligent GUI automation systems that leverage the capabilities of LLMs.

### 5.1 Architecture and Workflow In a Nutshell

In Figure 5, we present the architecture of an LLM-brained GUI agent, showcasing the sequence of operations from user input to task completion. The architecture comprises several integrated components, each contributing to the agent's ability to interpret and execute tasks based on user-provided natural language instructions. Upon receiving a user request, the agent follows a systematic workflow that includes environment perception, prompt engineering, model inference, action execution, and continuous memory utilization until the task is fully completed.

In general, it consists of the following components:

1) Operating Environment: The environment defines the operational context for the agent, encompassing platforms such as mobile devices, web browsers, and desktop operating systems like Windows. To interact meaningfully, the agent perceives the environment's current state through screenshots, widget trees, or other methods of capturing UI structure [167]. It continuously monitors feedback on each action's impact, adjusting its strategy in real time to ensure effective task progression.

2) Prompt Engineering: Following environment perception, the agent constructs a detailed prompt to guide the LLM's inference [168]. This prompt incorporates user instructions, processed visual data (e.g., screenshots), UI element layouts, properties, and any additional context relevant to the task. This structured input maximizes the LLM's ability to generate coherent, context-aware responses aligned with the current GUI state.

3) Model Inference: The constructed prompt is passed to a LLM, the agent's inference core, which produces a sequence of plans, actions and insights required to fulfill the user's request. This model may be a general-purpose LLM or a specialized model fine-tuned with GUI-specific data, enabling a more nuanced understanding of GUI interactions, user flows, and task requirements.


<!-- Meanless: JOURNAL OF LATEX CLASS FILES, DECEMBER 2024<br>12 -->

<!-- Media -->

<!-- figureText: Prompt Engineering<br>Input<br>Memory<br>Instruction<br>Example 1<br>Step t-1<br>Step t-2<br>Step t-n<br>Request<br>Input<br>Example 1<br>Action Execution<br>Input<br>Model Inference<br>Action<br>IIII<br>Input<br>Example n<br>Plan<br>LLM<br>Input<br>\{"type": Botton, "title":<br>"New", "position": [0.45, 0.78] \}<br>Perception<br>\{"type": Edit, "title":<br>"Document", "position": $\left\lbrack {{0.87},{0.43}}\right\rbrack \}$<br>"Design" "position": $\left\lbrack {{0.25},{0.21}}\right\rbrack \}$<br>\{"type": ComboBox, "title":<br>"SaveAs" "position": [0.67, 0.32] \}<br>Screenshots<br>Widget Tree<br>UI Element Properties<br>Environment State<br>Operating Environment -->

<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2025/11/2025_11_04__12_39_04_f942d7.jpg"/>

Fig. 5: An overview of the architecture and workflow of a basic LLM-powered GUI agent.

<!-- Media -->

4) Actions Execution: Based on the model's inference results, the agent identifies specific actions (such as mouse clicks, keyboard inputs, touchscreen gestures, or API calls) required for task execution [13]. An executor within the agent translates these high-level instructions into actionable commands that impact the GUI directly, effectively simulating human-like interactions across diverse applications and devices.

5) Memory: For multi-step tasks, the agent maintains an internal memory to track prior actions, task progress, and environment states [54]. This memory ensures coherence throughout complex workflows, as the agent can reference previous steps and adapt its actions accordingly. An external memory module may also be incorporated to enable continuous learning, access external knowledge, and enhance adaptation to new environments or requirements.

By iteratively traversing these stages and assembling the foundational components, the LLM-powered GUI agent operates intelligently, seamlessly adapting across various software interfaces and bridging the gap between language-based instruction and concrete action. Each component is critical to the agent's robustness, responsiveness, and capability to handle complex tasks in dynamic environments. In the following subsections, we detail the design and core techniques underlying each of these components, providing a comprehensive guide for constructing LLM-powered GUI agents from the ground up.

### 5.2 Operating Environment

The operating environment for LLM-powered GUI agents encompasses various platforms, such as mobile, web, and desktop operating systems, where these agents can interact with graphical interfaces. Each platform has distinct characteristics that impact the way GUI agents perceive, interpret, and act within it. Examples of GUIs from each platform are shown in Figure 6 This section details the nuances of each platform, the ways agents gather environmental information, and the challenges they face in adapting to diverse operating environments.

<!-- Media -->

<!-- figureText: (a) Web GUI<br>(b) Mobile GUI<br>(c) Computer GUI -->

<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2025/11/2025_11_04__12_39_04_e5ae6c.jpg"/>

Fig. 6: Examples of GUIs from web, mobile and computer platforms.

<!-- Media -->

#### 5.2.1 Platform

The operating environment for LLM-powered GUI agents encompasses various platforms, such as mobile, web, and desktop operating systems, where these agents can interact with graphical interfaces. Each platform has distinct characteristics that impact the way GUI agents perceive, interpret, and act within it. Examples of GUIs from each platform are shown in Figure 6. This section details the nuances of each platform, the ways agents gather environmental information, and the challenges they face in adapting to diverse operating environments.

1) Mobile Platforms: Mobile devices operate within constrained screen real estate, rely heavily on touch interactions [170], and offer varied app architectures (e.g., native vs. hybrid apps). Mobile platforms often use accessibility frameworks, such as Android's Accessibility AP | 171 and iOS's VoiceOver Accessibility Inspector ${}^{5}$ to expose structured information about UI elements. However, GUI agents must handle additional complexities in mobile environments, such as gesture recognition [169], app navigation [172], and platform-specific constraints (e.g., security and privacy permissions) [173], [174].


<!-- Meanless: 13 -->

<!-- Media -->

JOURNAL OF LATEX CLASS FILES, DECEMBER 2024

TABLE 3: Summary of platform-specific challenges, action spaces, and typical tasks for Web, Mobile, and Computer GUI environments.

<table><tr><td>Platform</td><td>Typical GUI Challenges</td><td>Action Space</td><td>Representative Tasks</td></tr><tr><td>Mobile</td><td>- Constrained screen real estate <br> - Heavy reliance on touch and gesture recognition [169] <br> - App architectures (native vs. hybrid) <br> - Accessibility frameworks (e.g., Android's Accessibility API, iOS VoiceOver) <br> - Platform-specific constraints (permissions, security, privacy)</td><td>- Tap, swipe, pinch, and other touch gestures <br> - Virtual keyboard input <br> - In-app navigation (menus, tabs) <br> - Accessing hardware features (camera, GPS)</td><td>- App-based login and form filling <br> - Messaging, social media posting <br> - Location-based services and map interactions <br> - Handling push notifications and permission dialogs</td></tr><tr><td>Web</td><td>- Dynamic and responsive layouts <br> - Asynchronous updates (AJAX, fetch APIs) <br> - HTML/DOM-based structures <br> - Cross-browser inconsistencies</td><td>- Click, hover, scroll <br> - DOM-based form filling <br> - Link navigation and element inspection <br> - JavaScript event triggering</td><td>- Form completion (registrations, checkouts) <br> - Data extraction/web scraping <br> - Searching and filtering (e.g., e-commerce) <br> - Multi-step web navigation (redirects, pop-ups)</td></tr><tr><td>Computer</td><td>- Full-fledged OS-level interfaces <br> - Multi-window operations system-level shortcuts <br> - Automation APIs (e.g., Windows UI Automation [32] <br> - Frequent software updates requiring adaptation <br> - Complex, multi-layered software suites</td><td>- Mouse click, drag-and-drop <br> - Keyboard shortcuts and text input <br> - Menu navigation, toolbars <br> - Access to multiple application windows</td><td>- File management and system settings <br> - Productivity software usage (office suites, IDEs) <br> - Installing/uninstalling applications <br> - Coordinating multi-application work-flows</td></tr></table>

<!-- figureText: the spectrum is a considerably integration that spectra $\left( 1\right)$ to a point $\left( 1\right)$ - ${shape}\left( {X - b}\right) + {shape}\left( X\right)$ -<br>6162 63<br>sulps the mass of current task.",<br>55 56 55 58 58 59<br>Somme<br>- continue<br>(a) A clean GUI screenshot.<br>(b) A GUI screenshot with widgets highlighted by SoM.<br>(c) A GUI screenshot with widgets highlighted by bounding boxes. -->

<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2025/11/2025_11_04__12_39_04_3ac008.jpg"/>

Fig. 7: Examples of different variants of VS Code GUI screenshots.

<!-- Media -->

2) Web Platforms: Web applications provide a relatively standardized interface, typically accessible through Hy-pertext Markup Language (HTML) and Document Object Model (DOM) structures [175], [176]. GUI agents can leverage HTML attributes, such as element ID, class, and tag, to identify interactive components. Web environments also present dynamic content, responsive layouts, and asynchronous updates (e.g., AJAX requests) [177], requiring agents to continuously assess the DOM and adapt their actions to changing interface elements.

---

<!-- Footnote -->

4. https://developer.android.com/reference/android/

accessibilityservice/AccessibilityService

5. https://developer.apple.com/documentation/accessibility/

accessibility-inspector

<!-- Footnote -->

---


<!-- Meanless: JOURNAL OF LATEX CLASS FILES, DECEMBER 2024<br>14 -->

<!-- Media -->

<!-- figureText: GUI<br>Compress Pictures<br>Remove<br>Corrections<br>Color<br>Artistic<br>Transparency<br>Change Picture v<br><icon/><br>Adjust<br>Botton - Remove Background<br>Menultem - Corrections<br>Menultem - Color<br>Menultem - Artistic Effects<br>GroupBox - Adjust<br>Menultem - Transparency<br>Menultem - Compress Pictures..<br>Menultem - Change Picture<br>Button - Reset Picture<br>SplitButton - Reset Picture<br>Menultem - More Options<br>Widget Tree -->

<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2025/11/2025_11_04__12_39_04_0b5bba.jpg"/>

Fig. 8: An example of a GUI and its widget tree.

<!-- Media -->

3) Computer Platforms: Computer OS platforms, such as Windows, offer full control over GUI interactions. Agents can utilize system-level automation APIs, such as Windows UI Automation [32], to obtain comprehensive UI element data, including type, label, position, and bounding box. These platforms often support a broader set of interaction types, mouse, keyboard, and complex multi-window operations. These enable GUI agents to execute intricate workflows. However, these systems also require sophisticated adaptation for diverse applications, ranging from simple UIs to complex, multi-layered software suites.

In summary, the diversity of platforms, spanning mobile, web, and desktop environments, enable GUI agents to deliver broad automation capabilities, making them a generalized solution adaptable across a unified framework. However, each platform presents unique characteristics and constraints at both the system and application levels, necessitating a tailored approach for effective integration. By considering these platform-specific features, GUI agents can be optimized to address the distinctive requirements of each environment, thus enhancing their adaptability and reliability in varied automation scenarios.

#### 5.2.2 Environment State Perception

Accurately perceiving the current state of the environment is essential for LLM-powered GUI agents, as it directly informs their decision-making and action-planning processes. This perception is enabled by gathering a combination of structured data, such as widget trees, and unstructured data, like screenshots, to capture a complete representation of the interface and its components. In Table 4, we outline key toolkits available for collecting GUI environment data across various platforms, and below we discuss their roles in detail:

<!-- Media -->

<table><tr><td>Widget</td><td>Widget Name</td><td>Position</td><td>Attributes</td></tr><tr><td><img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2025/11/2025_11_04__12_39_04_8abe11.jpg"/> <br> Remove Background</td><td>Button - 'Remove Background'</td><td>L-3810, T128, R-3708, B243</td><td>title='Remove Background'; auto_id='PictureBackgroundRemoval'; control_type='Button</td></tr><tr><td><img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2025/11/2025_11_04__12_39_04_95aaac.jpg"/>  Corrections</td><td>Menultem - 'Corrections'</td><td>L-3689, T128, R-3592, B243</td><td>title='Corrections'; auto_id='PictureCorrectionsMenu'; control_type='MenuItem'</td></tr><tr><td><img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2025/11/2025_11_04__12_39_04_cc9a23.jpg"/></td><td>Menultem - 'Color'</td><td>L-3589, T128, R-3527, B243</td><td>title='Color'; auto_id='PictureColorMenu'; control type='Menultem'</td></tr><tr><td><img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2025/11/2025_11_04__12_39_04_af54fd.jpg"/> <br> Artistic Effects -</td><td>Menultem - 'Artistic Effects'</td><td>L-3524, T128, R-3448, B243</td><td>title='Artistic Effects'; auto id='PictureArtisticEffectsGallery'; control_type='MenuItem'</td></tr><tr><td><img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2025/11/2025_11_04__12_39_04_1f4c02.jpg"/></td><td>Menultem - 'Transparency'</td><td>L-3445, T128, R-3336, B243</td><td>title='Transparency'; auto_id='PictureTransparencyGallery'; control_type='MenuItem'</td></tr><tr><td>E. Compress Pictures</td><td>Button - 'Compress Pictures...'</td><td>L-3333, T128, R-3138, B164</td><td>title='Compress Pictures...'; auto id='PicturesCompress'; control type='Button'</td></tr><tr><td><icon/></td><td>MenuItem - 'Change Picture'</td><td>L-3333, T167, R-3149, B203</td><td>title='Change Picture'; auto id='PictureChangeMenu'; control_type='Menultem'</td></tr><tr><td><icon/></td><td>SplitButton - 'Reset Picture'</td><td>L-3333, T206, R-3160, B242</td><td>title='Reset Picture'; control type='SplitButton'</td></tr></table>

Fig. 9: Examples of UI element properties in the PowerPoint application for GUI Agent interaction.

<!-- Media -->

1) GUI Screenshots: Screenshots provide a visual snapshot of the application, capturing the entire state of the GUI at a given moment. They offer agents a reference for layout, design, and visual content, which is crucial when structural details about UI elements are either limited or unavailable. Visual elements like icons, images, and other graphical cues that may hold important context can be analyzed directly from screenshots. Many platforms have built-in tools to capture screenshots (e.g., Windows Snipping Too 7 macOS Screenshot Utility and Android's MediaProjection AP 9, and screenshots can be enhanced with additional annotations, such as Set-of-Mark (SoM) highlights [178] or bounding boxes 179 around key UI components, to streamline agent decisions. Figure 7 illustrates various screenshots of the VS Code GUI, including a clean version, as well as ones with SoM and bounding boxes that highlight actionable components, helping the agent focus on the most critical areas of the interface.

2) Widget Trees: Widget trees present a hierarchical view of interface elements, providing structured data about the layout and relationships between components [180]. We show an example of a GUI and its widget tree in Figure 8. By accessing the widget tree, agents can identify attributes such as element type, label, role, and relationships within the interface, all of which are essential for contextual understanding. Tools like Windows UI Automation and macOS's Accessibility AP ${}^{10}$ provide structured views for desktop applications, while Android's Accessibility API and HTML DOM structures serve mobile and web platforms, respectively. This hierarchical data is indispensable for agents to map out logical interactions and make informed choices based on the UI structure.

---

<!-- Footnote -->

6. https://learn.microsoft.com/en-us/dotnet/framework/ui-automation/ ui-automation-overview

7. https://support.microsoft.com/en-us/windows/

use-snipping-tool-to-capture%2Dscreenshots%2D00246869%

2D1843%2D655f%2Df220%2D97299b865f6b

8. https://support.apple.com/guide/mac-help/ take-a-screenshot-mh26782/mac

9. https://developer.android.com/reference/android/media/projection/ MediaProjection

<!-- Footnote -->

---


<!-- Meanless: JOURNAL OF IATEX CLASS FILES, DECEMBER 2024<br>15 -->

<!-- Media -->

TABLE 4: Key toolkits for collecting GUI environment data.

<table><tr><td>Tool</td><td>Platform</td><td>Environment</td><td>Accessible Information</td><td>Highlight</td><td>Link</td></tr><tr><td>Selenium</td><td>Web</td><td>Browser platform) (Cross-</td><td>DOM elements, HTML structure, CSS properties</td><td>Extensive browser support and automation capabilities</td><td>https://www.selenium.dev/</td></tr><tr><td>Puppeteer</td><td>Web</td><td>Browser Firefox) (Chrome,</td><td>DOM elements, HTML/CSS. network requests</td><td>Headless browser automa tion with rich API</td><td>https://pptr.dev/</td></tr><tr><td>Playwright</td><td>Web</td><td>Browser platform) (Cross-</td><td>DOM elements, HTML/CSS. network interactions</td><td>Multi-browser support with automation and testing capabilities</td><td>https://playwright.dev/</td></tr><tr><td>TestCafe</td><td>Web</td><td>Browser platform) (Cross-</td><td>DOM elements, HTML structure, CSS properties</td><td>Easy setup with JavaScript/- TypeScript support</td><td>https://testcafe.io/</td></tr><tr><td>BeautifulSoup</td><td>Web</td><td>HTML Parsing</td><td>HTML content, DOM elements</td><td>Python library for parsing HTML and XML documents</td><td>https://www.crummy.com/software/ BeautifulSoup/</td></tr><tr><td>Protractor</td><td>Web</td><td>Browser (Angular)</td><td>DOM elements, Angular-specific attributes</td><td>Designed for Angular appli cations, integrates with Selenium</td><td>https://www.protractortest.org/</td></tr><tr><td>WebDriverIO</td><td>Web</td><td>Browser platform) (Cross-</td><td>DOM elements, HTML/CSS network interactions</td><td>Highly extensible with a vast plugin ecosystem</td><td>https://webdriver.io/</td></tr><tr><td>Ghost Inspector</td><td>Web</td><td>Browser platform) (Cross-</td><td>DOM elements, screenshots, test scripts</td><td>Cloud-based automated browser testing and monitoring</td><td>https://ghostinspector.com/</td></tr><tr><td>Cypress</td><td>Web</td><td>Browser platform) (Cross-</td><td>DOM elements, HTML/CSS. network requests</td><td>Real-time reloads and interactive debugging</td><td>https://www.cypress.io/</td></tr><tr><td>UIAutomator</td><td>Mobile</td><td>Android</td><td>UI hierarchy, widget properties, screen content</td><td>Native Android UI testing framework</td><td>https://developer.android.com/training/ testing/ui-automator</td></tr><tr><td>Espresso</td><td>Mobile</td><td>Android</td><td>UI components, view hierarchy, widget properties</td><td>Google's native Android UI testing framework</td><td>https://developer.android.com/training/ testing/espressc</td></tr><tr><td>Android View Hierarchy</td><td>Mobile</td><td>Android</td><td>UI hierarchy, widget properties, layout information</td><td>View hierarchy accessible via developer tools</td><td>https://developer.android.com/studio/ debua/layout-inspector</td></tr><tr><td>iOS Accessibility Inspector</td><td>Mobile</td><td>iOS</td><td>Accessibility tree, UI elements, properties</td><td>Tool for inspecting iOS app UI elements</td><td>https://developer.apple.com/ <br> documentation/accessibility/ <br> accessibility-inspector</td></tr><tr><td>XCUITest</td><td>Mobile</td><td>iOS</td><td>UI elements, accessibility properties, view hierarchy</td><td>Apple's iOS UI testing framework</td><td>https://developer.apple.com/ <br> documentation/xctest/user interface tests</td></tr><tr><td>Flutter Driver</td><td>Mobile</td><td>Flutter apps</td><td>Widget tree, properties, interactions</td><td>Automation for Flutter applications</td><td>https://flutter.dev/docs/testing</td></tr><tr><td>Android's Me-diaProjection API</td><td>Mobile</td><td>Android</td><td>Screenshots, screen recording</td><td>Capturing device screen content programmatically</td><td>https://developer.android.com/reference/ android/media/projection/MediaProjection</td></tr><tr><td>Windows UI Automation</td><td>Computer</td><td>Windows</td><td>Control properties, widget trees, accessibility tree</td><td>Native Windows support with OS integration</td><td>https://docs.microsoft.com/windows/win32/ winauto/entry-uiauto-win32</td></tr><tr><td>Sikuli</td><td>Computer</td><td>Windows macOS Linux</td><td>Screenshots (image recognition), UI elements</td><td>Image-based automation using computer vision</td><td>http://sikulix.com/</td></tr><tr><td>Autolt</td><td>Computer</td><td>Windows</td><td>Window titles, control properties, coordinates</td><td>Scripting language for Windows GUI automation</td><td>https://www.autoitscript.com/site/autoit/</td></tr><tr><td>Inspect.exe</td><td>Computer</td><td>Windows</td><td>UI elements, control properties, accessibility tree</td><td>Tool for inspecting Windows UI elements</td><td>https://docs.microsoft.com/windows/win32/ winauto/inspect-objects</td></tr><tr><td>macOS Accessibility API</td><td>Computer</td><td>macOS</td><td>Accessibility tree, UI elements, control properties</td><td>macOS support for accessibility and UI automation</td><td>https://developer.apple.com/accessibility/</td></tr><tr><td>Pywinauto</td><td>Computer</td><td>Windows</td><td>Control properties, UI hierarchy, window information</td><td>Python-based Windows GUI automation</td><td>https://pwww.electronis.org/docs/latest/</td></tr><tr><td>Electron Inspector</td><td>Computer</td><td>Electron apps</td><td>DOM elements, HTML/CSS JavaScript state</td><td>Tool for Electron applications</td><td>https://www.electronis.org/docs/latest/ tutorial/automated-testing</td></tr><tr><td>Windows Snipping Tool</td><td>Computer</td><td>Windows</td><td>Screenshots</td><td>Tool for capturing screen-shots in Windows</td><td>https://www.microsoft.com/en-us/windows/ tips/snipping-tool</td></tr><tr><td>macOS Screenshot Utility</td><td>Computer</td><td>macOS</td><td>Screenshots, screen recording</td><td>Tool for capturing screen shots and recording screen</td><td>https://support.apple.com/guide/ <br> mac-help/take-a-screenshot-or% <br> 2Dscreen-recording%2Dmh26782/mac</td></tr><tr><td>AccessKit</td><td>Cross-Platform</td><td>Various OS</td><td>Accessibility tree, control properties, roles</td><td>Standardized APIs across platforms</td><td>https://github.com/AccessKit/accesskit</td></tr><tr><td>Appium</td><td>Cross-Platform</td><td>Android, iOS, Windows, macOS</td><td>UI elements, accessibility properties, gestures</td><td>Mobile automation framework</td><td>https://appium.io/</td></tr><tr><td>Robot Framework</td><td>Cross-Platform</td><td>Web, Mobile, Desktop</td><td>UI elements, DOM, screen-shots</td><td>Extensible with various libraries</td><td>https://robotframework.org/</td></tr><tr><td>Cucumber</td><td>Cross-Platform</td><td>Web, Mobile, Desktop</td><td>Step definitions, UI interactions</td><td>BDD framework supporting automation tools</td><td>https://scucumber.io/</td></tr><tr><td>TestComplete</td><td>Cross-Platform</td><td>Web, Mobile, Desktop</td><td>UI elements, DOM, control properties</td><td>Tool with extensive feature set</td><td>https://smartbear.com/product/ <br> testcomplete/overview/</td></tr><tr><td>Katalon Studio</td><td>Cross-Platform</td><td>Web, Mobile, Desktop</td><td>UI elements, DOM, screen-shots</td><td>All-in-one automation solution</td><td>https://www.katalon.com/</td></tr><tr><td>Ranorex</td><td>Cross-Platform</td><td>Web, Mobile, Desktop</td><td>UI elements, DOM, control properties</td><td>Tool with strong reporting fea tures</td><td>https://www.ranorex.com/</td></tr><tr><td>Applitools</td><td>Cross-Platform</td><td>Web, Mobile, Desktop</td><td>Screenshots, visual check points, DOM elements</td><td>AI-powered visual testing</td><td>https://applitools.com/</td></tr></table>

<!-- Media -->

---

<!-- Footnote -->

10. https://developer.apple.com/library/archive/documentation/

Accessibility/Conceptual/AccessibilityMacOSX/

<!-- Footnote -->

---


<!-- Meanless: JOURNAL OF LATEX CLASS FILES, DECEMBER 2024<br>16 -->

3) UI Element Properties: Each UI element in the interface contains specific properties, such as control type, label text, position, and bounding box dimensions, that help agents target the appropriate components. These properties are instrumental for agents to make decisions about spatial relationships (e.g., adjacent elements) and functional purposes (e.g., distinguishing between buttons and text fields). For instance, web applications reveal properties like DOM attributes (id, class, name) and CSS styles that provide context and control information. These attributes assist agents in pinpointing precise elements for interaction, enhancing their ability to navigate and operate within diverse UI environments. Figure 9 illustrates examples of selected UI element properties extracted by the Windows UI Automation API, which support GUI agents in decision-making.

4) Complementary CV Approaches: When structured information is incomplete or unavailable, computer vision techniques can provide additional insights [181]. For instance, OCR allows agents to extract text content directly from screenshots, facilitating the reading of labels, error messages, and instructions [121]. Furthermore, advanced object detection [120] models like SAM (Segment Anything Model) [182], DINO [183] and OmniParser 184 can identify and classify UI components in various layouts, supporting the agent in dynamic environments where UI elements may frequently change. These vision-based methods ensure robustness, enabling agents to function effectively even in settings where standard UI APIs are insufficient. We illustrate an example of this complementary information in Figure 10 and further detail these advanced computer vision approaches in Section 5.7.1

Together, these elements create a comprehensive, multimodal representation of the GUI environment's current state, delivering both structured and visual data. By incorporating this information into prompt construction, agents are empowered to make well-informed, contextually aware decisions without missing critical environmental cues.

#### 5.2.3 Environment Feedback

Effective feedback mechanisms are essential for GUI agents to assess the success of each action and make informed decisions for subsequent steps. Feedback can take several forms, depending on the platform and interaction type. Figure 11 presents examples of various types of feedback obtained from the environment.

1) Screenshot Update: By comparing before-and-after screenshots, agents can identify visual differences that signify state changes in the application. Screenshot analysis can reveal subtle variations in the interface, such as the appearance of a notification, visual cues, or confirmation messages, that may not be captured by structured data [185].

2) UI Structure Change: After executing an action, agents can detect modifications in the widget tree structure, such as the appearance or disappearance of elements, updates to element properties, or hierarchical shifts [186]. These changes indicate successful interactions (e.g., opening a dropdown or clicking a button) and help the agent determine the next steps based on the updated environment state.

3) Function Return Values and Exceptions: Certain platforms offer direct feedback on action outcomes through function return values or system-generated exceptions 187 . For example, API responses or JavaScript return values can confirm action success on web platforms, while exceptions or error codes can signal failed interactions, guiding the agent to retry or select an alternative approach.

These feedback provided by the environment is crucial for GUI agents to assess the outcomes of their previous actions. This real-time information enables agents to evaluate the effectiveness of their interventions and determine whether to adhere to their initial plans or pivot towards alternative strategies. Through this process of self-reflection, agents can adapt their decision-making, optimizing task execution and enhancing overall performance in dynamic and varied application environments.

### 5.3 Prompt Engineering

In the operation of LLM-powered GUI agents, effective prompt construction is a crucial step that encapsulates all necessary information for the agent to generate appropriate responses and execute tasks successfully [168]. After gathering the relevant data from the environment, the agent formulates a comprehensive prompt that combines various components essential for inference by the LLM. Each component serves a specific purpose, and together they enable the agent to execute the user's request efficiently. Figure 12 illustrates a basic example of prompt construction in an LLM-brained GUI agent. The key elements of the prompt are summarized as follows:

1) User Request: This is the original task description provided by the user, outlining the objective and desired outcome. It serves as the foundation for the agent's actions and is critical for ensuring that the LLM understands the context and scope of the task.

2) Agent Instruction: This section provides guidance for the agent's operation, detailing its role, rules to follow, and specific objectives. Instructions clarify what inputs the agent will receive and outline the expected outputs from the LLM, establishing a framework for the inference process. The core agent instructions are usually embedded within the base system prompt of the LLM, with supplementary instructions dynamically injected or updated based on environmental feedback and contextual adaptation.

3) Environment States: The agent includes perceived GUI screenshots and UI information, as introduced in Section 5.2.2. This multimodal data may consist of various versions of screenshots (e.g., a clean version and a SoM annotated version) to ensure clarity and mitigate the risk of UI controls being obscured by annotations. This comprehensive representation of the environment is vital for accurate decision-making.

4) Action Documents: This component outlines the available actions the agent can take, detailing relevant documentation, function names, arguments, return values, and any other necessary parameters. Providing this 5.4 information equips the LLM with the context needed to select and generate appropriate actions for the task at hand.


<!-- Meanless: JOURNAL OF LATEX CLASS FILES, DECEMBER 2024<br>17 -->

<!-- Media -->

<!-- figureText: Widgets detected by APIs<br>95 77<br>125<br>122<br>55<br>CV-Detected Widgets Information<br>${29.30313233103}.{104.34105i1064142}.{434445110i112}$<br>Sensitivity<br>...<br>foregraph.<br>Voice<br>Sensing<br>123<br>Id: 1, type: icon, label: Page 4 thumbnail<br>Single Agent vs. MultiAgent<br>Single Agent vs. MultiAgent<br>ld: 2, type: icon, label: Page 5 thumbnail<br>Single Agent<br>Features: Collaboration<br>Id: 3, type: icon, label: Page 6 thumbnail<br>Plot the stock price for MSFT..<br>Build a mobile application for stock transaction...<br>ld: 4, type: icon, label: Page 7 thumbnail<br>ld: 5, type: editbox, label: Canvas<br>Widgets detected by CV models -->

<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2025/11/2025_11_04__12_39_04_5e4beb.jpg"/>

Fig. 10: An example illustrating the use of a CV approach to parse a PowerPoint GUI and detect non-standard widgets, inferring their types and labels.

<!-- Media -->

5) Demonstrated Examples: Including example input/output pairs is essential to activate the in-context learning 97 capability of the LLM. These examples help the model comprehend and generalize the task requirements, enhancing its performance in executing the GUI agent's responsibilities.

6) Complementary Information: Additional context that aids in planning and inference may also be included. This can consist of historical data retrieved from the agent's memory (as detailed in Section 5.6) and external knowledge sources, such as documents obtained through retrieval-augmented generation (RAG) methods 188], [189]. This supplemental information can provide valuable insights that further refine the agent's decision-making processes.

The construction of an effective prompt is foundational for the performance of LLM-powered GUI agents. By systematically incorporating aforementioned information, the agent ensures that the LLM is equipped with the necessary context and guidance to execute tasks accurately and efficiently.

### 5.4 Model Inference

The constructed prompt is submitted to the LLM for inference, where the LLM is tasked with generating both a plan and the specific actions required to execute the user's request. This inference process is critical as it dictates how effectively the GUI agent will perform in dynamic environments. It typically involves two main components: planning and action inference, as well as the generation of complementary outputs. Figure 13 shows an example of the LLM's inference output.

#### 5.4.1 Planning

Successful execution of GUI tasks often necessitates a series of sequential actions, requiring the agent to engage in effective planning [52], [190]. Analogous to human cognitive processes, thoughtful planning is essential to organize tasks, schedule actions, and ensure successful completion [51], 191]. The LLM must initially conceptualize a long-term goal while simultaneously focusing on short-term actions to initiate progress toward that goal 192.

To effectively navigate the complexity of multi-step tasks, the agent should decompose the overarching task into manageable subtasks and establish a timeline for their execution 193]. Techniques such as CoT reasoning [100] can be employed, enabling the LLM to develop a structured plan that guides the execution of actions. This plan, which can be stored for reference during future inference steps, enhances the organization and focus of the agent's activities.

The granularity of planning may vary based on the nature of the task and the role of the agent [51]. For complex tasks, a hierarchical approach that combines global planning (identifying broad subgoals) with local planning (defining detailed steps for those subgoals) can significantly improve the agent's ability to manage long-term objectives effectively 194].


<!-- Meanless: JOURNAL OF LATEX CLASS FILES, DECEMBER 2024<br>18 -->

<!-- Media -->

<!-- figureText: After<br>Before<br>Video<br>Audio<br>Screen Recording<br>Click on the "Video"<br>Video<br>Screen Recording<br>Insert Video From<br>\{ "click_success" : True\}<br>口味<br>Media<br>This Device...<br>Stock Videos...<br>Recordinc<br>Online Videos.<br>Function Return Values and Exceptions<br>Screenshot Update<br>Menultem This Device .<br>Menultem - Video<br>Menultem - Video<br>Menultem - Video<br>Menultem Stock Videos<br>Groupbox - Media<br>Menultem - Audio<br>Groupbox - Media<br>MenuItem - Audio<br>Menultem Online Videos ...<br>Difference<br>Button - Screen Recording<br>Button - Screen Recording<br>Before<br>After<br>UI Structure Change -->

<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2025/11/2025_11_04__12_39_04_805d5a.jpg"/>

Fig. 11: Examples of various types of feedback obtained from a PowerPoint application environment.

<!-- Media -->

#### 5.4.2 Action Inference

Action inference is the core objective of the inference stage, as it translates the planning into executable tasks. The inferred actions are typically expressed as function call strings, encompassing the function name and relevant parameters. These strings can be readily converted into real-world interactions with the environment, such as clicks, keyboard inputs, mobile gestures, or API calls. A detailed discussion of these action types is presented in Section 5.5

The input prompt must include a predefined set of actions available for the agent to select from. The agent can choose an action from this set or, if allowed, generate custom code or API calls to interact with the environment [161]. This flexibility can enhance the agent's adaptability to unforeseen circumstances; however, it may introduce reliability concerns, as the generated code may be prone to errors.

<!-- Media -->

10. https://developer.apple.com/library/archive/documentation/ AppleScript/Conceptual/AppleScriptLangGuide/introduction/ASLR_ intro.html

---

	11. https://www.macosxautomation.com/automator/

	12. https://docs.blender.org/manual/en/latest/sculpt_paint/sculpting/

introduction/gesture tools.html

	13. https://developer.android.com/reference/android/speech/

SpeechRecognizer

	14. https://developer.apple.com/documentation/sirikit/

	15. https://pypi.org/project/pyperclip/

	16. https://clipboardjs.com/

	17. https://developer.android.com/develop/sensors-and-location/

sensors/sensors overview

	18. https://learn.microsoft.com/en-us/previous-versions/office/

office-365-api/

	19. https://developer.android.com/reference

	20. https://developer.apple.com/ios/

	21. https://learn.microsoft.com/en-us/windows/win32/api/

	22. https://developer.apple.com/library/archive/documentation/

Cocoa/Conceptual/CocoaFundamentals/WhatlsCocoa/WhatlsCocoa.

	23. https://developer.mozilla.org/en-US/docs/Web/API/Fetch_API

	24. https://axios-http.com/docs/api_intro

	25. https://platform.openai.com/docs/overview

---

<!-- Media -->

#### 5.4.3 Complementary Outputs

In addition to planning and action inference, the LLM can also generate complementary outputs that enhance the agent's capabilities. These outputs may include reasoning processes that clarify the agent's decision-making (e.g., CoT reasoning), messages for user interaction, or communication with other agents or systems, or the status of the task (e.g., continue or finished). The design of these functionalities can be tailored to meet specific needs, thereby enriching the overall performance of the GUI agent.

By effectively balancing planning and action inference while incorporating complementary outputs, agents can navigate complex tasks with a higher degree of organization and adaptability.

### 5.5 Actions Execution

Following the inference process, a crucial next step is for the GUI agent to execute the actions derived from the inferred commands within the GUI environment and subsequently gather feedback. Although the term "GUI agent" might suggest a focus solely on user interface actions, the action space can be greatly expanded by incorporating various toolboxes that enhance the agent's versatility. Broadly, the actions available to GUI agents fall into three main categories: (i) UI operations [144], (ii) native API calls [196], and (iii) AI tools [197]. Each category offers unique advantages and challenges, enabling the agent to tackle a diverse range of


<!-- Meanless: JOURNAL OF LATEX CLASS FILES, DECEMBER 2024<br>19 -->

<!-- Media -->

<!-- figureText: Output<br>Action:<br>Click(button="left", double=False,<br>object = "Remove Background")<br>Inference<br>Plan:<br>LLM<br>1. After clicking the "Remove Background", I will click the "Transparency" menu.<br>2. I will click the "Transparency 50%" to adjust the transparency of the picture.<br>Complementary Outputs:<br>Thought:<br>Status: Continue<br>clicking the "Remove Background". -->

<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2025/11/2025_11_04__12_39_04_8fc7fb.jpg"/>

Fig. 13: An example of the LLM's inference output in a GUI agent.

<!-- Media -->

<!-- Media -->

<!-- figureText: Prompt Engineering<br>User Request:<br>Action Information:<br>Create a PowerPoint slide based on ...<br>click:<br>summary: -<br>Agent Instruction:<br>API call: click(button: str, double: bool)<br>] Args:<br>You are a GUI agent, your task is to ... Please follow the guidelines below ...<br>button: The mouse button to click. One of 'left'.<br>Tright", (Default: "left")<br>- double: Whether to perform a double click or not (Default: False)'<br>1. ...<br>[3] Return: bool<br>2. ...<br>Environment State:<br>Demonstrated Examples:<br>40<br>example1:<br>Request:<br>Send a message to Tom on Teams to ask him if he can join the meeting at 3pm.<br>"New" "position": $\left\lbrack {{0.45},{0.78}}\right\rbrack$ \}<br>\{ "type": Edit, "title":<br>Responses<br>"Document", "position": [0.87, 0.43])<br>\{ "type": Botton, "title":<br>Thought: |-<br>"Design" "position": $\left\lbrack {{0.25},{0.21}}\right\rbrack$<br>\{ "type":ComboBox, "title",<br>Screenshots<br>Widget Tree<br>UI Element Properties<br>in the control named 'Type a new message' for the current action.<br>Control: |-<br>Type a new message<br>Function: -<br>text<br>Args:<br>Complementary Information:<br>\{"text": "Hello Tom. It's 3 PM. Are you available to join the meeting now?"\}<br>Status: -<br>CONTINUE<br>Memory<br>External Knowledge<br>...<br>Plan:<br>- (1) Click the Send button to send the message. -->

<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2025/11/2025_11_04__12_39_04_41a238.jpg"/>

Fig. 12: A basic example of prompt construction in a LLM-brained GUI agent.

<!-- Media -->

tasks more effectively. We summarize the various actions commonly used in GUI agents, categorized into distinct types, in Table 5 and provide detailed explanations of each category below.

#### 5.5.1 UI Operations

UI operations encompass the fundamental interactions that users typically perform with GUIs in software applications. These operations include various forms of input, such as mouse actions (clicks, drags, hovers), keyboard actions (key presses, combinations), touch actions (taps, swipes), and gestures (pinching, rotating). The specifics of these actions may differ across platforms and applications, necessitating a tailored approach for each environment.

While UI operations form the foundation of agent interactions with the GUI, they can be relatively slow due to the sequential nature of these tasks. Each operation must be executed step by step, which can lead to increased latency, especially for complex workflows that involve numerous interactions. Despite this drawback, UI operations are crucial for maintaining a broad compatibility across various applications, as they leverage standard user interface elements and interactions.

#### 5.5.2 Native API Calls

In contrast to UI operations, some applications provide native APIs that allow GUI agents to perform actions more efficiently. These APIs offer direct access to specific functionalities within the application, enabling the agent to execute complex tasks with a single command 198]. For instance, calling the Outlook API allows an agent to send an email in one operation, whereas using UI operations would require a series of steps, such as navigating through menus and filling out forms [199].

While native APIs can significantly enhance the speed and reliability of action execution, their availability is limited. Not all applications or platforms expose APIs for external use, and developing these interfaces can require substantial effort and expertise. Consequently, while native APIs present a powerful means for efficient task completion, they may not be as generalized across different applications as UI operations.


<!-- Meanless: JOURNAL OF LATEX CLASS FILES, DECEMBER 2024<br>20 -->

<!-- Media -->

TABLE 5: Overview of actions for GUI agents.

<table><tr><td>Action</td><td>Category</td><td>Original Executor</td><td>Examples</td><td>Platform</td><td>Environment</td><td>Toolkit</td></tr><tr><td>Mouse actions</td><td>UI Operations</td><td>Mouse</td><td>Click, scroll, hover, drag</td><td>Computer</td><td>Windows</td><td>UI Automation 6, Pywinauto</td></tr><tr><td>Mouse actions</td><td>UI Operations</td><td>Mouse</td><td>Click, scroll, hover, drag</td><td>Computer</td><td>macOS</td><td>AppleScript 10, Automator 11</td></tr><tr><td>Mouse actions</td><td>UI Operations</td><td>Mouse</td><td>Click, scroll, hover, drag</td><td>Web</td><td>Browser</td><td>Selenium, Puppeteer</td></tr><tr><td>Keyboard actions</td><td>UI Operations</td><td>Keyboard</td><td>Typing, key presses, shortcuts</td><td>Computer</td><td>Windows</td><td>UI Automation 6. Pywinauto</td></tr><tr><td>Keyboard actions</td><td>UI Operations</td><td>Keyboard</td><td>Typing, key presses, shortcuts</td><td>Computer</td><td>macOS</td><td>AppleScript 10, Automator 1</td></tr><tr><td>Keyboard actions</td><td>UI Operations</td><td>Keyboard</td><td>Typing, key presses, shortcuts</td><td>Web</td><td>Browser</td><td>Selenium. Puppeteer</td></tr><tr><td>Touch actions</td><td>UI Operations</td><td>Touchscreen</td><td>Tap, swipe, pinch, zoom</td><td>Mobile</td><td>Android</td><td>Appium, UIAu-tomator</td></tr><tr><td>Touch actions</td><td>UI Operations</td><td>Touchscreen</td><td>Tap, swipe, pinch, zoom</td><td>Mobile</td><td>iOS</td><td>Appium, XCUITest</td></tr><tr><td>Gesture actions</td><td>UI Operations</td><td>User hand</td><td>Rotate, multi-finger gestures</td><td>Mobile</td><td>Android, iOS</td><td>Appium , G tureTools 12</td></tr><tr><td>Voice commands</td><td>UI Operations</td><td>User voice</td><td>Speech input, voice commands</td><td>Mobile</td><td>Android</td><td>SpeechRecognize 13</td></tr><tr><td>Voice commands</td><td>UI Operations</td><td>User voice</td><td>Speech input, voice commands</td><td>Mobile</td><td>iOS</td><td>SiriKit 14</td></tr><tr><td>Clipboard operations</td><td>UI Operations</td><td>System clipboard</td><td>Copy, paste</td><td>Cross-platform</td><td>Cross-OS</td><td>Pyperclip Clipboard.js 16</td></tr><tr><td>Screen interactions</td><td>UI Operations</td><td>User</td><td>Screen rotation, shake</td><td>Mobile</td><td>Android, iOS</td><td>Device sensors APIs 17</td></tr><tr><td>Shell Commands</td><td>Native API Calls</td><td>Command Line Interface</td><td>File manipulation, system operations, script execution</td><td>Computer</td><td>Unix/Linux, macOS</td><td>Bash, Terminal</td></tr><tr><td>Application APIs</td><td>Native API Calls</td><td>Application APIs</td><td>Send email, create document. fetch data</td><td>Computer</td><td>Windows</td><td>Microsoft Office COM APIs 18</td></tr><tr><td>Application APIs</td><td>Native API Calls</td><td>Application APIs</td><td>Access calendar. send messages</td><td>Mobile</td><td>Android</td><td>Android SDK APIs 19</td></tr><tr><td>Application APIs</td><td>Native API Calls</td><td>Application APIs</td><td>Access calendar, send messages</td><td>Mobile</td><td>iOS</td><td>iOS SDK APIs 20</td></tr><tr><td>System APIs</td><td>Native API Calls</td><td>System APIs</td><td>File operations, network requests</td><td>Computer</td><td>Windows</td><td>Win32 API 21</td></tr><tr><td>System APIs</td><td>Native API Calls</td><td>System APIs</td><td>File operations, network requests</td><td>Computer</td><td>macOS</td><td>Cocoa APIs 22</td></tr><tr><td>Web APIs</td><td>Native API Calls</td><td>Web Services</td><td>Fetch data, submit forms</td><td>Web</td><td>Browser</td><td>Fetch API ${}^{23}$ . Axios 24</td></tr><tr><td>AI Models</td><td>AI Tools</td><td>AI Models</td><td>Screen understanding, summarization, image generation</td><td>Cross-platform</td><td>Cross-OS</td><td>DALL·E 1951 OpenAI APIs 26</td></tr></table>

<!-- Media -->

#### 5.5.3 AI Tools

The integration of AI tools into GUI agents represents a transformative advancement in their capabilities. These tools can assist with a wide range of tasks, including content summarization from screenshots or text, document enhancement, image or video generation (e.g., calling ChatGPT [11], DALL-E 195]), and even invoking other agents or Copilot tools for collaborative assistance. The rapid development of generative AI technologies enables GUI agents to tackle complex challenges that were previously beyond their capabilities.

By incorporating AI tools, agents can extend their functionality and enhance their performance in diverse contexts. For example, a GUI agent could use an AI summarization tool to quickly extract key information from a lengthy document or leverage an image generation tool to create custom visuals for user presentations. This integration not only streamlines workflows but also empowers agents to deliver high-quality outcomes in a fraction of the time traditionally required.

#### 5.5.4 Summary

An advanced GUI agent should adeptly leverage all three categories of actions: UI operations for broad compatibility, native APIs for efficient execution, and AI tools for enhanced capabilities. This multifaceted approach enables the agent to operate reliably across various applications while maximizing efficiency and effectiveness. By skillfully navigating these action types, GUI agents can fulfill user requests more proficiently, ultimately leading to a more seamless and productive user experience.

### 5.6 Memory

For a GUI agent to achieve robust performance in complex, multi-step tasks, it must retain memory, enabling it to manage states in otherwise stateless environments. Memory allows the agent to track its prior actions, their outcomes, and the task's overall status, all of which are crucial for informed decision-making in subsequent steps [200]. By establishing continuity, memory transforms the agent from a reactive system into a proactive, stateful one, capable of self-adjustment based on accumulated knowledge. The agent's memory is generally divided into two main types: Short-Term Memory [201] and Long-Term Memory [202]. We show an overview of different types of memory in GUI agents in Table 6


<!-- Meanless: 21 -->

<!-- Media -->

<!-- figureText: Task A: Play the Recap of Arcane Season 1.<br>Task B: Download the game related to the video just played.<br>Step1<br>Step1<br>Plan 1:<br>Plan 1:<br>(1) I need to use google to search<br>(1) The previous task play the video of<br>"Arcane", which is related to the "League of Legend", which I should search for.<br>“Recap of Arcane Season 1” .<br>(2) Find the top searched result of video and click.<br>(3) Play the video.<br>Step2<br>(2) Enter the official site of the game. (3) Click Download.<br>Step2<br>Plan 2:<br>Plan 2:<br>(1) I need to click the "Arcane Season 1 Recap | Netflix".<br>(1) I need to click the first result of "League of Legend" to enter the website.<br>(2) Play the video just entered.<br>(2) Click Download.<br>Step3<br>Step3<br>GUI Agent<br>GUI Agent<br>Plan 3:<br>Plan 3:<br>(1) Click the "Play" button to play the video.<br>(1) Click Download for Windows to download the game.<br>Short-term Memory:<br>Short-term Memory:<br>Input("Arcane Season1 Recap”)<br>Input("Arcane<br>Season1 Recap”)<br>Click("Arcane Season 1 | Recap | Netflix" )<br>Click("Play")<br>Click("League of Legend")<br>Click("Download for Windows"、<br>Click("Search")<br>Click("Search")<br>Plan 1<br>Plan 2<br>Plan 3<br>Plan 1<br>Plan 2<br>Plan 3<br>Step 2<br>Step 3<br>Step 2<br>Step 3<br>Long-term Memory:<br>Task 1: I just used Google to search for the "Recap of Arcane Season 1", and play the top searched results on YouTube -->

<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2025/11/2025_11_04__12_39_04_3303f3.jpg"/>

Fig. 14: Illustration of short-term memory and long-term memory in an LLM-brained GUI agent.

TABLE 6: Summary of memory in GUI agents.

<table><tr><td>Memory Element</td><td>Memory Type</td><td>Description</td><td>Storage Medium/Method</td></tr><tr><td>Action</td><td>Short-term</td><td>Historical actions trajectory taken in the environment</td><td>In-memory, Context window</td></tr><tr><td>Plan</td><td>Short-term</td><td>Plan passed from previous step</td><td>In-memory, Context window</td></tr><tr><td>Execution Results</td><td>Short-term</td><td>Return values, error traces, and other environmental feedback</td><td>In-memory, Context window</td></tr><tr><td>Environment State</td><td>Short-term</td><td>Important environment state data, e.g., UI elements</td><td>In-memory, Context window</td></tr><tr><td>Self-experience</td><td>Long-term</td><td>Task completion trajectories from historical tasks</td><td>Database, Disk</td></tr><tr><td>Self-guidance</td><td>Long-term</td><td>Guidance and rules summarized from historical trajectories</td><td>Database, Disk</td></tr><tr><td>External Knowledge</td><td>Long-term</td><td>Other external knowledge sources aiding task completion</td><td>External Knowledge Base</td></tr><tr><td>Task Success Metrics</td><td>Long-term</td><td>Metrics from task success or failure rates across sessions</td><td>Database, Disk</td></tr></table>

<!-- Media -->

#### 5.6.1 Short-Term Memory

Short-Term Memory (STM) provides the primary, ephemeral context used by the LLM during runtime [203]. STM stores information pertinent to the current task, such as recent plans, actions, results, and environmental states, and continuously updates to reflect the task's ongoing status. This memory is particularly valuable in multi-step tasks, where each decision builds on the previous one, requiring the agent to maintain a clear understanding of the task's trajectory. As illustrated in Figure 14 during the completion of independent tasks, the task trajectory, comprising actions and plans-is stored in the STM. This allows the agent to track task progress effectively and make more informed decisions.

However, STM is constrained by the LLM's context window, limiting the amount of information it can carry forward. To manage this limitation, agents can employ selective memory management strategies, such as selectively discarding or summarizing less relevant details to prioritize the most impactful information. Despite its limited size, STM is essential for ensuring coherent, contextually aware interactions and supporting the agent's capacity to execute complex workflows with immediate, relevant feedback.

#### 5.6.2 Long-Term Memory

Long-Term Memory (LTM) serves as an external storage repository for contextual information that extends beyond the immediate runtime 204. Unlike STM, which is transient, LTM retains historical task data, including previously completed tasks, successful action sequences, contextual tips, and learned insights. LTM can be stored on disk or in a database, enabling it to retain larger volumes of information than what is feasible within the LLM's immediate context window. In the example shown in Figure 14 when the second task requests downloading a game related to the previous task, the agent retrieves relevant information from its LTM. This enables the agent to accurately identify the correct game, facilitating efficient task completion.

LTM contributes to the agent's self-improvement over time by preserving examples of successful task trajectories, operational guidelines, and common interaction patterns. When approaching a new task, the agent can leverage RAG techniques to retrieve relevant historical data, which enhances its ability to adapt strategies based on prior success. This is similar to the lifelong learning [53], which makes LTM instrumental in fostering an agent's capacity to "learn" from experience, enabling it to perform tasks with greater accuracy and efficiency as it accumulates insights across sessions. For instance, [205] provides an illustrative example of using past task trajectories stored in memory to guide and enhance future decision-making, a technique that is highly adaptable for GUI agents. It also enables better personalization by retaining information about previous tasks.


<!-- Meanless: JOURNAL OF LATEX CLASS FILES, DECEMBER 2024<br>22 -->

### 5.7 Advanced Enhancements

While most LLM-brained GUI agents incorporate fundamental components such as perception, planning, action execution, and memory, several advanced techniques have been developed to significantly improve the reasoning and overall capabilities of these agents. Here, we outline shared advancements widely adopted in research to guide the development of more specialized and capable LLM-brained GUI agents.

#### 5.7.1 Computer Vision-Based GUI Grounding

Although various tools (Section 4) enable GUI agents to access information like widget location, captions, and properties, certain non-standard GUIs or widgets may not adhere to these tools' protocols [243], rendering their information inaccessible. Additionally, due to permission management, these tools are not always usable. Such incomplete information can present significant challenges for GUI agents, as the LLM may need to independently locate and interact with required widgets by estimating their coordinates to perform actions like clicking—a task that is inherently difficult without precise GUI data.

CV models offer a non-intrusive solution for GUI grounding directly from screenshots, enabling the detection, localization, segmentation, and even functional estimation of widgets [103], 244-246]. This approach allows agents to interpret the visual structure and elements of the GUI without relying on system-level tools or internal metadata, which may be unavailable or incomplete. CV-based GUI parsing provides agents with valuable insights into interactive components, screen layout, and widget functionalities based solely on visual cues, enhancing their ability to recognize and act upon elements on the screen. Figure 10 provides an illustrative example of how a CV-based GUI parser works. While standard API-based detection captures predefined widgets, the CV model can identify additional elements, such as thumbnails and canvases, which may not have explicit API representations in the PowerPoint interface. This enhances widget recognition, allowing the agent to detect components beyond the scope of API detection. We show an overview of related GUI grounding models and benchmarks in Table 7 8 and 9

A notable example is OmniParser [184], which implements a multi-stage parsing technique involving a fine-tuned model for detecting interactable icons, an OCR module for extracting text, and an icon description model that generates localized semantic descriptions for each UI element. By integrating these components, OmniParser constructs a structured representation of the GUI, enhancing an agent's understanding of interactive regions and functional elements. This comprehensive parsing strategy has shown to significantly improve GPT-4V's screen comprehension and interaction accuracy.

Such CV-based GUI grounding layers provide critical grounding information that significantly enhances an agent's ability to interact accurately and intuitively with diverse GUIs. This is particularly beneficial for handling custom or nonstandard elements that deviate from typical accessibility protocols. Additionally, prompting methods like iterative narrowing have shown promise in improving the widget grounding capabilities of VLMs [208]. Together, these approaches pave the way for more adaptable and resilient GUI agents, capable of operating effectively across a broader range of screen environments and application contexts.

Several works have introduced benchmarks to evaluate the GUI grounding capabilities of models and agents. For instance, ScreenSpot [25] serves as a pioneering benchmark designed to assess the GUI grounding performance of LLM-powered agents across diverse platforms, including iOS, Android, macOS, Windows, and web environments. It features a dataset with over 600 screenshots and 1,200 instructions, focusing on complex GUI components such as widgets and icons. This benchmark emphasizes the importance of GUI grounding in enhancing downstream tasks like web automation and mobile UI interaction. Building upon this, ScreenSpot-Pro [241] extends the scope to more professional, high-resolution environments. This evolved version includes 1,581 tasks with high-quality annotations, encompassing domains such as software development, creative tools, CAD, scientific applications, and office productivity. Key features of ScreenSpot-Pro include authentic high-resolution screenshots and meticulous annotations provided by domain experts.

These benchmarks provide critical evaluation criteria for assessing GUI grounding capabilities, thereby advancing the development of GUI agents for improved GUI understanding and interaction.

#### 5.7.2 Multi-Agent Framework

The adage "two heads are better than one" holds particular relevance for GUI automation tasks, where a single agent, though capable, can be significantly enhanced within a multi-agent framework [247], [248]. Multi-agent systems leverage the collective intelligence, specialized skills, and complementary strengths of multiple agents to tackle complex tasks more effectively than any individual agent could alone. In the context of GUI agents, multi-agent systems offer advanced capabilities through two primary mechanisms: (i) specialization and (ii) inter-agent collaboration. Figure 15 illustrates an example of how an LLM-powered multi-agent collaborates to create a desk.

1) Specialization of Agents: In a multi-agent framework, each agent is designed to specialize in a specific role or function, leveraging its unique capabilities to contribute to the overall task. As illustrated in the Figure 15 specialization enables distinct agents to focus on different aspects of the task pipeline. For instance, the "Document Extractor" specializes in extracting relevant content from local documents, such as PDFs, while the "Web Retriever" focuses on gathering additional information from online sources. Similarly, the "Designer" transforms the retrieved information into visually appealing slides, and the "Evaluator" provides feedback to refine and improve the output. This functional separation ensures that each agent becomes highly adept at its designated task, leading to improved efficiency and quality of results 249].


<!-- Meanless: JOURNAL OF LATEX CLASS FILES, DECEMBER 2024<br>23 -->

<!-- Media -->

TABLE 7: A summary of of GUI grounding models (Part I).

<table><tr><td>Model/ Benchmark</td><td>Platform</td><td>Foundation Model</td><td>Size</td><td>Dataset</td><td>Input</td><td>Output</td><td>Highlight</td><td>Link</td><td></td></tr><tr><td>OmniParser [184]</td><td>Mobile, Desktop, and Web</td><td>BLIP-2 206] YOLOv8 207]</td><td>/</td><td>67,000 UI screenshots with bounding box annotations and 7,185 icon-description pairs generated using GPT-4</td><td>UI screenshots</td><td>IDs, bounding boxes, and descriptions of in-teractable elements</td><td>Introduces a purely vision-based screen parsing framework for general UI understanding without external information, significantly improving action prediction accuracy for LLM-driven agents</td><td>https://github.com/microsoft/ OmniParser</td><td></td></tr><tr><td>Iterative Narrowing [208]</td><td>Mobile, Web, and Desktop</td><td>Qwen2-VL and OS-Atlas-Base</td><td>/</td><td>ScreenSpot 25]</td><td>A GUI screenshot and a natural language query</td><td>(x,y) coordinates representing the target location in the GUI</td><td>Progressively crops regions of the GUI to refine predictions, enhancing precision for GUI ground. ing tasks</td><td colspan="2">https://github.<br> com/ant-8/ <br> GUI-Grounding-via-Iterative-Narro</td></tr><tr><td>Iris 209]</td><td>Mobile (iOS, Android), Desktop (Windows, macOS), and Web</td><td>Qwen-VL 210]</td><td>9.6B</td><td>850K GUI-specific annotations and 150K vision-language instructions</td><td>High-resolution GUI screenshots with natural language instructions</td><td>Referring: Generates detailed descriptions of UI elements. Grounding: Locates U elements on the screen.</td><td>Information-Sensitive Cropping for efficient handling of high-resolution GUI images, and Self-Refining Dual Learning to iteratively enhance GUI grounding and referring tasks without additional annotations</td><td>/</td><td></td></tr><tr><td>Attention-driven Grounding 211]</td><td>Mobile, Web, and Desktop</td><td>MiniCPM-Llama3-V 2.5</td><td>8.5B</td><td>Mind2Web ScreenSpot VisualWebBench 213]</td><td>shots textual queries GUI screen-and user</td><td>Element localization via bounding boxes, text-to-image mapping for grounding. and actionable descriptions of GUI components</td><td>Utilizes attention mechanisms in pre-trained MLLMs without fine-tuning</td><td>https: <br> //github.com/ <br> HeimingX/TAG</td><td></td></tr><tr><td>Aria-UI 214]</td><td>Web, Desktop, and Mobile</td><td>Aria 215]</td><td>3.9B</td><td>3.9 million elements and 11.5 million samples</td><td>GUI screen-shots, user instructions, and action histories</td><td>Pixel coordinates for GUI elements and corresponding actions</td><td>A purely vision-based approach avoiding reliance on AXTree-like inputs</td><td>https: <br> //ariaui.github.io</td><td></td></tr><tr><td>UGround 216]</td><td>Web, Desktop (Windows, MacOS, Linux), Mobile (Android, iOS)</td><td>LLaVA-NeXT-7B 217]</td><td>7B</td><td>Web-Hybrid and other existing datasets</td><td>queries GUI screen-shots, user</td><td>Pixel coordinates of GUI elements</td><td>A universal GUI grounding model that relies solely on vision, eliminating the need for text-based representations</td><td>https: <br> //osu-nlp-group. <br> github.io/ <br> UGround/</td><td></td></tr><tr><td>GUI-Bee 218]</td><td>Web</td><td>SeeClick 251. QwenGUI 219], and UIX-7B 220]</td><td>7B-13B</td><td>NovelScreenSpot</td><td>GUI screen-shots, user queries. accessibility tree</td><td>GUI element grounding locations, actions and function calls, navigation steps, predicted GUI changes after interaction</td><td>Autonomously explores GUI environments, with Q-ICRL optimizing exploration efficiency and enhancing data diversity.</td><td>https://qui-bee.github.io</td><td></td></tr></table>

<!-- Media -->

2) Collaborative Inter-Agent Dynamics: The multi-agent system shown in the Figure 15 exemplifies how agents collaborate dynamically to handle complex tasks. The process begins with the "Document Extractor" and "Web Retriever", which work in parallel to collect information from local and online sources. The retrieved data is communicated to the "Designer", who synthesizes it into a cohesive set of slides. Once the slides are created, the "Evaluator" reviews the output, providing feedback for refinement. These agents share information, exchange context, and operate in a coordinated manner, reflecting a human-like teamwork dynamic. For example, as depicted, the agents' roles are tightly integrated—each output feeds into the next stage, creating a streamlined workflow that mirrors real-world collaborative environments [19].

In such a system, agents can collectively engage in tasks requiring planning, discussion, and decision-making. Through these interactions, the system taps into each agent's domain expertise and latent potential for specialization, maximizing overall performance across diverse, multi-step processes.

#### 5.7.3 Self-Reflection

"A fault confessed is half redressed". In the context of GUI multi-agent systems, self-reflection refers to the agents' capacity to introspectively assess their reasoning, actions, and decisions throughout the task execution process [250]. This capability allows agents to detect potential mistakes, adjust strategies, and refine actions, thereby improving the quality and robustness of their decisions, especially in complex or unfamiliar GUI environments. By periodically evaluating their own performance, self-reflective agents can adapt dynamically to produce more accurate and effective results [251].


<!-- Meanless: JOURNAL OF LATEX CLASS FILES, DECEMBER 2024<br>24 -->

<!-- Media -->

TABLE 8: A summary of of GUI grounding models (Part II).

<table><tr><td>Model/ Benchmark</td><td>Platform</td><td>Foundation Model</td><td>Size</td><td>Dataset</td><td>Input</td><td>Output</td><td>Highlight</td><td>Link</td></tr><tr><td>RWKV-UI 221]</td><td>Web</td><td>1.6B</td><td>SIGLIP [222], DINOv2 223], SAM [182]</td><td>Websight WebUI-7kbal 225 Web2Code 226</td><td>High-resolution webpage images</td><td>Element grounding, Action prediction, CoT reasoning</td><td>Introduces a high-resolution three-encoder architecture with visual prompt engineering and CoT reasoning</td><td>/</td></tr><tr><td>TRISHUL 227]</td><td>Web, Desktop, and Mobile platforms</td><td>/ (Training-Free)</td><td>/ (Training-Free)</td><td>/ (Training-Free)</td><td>GUI Screen-shots, user instruction-s/queries. hierarchical screen parsing outputs, OCR-extracted text descriptors</td><td>Action grounding, functionality descriptions of GUI elements. GUI referring, and SoMs</td><td>Utilizes hierarchical screen parsing and spatially enhanced element descriptions to enhance LVLMs without additional training</td><td>/</td></tr><tr><td>AutoGUI 228]</td><td>Web, Mobile</td><td>Qwen-VL- 10B 210 SliME-8B 229]</td><td>10B / 8B</td><td>AutoGUI-704k</td><td>GUI screen-shots, User queries</td><td>Element functionalities, Element locations</td><td>Automatically labels UI elements based on interaction-induced changes, making it scalable and high-quality.</td><td>https:// autogui-project. github.io/</td></tr><tr><td>Query Inference [230]</td><td>Mobile Android</td><td>Qwen2-VL- 7B-Instruct 231]</td><td>7B</td><td>UIBERT 232]</td><td>GUI shots</td><td>Action-oriented queries, Coordinates</td><td>Improves reasoning without requiring large-scale training data</td><td>https://github.com/ZrW00/ GUIPivot</td></tr><tr><td>WinClick 233]</td><td>Windows OS</td><td>Phi3-Vision 234]</td><td>4.2B</td><td>WinSpot Benchmark</td><td>GUI screen-shots, Natural language instructions</td><td>Element locations</td><td>The first GUI ground ing model specifically tailored for Windows.</td><td>https://github.com/zackhuiliii/ WinSpot</td></tr><tr><td>FOCUS 235]</td><td>Web, mobile applications, and desktop</td><td>Qwen2-VL- 2B-Instruct 231]</td><td>2B</td><td>GUICourse Aguvis-stage1 236] Wave-UI Desktop-UI 238]</td><td>GUI screenshot + task instruction</td><td>Normalized coordinates (x, y)</td><td>A dual-system GUI grounding architecture inspired by human cognition, whic dynamically switches between fast (intuitive) and slow (analytical) grounding modes based on task complexity</td><td>https: //github.com/ sugarandgugu/ Focus</td></tr><tr><td>UI-E2I-Synth 239]</td><td>Web. Windows, and Android</td><td>InternVL2- 4B and Qwen2-VL- 7B</td><td>4B and 7B</td><td>1.6M screenshots, 9.9M instructions</td><td>GUI screenshot</td><td>Element coordinates</td><td>Introduces a three-stage synthetic data pipeline for GUI grounding with both explicit and implicit instruction synthesis</td><td>https: <br> //colmon46. <br> github.io/ <br> i2e-bench-leaderboar</td></tr><tr><td>RegionFocus 240]</td><td>Web-based and Desktop interfaces</td><td>UI-TARS and Qwen2.5- VL</td><td>72B</td><td>None (test-time only)</td><td>GUI screen-shots with a point of interest</td><td>Coordinate-based actions</td><td>Introduces a visua test-time scaling framework that zooms into salient UI regions and integrates an image as-map mechanism to track history and avoid repeated mistakes—boosting grounding accuracy without model retraining</td><td>https://github.com/tiangeluo/ RegionFocus</td></tr></table>

TABLE 9: A summary of of GUI grounding benchmarks.

<table><tr><td>Benchmark</td><td>Platform</td><td>Dataset</td><td>Input</td><td>Output</td><td>Highlight</td><td>Link</td></tr><tr><td>ScreenSpot 25]</td><td>iOS, Android, macOS, and Windows</td><td>Over 600 screenshots and 1,200 instructions</td><td>GUI screenshots accompanied by user instructions</td><td>Bounding boxes or coordinates of actionable GUI elements</td><td>A realistic and diverse GUI grounding benchmark covering multiple platforms and a variety of elements</td><td>https://github.com/njucckevin/ SeeClick</td></tr><tr><td>ScreenSpot-Pro 241]</td><td>Windows, macOS, Linux</td><td>1,581 instruction-screenshot pairs covering 23 applications across 5 industries and 3 operating systems</td><td>High-resolution GUI screen-shots paired with natural language instructions</td><td>Bounding boxes for locating target UI elements</td><td>Introduces a high-resolution benchmark for professional environments</td><td>https: <br> //github.com/ <br> likaixin2000/ <br> ScreenSpot-Pro-GUI-Grounding</td></tr><tr><td>PixelWeb 242]</td><td>Web</td><td>100,000 webpages</td><td>Rendered webpage screenshots and information</td><td>BBox, mask, contour</td><td>The first GUI dataset to provide pixel-level annotations—including mask and contour—for web UIs, enabling high-precision GUI grounding and detection tasks</td><td>https: //huggingface. co/datasets/ cyberalchemist/ PixelWeb</td></tr></table>

<!-- Media -->


<!-- Meanless: JOURNAL OF LATEX CLASS FILES, DECEMBER 2024<br>25 -->

Task: Create a desk for LLM-based multi-agent system.

<!-- Media -->

<!-- figureText: Local Documents<br>Document Extractor<br>Created Desk<br>Designer<br>Creation<br>Recap: What is LLM-based Agent?<br>Communication<br>Feedback<br>Web Retriever<br>Webpages<br>Evaluator -->

<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2025/11/2025_11_04__12_39_04_a1a6cd.jpg"/>

<!-- Media -->

Fig. 15: An example of multi-agent system collaboration in creating a desk.

<!-- Media -->

<!-- figureText: Task: Make Line Drawing effect to the figure in the page.<br>Step 1: Click("Design")<br>Step 2: Click("Picture Format")<br>Step 3: Click( "Artistic Effects" )<br>Step 4: Click( "Line Drawing" )<br>Particle<br>PPPP<br>PPPAN<br>PPPP<br>PROPPA<br>PPER<br>APH<br>GUI Agent<br>I accessed the "Design" menu but couldn't find an option to create a line drawing effect on the screen. Perhaps there's another approach I should consider...<br>Self-reflection<br>Ah... After some reflection and experimentation, it seems I need to first select the figure, then navigate to the "Picture Format" tab. There, I can find the "Artistic Effects" menu, where the "Line Drawing" option is hidden. Task complete! -->

<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2025/11/2025_11_04__12_39_04_a51903.jpg"/>

Fig. 16: An example of self-reflection in task completion of an LLM-powered GUI agent.

<!-- Media -->

Self-reflection is particularly critical for GUI agents due to the variable nature of user interfaces and the potential for errors, even in human-operated systems. GUI agents frequently encounter situations that deviate from expectations, such as clicking the wrong button, encountering unexpected advertisements, navigating unfamiliar interfaces, receiving error messages from API calls, or even responding to user feedback on task outcomes. To ensure task success, a GUI agent must quickly reflect on its actions, assess these feedback signals, and adjust its plans to better align with the desired objectives.

As illustrated in Figure 16 when the agent initially fails to locate the "Line Drawing" option in the Design menu, self-reflection enables it to reconsider and identify its correct location under Artistic Effects" in the "Picture Format" menu, thereby successfully completing the task.

In practice, self-reflection techniques for GUI agents typically involve two main approaches: (i) ReAct [252] and (ii) Reflexion [253].

1) ReAct (Reasoning and Acting): ReAct integrates self-reflection into the agent's action chain by having the agent evaluate each action's outcome and reason about the next best step. In this framework, the agent doesn't simply follow a linear sequence of actions; instead, it adapts dynamically, continuously reassessing its strategy in response to feedback from each action. For example, if a GUI agent attempting to fill a form realizes it has clicked the wrong field, it can adjust by backtracking and selecting the correct element. Through ReAct, the agent achieves higher consistency and accuracy, as it learns to refine its behavior with each completed step.

2) Reflexion: Reflexion emphasizes language-based feedback, where agents receive and process feedback from the environment as linguistic input, referred to as self-reflective feedback. This feedback is contextualized and used as input in subsequent interactions, helping the agent to learn rapidly from prior mistakes. For instance, if a GUI agent receives an error message from an application, Reflexion enables the agent to process this message, update its understanding of the interface, and avoid similar mistakes in future interactions. Reflexion's iterative feedback loop promotes continuous improvement and is particularly valuable for GUI agents navigating complex, multi-step tasks. Overall, self-reflection serves as an essential enhancement in GUI multi-agent systems, enabling agents to better navigate the variability and unpredictability of GUI environments. This introspective capability not only boosts individual agent performance but also promotes resilience, adaptability, and long-term learning in a collaborative setting.


<!-- Meanless: JOURNAL OF LATEX CLASS FILES, DECEMBER 2024<br>26 -->

#### 5.7.4 Self-Evolution

Self-evolution [254] is a crucial attribute that GUI agents should possess, enabling them to enhance their performance progressively through accumulated experience. In the context of GUI multi-agent systems, self-evolution allows not only individual agents to improve but also facilitates collective learning and adaptation by sharing knowledge and strategies among agents. During task execution, GUI agents generate detailed action trajectories accompanied by complementary information such as environment states, internal reasoning processes (the agent's thought processes), and evaluation results. This rich data serves as a valuable knowledge base from which GUI agents can learn and evolve. The knowledge extracted from this experience can be categorized into three main areas:

1) Task Trajectories: The sequences of actions executed by agents, along with the corresponding environment states, are instrumental for learning [255]. These successful trajectories can be leveraged in two significant ways. First, they can be used to fine-tune the core LLMs that underpin the agents. Fine-tuning with such domain-specific and task-relevant data enhances the model's ability to generalize and improves performance on similar tasks in the future. Second, these trajectories can be utilized as demonstration examples to activate the in-context learning capabilities of LLMs during prompt engineering. By including examples of successful task executions in the prompts, agents can better understand and replicate the desired behaviors without additional model training.

For instance, suppose an agent successfully completes a complex task that involves automating data entry across multiple applications. The recorded action trajectory-comprising the steps taken, decisions made, and contextual cues—can be shared with other agents. These agents can then use this trajectory as a guide when faced with similar tasks, reducing the learning curve and improving efficiency.

2) Guidance and Rules: From the accumulated experiences, agents can extract high-level rules or guidelines that encapsulate best practices, successful strategies, and lessons learned from past mistakes [256], [257]. Such guidance can be acquired by the LLM itself through trajectory summarization [256], or even via search-based algorithms, such as Monte Carlo Tree Search (MCTS) [257]. This knowledge can be formalized into policies or heuristics that agents consult during decision-making processes, thereby enhancing their reasoning capabilities.

For example, if agents repeatedly encounter errors when attempting to perform certain actions without proper prerequisites (e.g., trying to save a file before specifying a file path), they can formulate a rule to check for these prerequisites before executing the action. This proactive approach reduces the likelihood of errors and improves task success rates.

3) New Toolkits: Throughout their interactions, GUI agents may discover or develop more efficient methods, tools, or sequences of actions that streamline task execution 161]. These may include optimized API calls, macros, or combinations of UI operations that accomplish tasks more effectively than previous approaches. LLMs can be leveraged to automatically analyze execution trajectories in order to summarize, discover, and generate high-level shortcuts or frequently used fast APIs, which can then be reused for future executions [258]. By incorporating these new tools into their repertoire, agents expand their capabilities and enhance overall efficiency.

As an example, an agent might find that using a batch processing API can automate repetitive tasks more efficiently than performing individual UI operations in a loop. This new approach can be shared among agents within the multi-agent system, allowing all agents to benefit from the improved method and apply it to relevant tasks.

Figure 17 illustrates how a GUI agent evolves through task completion. During its operations, the agent adds new capabilities to its skill set, such as an image summarization toolkit, gains insights from reading a paper on creating GUI agents, and stores task trajectories like webpage extraction in its experience pool. When assigned a new task, such as "Learn to make a GUI agent from a GitHub repository", the agent draws on its acquired skills and past experiences to adapt and perform effectively.

This dynamic evolution highlights the agent's ability to continually learn, grow, and refine its capabilities. By leveraging past experiences, incorporating new knowledge, and expanding its toolset, GUI agents can adapt to diverse challenges, improve task execution, and significantly enhance the overall performance of the system, fostering a collaborative and ever-improving environment.

#### 5.7.5 Reinforcement Learning

Reinforcement Learning (RL) [259] has witnessed significant advancements in aligning LLMs with desired behaviors [260], and has recently been employed in the development of LLM agents [50], [261]. In the context of GUI multi-agent systems, RL offers substantial potential to enhance the performance, adaptability, and collaboration of GUI agents. GUI automation tasks naturally align with the structure of a Markov Decision Process (MDP) [262], making them particularly well-suited for solutions based on RL. In this context, the state corresponds to the environment perception (such as GUI screenshots, UI element properties, and layout configurations), while actions map directly to UI operations, including mouse clicks, keyboard inputs, and API calls. Rewards can be explicitly defined based on various performance metrics, such as task completion, efficiency, and accuracy, allowing the agent to optimize its actions for maximal effectiveness. Figure 18 illustrates an example of MDP modeling for task completion in a GUI agent, where state, action and reward are clearly defined.

By formulating GUI agent interactions as an MDP, we can leverage RL techniques to train agents that learn optimal policies for task execution through trial and error [263]. This approach enables agents to make decisions that maximize cumulative rewards over time, leading to more efficient and effective task completion. For example, an agent learning to automate form filling in a web application can use RL to discover the most efficient sequence of actions to input data and submit the form successfully, minimizing errors and redundant steps. This process helps align the agents more closely with desired behaviors in GUI automation tasks, especially in complex or ambiguous situations where predefined action sequences are insufficient.


<!-- Meanless: JOURNAL OF LATEX CLASS FILES, DECEMBER 2024<br>27 -->

<!-- Media -->

<!-- figureText: Task 1: Summarize the image content<br>New Toolkits: Image summarization<br>New Task: Learn to make a GUI agent from a GitHub repository<br>Skill Set<br>Task 2: Read and summarize the paper<br>Tool 1<br>New Tool<br>Tool 2<br>Guidance and Rules: How to make a GUI agent?<br>Evolve<br>Completion<br>-<br>Task 3: Read and extract information from a webpage<br>Experience Pool<br>New Trajectory<br>New Guidance<br>Task Trajectory: Webpages extraction -->

<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2025/11/2025_11_04__12_39_04_313629.jpg"/>

Fig. 17: An example self-evolution in a LLM-powered GUI agent with task completion.

<!-- figureText: Task: Make Line Drawing effect to the figure in the page.<br>Action 1: Click("Picture Format")<br>Action 2: Click("Artistic Effects")<br>Action 3: Click("Line Drawing")<br>SPARE<br>Particle<br>appear<br>PROPED<br>PROPED<br>中国<br>PAN<br>State 1<br>State 2<br>State 3<br>Evaluator<br>GUI Agent<br>-<br>2<br>Reward<br>Action 1': Click("Design")<br>Action 2': Click("Designer")<br>Action 3': Click("Format 3")<br><icon/><br>State 1'<br>State 2'<br>State 3'<br>2<br>J<br><icon/> -->

<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2025/11/2025_11_04__12_39_04_2b6dc8.jpg"/>

Fig. 18: An example of MDP modeling for task completion in a GUI agent.

<!-- Media -->

As a representative approach, Bai et al., introduce DigiRL 264 , a two-phase RL framework for training GUI agents in dynamic environments. DigiRL begins with an offline RL phase that uses offline data to initialize the agent model, followed by online fine-tuning, where the model interacts directly with an environment to refine its strategies through live data within an Android learning environment using an LLM evaluator that provides reliable reward signals. This adaptive setting enables the agent to learn and respond effectively to the complexities of dynamic GUIs. Wang et al., propose DistRL 265], an RL fine-tuning pipeline specifically designed for on-device mobile control agents operating within Android. DistRL employs an asynchronous architecture, deploying RL fine-tuned agents across heterogeneous worker devices and environments for decentralized data collection. By leveraging off-policy RL techniques, DistRL enables centralized training with data gathered remotely from diverse environments, significantly enhancing the scalability and robustness of the model. These representative methods illustrate the potential of RL to improve GUI agents, demonstrating how both centralized and distributed RL frameworks can enable more responsive, adaptable, and effective GUI automation models in real-world applications.


<!-- Meanless: JOURNAL OF IATEX CLASS FILES, DECEMBER 2024<br>28 -->

#### 5.7.6 Summary & Takeaways

In conclusion, the advanced techniques significantly enhance the capabilities of LLM-brained GUI agents, making them more versatile, efficient, and adaptive within multi-agent frameworks. Importantly, these techniques are not mutually exclusive-many can be integrated to create more powerful agents. For instance, incorporating self-reflection within a multi-agent framework allows agents to collaboratively improve task strategies and recover from errors. By leveraging these advancements, developers can design LLM-brained GUI agents that are not only adept at automating complex, multi-step tasks but also capable of continuously improving through self-evolution, adaptability to dynamic environments, and effective inter-agent collaboration. Future research is expected to yield even more sophisticated techniques, further extending the scope and robustness of GUI automation.

### 5.8 From Foundations to Innovations: A Roadmap

Building robust, adaptable, and effective LLM-powered GUI agents is a multifaceted process that requires careful integration of several core components. With a solid foundation in architecture, design, environment interaction, and memory, as outlined in Section 5 we now shift our focus to the critical elements required for deploying these agents in practical scenarios. This exploration begins with an in-depth review of state-of-the-art LLM-brained GUI agent frameworks in Section 6 highlighting their advancements and unique contributions to the field. Building on this, we delve into the methodologies for optimizing LLMs for GUI agents, starting with data collection and processing strategies in Section 7 and progressing to model optimization techniques in Section 8 To ensure robust development and validation, we then examine evaluation methodologies and benchmarks in Section 9, which are essential for assessing agent performance and reliability. Finally, we explore a diverse range of practical applications in Section 10 demonstrating the transformative impact of these agents across various domains.

Together, these sections provide a comprehensive roadmap for advancing LLM-brained GUI agents from foundational concepts to real-world implementation and innovation. This roadmap, spanning from foundational components to real-world deployment, encapsulates the essential pipeline required to bring an LLM-powered GUI agent concept from ideation to implementation.

To provide a comprehensive view, we first introduce a taxonomy in Figure 19 which categorizes recent work on LLM-brained GUI agents across frameworks, data, models, evaluation, and applications. This taxonomy serves as a blueprint for navigating the extensive research and development efforts within each field, while acknowledging overlaps among categories where certain models, frameworks, or datasets contribute to multiple aspects of GUI agent functionality.

## 6 LLM-BRAINED GUI AGENT FRAMEWORK

The integration of LLMs has unlocked new possibilities for constructing GUI agents, enabling them to interpret user requests, analyze GUI components, and autonomously perform actions across diverse environments. By equipping these models with essential components and functionalities, as outlined in Section 5 researchers have created sophisticated frameworks tailored to various platforms and applications. These frameworks represent a rapidly evolving area of research, with each introducing innovative techniques and specialized capabilities that push the boundaries of what GUI agents can achieve.

We offer a detailed discussion of each framework, examining their foundational design principles, technical advancements, and the specific challenges they address in the realm of GUI automation. By delving into these aspects, we aim to provide deeper insights into how these agents are shaping the future of human-computer interaction and task automation, and the critical role they play in advancing this transformative field.

### 6.1 Web GUI Agents

Advancements in web GUI agents have led to significant strides in automating complex tasks within diverse and dynamic web environments. Recent frameworks have introduced innovative approaches that leverage multimodal inputs, predictive modeling, and task-specific optimizations to enhance performance, adaptability, and efficiency. In this subsection, we first summarize key web GUI agent frameworks in Tables 10 11 and 12 then delve into representative frameworks, highlighting their unique contributions and how they collectively push the boundaries of web-based GUI automation.

One prominent trend is the integration of multimodal capabilities to improve interaction with dynamic web content. For instance, SeeAct [17] harnesses GPT-4V's multimodal capacities to ground actions on live websites effectively. By leveraging both visual data and HTML structure, SeeAct integrates grounding techniques using image annotations, HTML attributes, and textual choices, optimizing interactions with real-time web content. This approach allows SeeAct to achieve a task success rate of 51.1% on real-time web tasks, highlighting the importance of dynamic evaluation in developing robust web agents.

Building upon the advantages of multimodal inputs, We-bVoyager 269 advances autonomous web navigation by supporting end-to-end task completion across real-world web environments. Utilizing GPT-4V for both visual (screenshots) and textual (HTML elements) inputs, WebVoyager effectively interacts with dynamic web interfaces, including those with dynamically rendered content and intricate interactive elements. This multimodal capability allows WebVoyager to manage complex interfaces with a success rate notably surpassing traditional text-only methods, setting a new benchmark in web-based task automation.

In addition to multimodal integration, some frameworks focus on parsing intricate web structures and generating executable code to navigate complex websites. WebAgent 267 employs a two-tiered model approach by combining HTML-T5 for parsing long, complex HTML documents with Flan-U-PaLM [498] for program synthesis. This modular design enables WebAgent to translate user instructions into executable Python code, autonomously handling complex, real-world websites through task-specific sub-instructions. WebAgent demonstrates a 50% improvement in success rates on real websites compared to traditional single-agent models, showcasing the advantages of integrating HTML-specific parsing with code generation for diverse and dynamic web environments.


<!-- Meanless: JOURNAL OF LATEX CLASS FILES, DECEMBER 2024<br>29 -->

<!-- Media -->

<!-- figureText: [335]<br>Computer<br>Frameworks<br>王王<br>고의하며<br>Raby<br>Foundations to Innovations<br>ORINE<br>standa<br>Web<br>Cross-Platform<br>[25]<br>[405]<br>1184<br>1372<br>1375<br>232<br>238<br>406<br>384<br>407 -->

<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2025/11/2025_11_04__12_39_04_33028a.jpg"/>

Fig. 19: A Taxonomy of frameworks, data, models, evaluations, and applications: from foundations to innovations in LLM-brained GUI agents.

<!-- Media -->

To enhance decision-making in web navigation, several frameworks introduce state-space exploration and search algorithms. LASER 268 models web navigation as state-space exploration, allowing flexible backtracking and efficient decision-making without requiring extensive in-context examples. By associating actions with specific states and leveraging GPT-4's function-calling feature for state-based action selection, LASER minimizes errors and improves task success, particularly in e-commerce navigation tasks such as WebShop and Amazon. This state-based approach provides a scalable and efficient solution, advancing the efficiency of LLM agents in GUI navigation.

Similarly, Search-Agent 274 innovatively introduces a best-first search algorithm to enhance multi-step reasoning in interactive web environments. By exploring multiple action paths, this approach improves decision-making, achieving up to a 39% increase in success rates across benchmarks like WebArena [412]. Search-Agent's compatibility with existing multimodal LLMs demonstrates the effectiveness of search-based algorithms for complex, interactive web tasks.


<!-- Meanless: JOURNAL OF LATEX CLASS FILES, DECEMBER 2024<br>30 -->

<!-- Media -->

<!-- figureText: (a) Goal-Oriented Selection<br>(b) Reflection-Enhanced Node Expansion<br>(c) Dynamic Evaluation and Simulation<br>(d) Maximal Value Backpropagation<br>Start<br>Start<br>Start<br>Start<br>Google<br>Explorer<br>Explorer<br>Explorer<br>Search Result Q=6<br>Search Result Q=6<br>Search Result Q=6<br>Search Result Q=6 → 9.5<br>Explorer<br>Explorer<br>Explorer<br>Netflix Article Q=3<br>Netflix Article Q=3<br>YouTube Q=9.6<br>Netflix Article Q=3<br>YouTube Q=9.6<br>Netflix Article Q=3<br>YouTube Q=9.6<br>Explorer<br>Video Play<br>Appraiser<br>Simulation & Evaluation -->

<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2025/11/2025_11_04__12_39_04_fb6de5.jpg"/>

Fig. 20: An illustration of the local optimization stage in WebPilot [275] using MCTS. Figure adapted from the original paper.

<!-- figureText: Task Completed<br>Step 1<br>Step 2<br>Task: Play the video of Arcane Season 1<br>LLM<br>NETFLIX<br>v = 0.4<br>LLM<br>Execution<br>LLM<br>Plan:<br>(1) Click "YouTube Arcane Season<br>1 | Recap | Netflix"<br>(2) Click the "Play" Button -->

<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2025/11/2025_11_04__12_39_04_b61dbb.jpg"/>

Fig. 21: An example illustrating how WebDreamer [282] uses an LLM to simulate the outcome of each action. Figure adapted from the original paper.

<!-- Media -->

Expanding on search-based strategies, WebPilot [275] employs a dual optimization strategy combining global and local Monte Carlo Tree Search (MCTS) [500] to improve adaptability in complex and dynamic environments. As illustrated in Figure 20 WebPilot decomposes overarching tasks into manageable sub-tasks, with each undergoing localized optimization. This approach allows WebPilot to continuously adjust its strategies in response to real-time observations, mimicking human-like decision-making and flexibility. Extensive testing on benchmarks like WebArena [412] and MiniWoB++ [146] demonstrates WebPilot's state-of-the-art performance, showcasing exceptional adaptability compared to existing methods.

Furthering the concept of predictive modeling, the WMA [266] introduces a world model to simulate and predict the outcomes of UI interactions. By focusing on transition-based observations, WMA allows agents to simulate action results before committing, reducing unnecessary actions and increasing task efficiency. This predictive capability is particularly effective in long-horizon tasks that require high accuracy, with WMA demonstrating strong performance on benchmarks such as WebArena [412] and Mind2Web [212].

Along similar lines, WebDreamer [282] introduces an innovative use of LLMs for model-based planning in web navigation, as depicted in Figure 21. WebDreamer simulates and evaluates potential actions and their multi-step outcomes using LLMs before execution [506], akin to a "dreamer" that envisions various scenarios. By preemptively assessing the potential value of different plans, WebDreamer selects and executes the plan with the highest expected value. This approach addresses critical challenges in web automation, such as safety concerns and the need for robust decision-making in complex and dynamic environments, demonstrating superiority over reactive agents in benchmarks like VisualWe-bArena [413] and Mind2Web-live [349].


<!-- Meanless: 31 -->

<!-- Media -->

JOURNAL OF IATEX CLASS FILES, DECEMBER 2024

TABLE 10: Overview of LLM-brained GUI agent frameworks on web platforms (Part I).

<table><tr><td>Agent</td><td>Platform</td><td>Perception</td><td>Action</td><td>Model</td><td>Architecture</td><td>Highlight</td><td>Link</td></tr><tr><td>WMA 266</td><td>Web</td><td>Accessibility tree from DOM</td><td>UI operations, e.g., clock, type, and hover</td><td>Llama-3.1-8B-Instruct [91] for predicting observations and GPT-4 for policy modeling</td><td>Single-agent with simulation-based observation</td><td>Uses a world model to predict state changes before committing actions, improving task success rates and minimizing unnecessary interactions with the environment</td><td>https://github.com/kyle8581/ WMA-Agents</td></tr><tr><td>WebAgent 267]</td><td>Web</td><td>HTML structure</td><td>UI interactions</td><td>HTML-T5 for planning and summarization and Flan-U-PaLM 498 for code generation</td><td>Two-stage architecture for planning and program synthesis</td><td>Leverages specialized LLMs to achieve HTML-based task planning and programmatic action execution</td><td>/</td></tr><tr><td>LASER 268]</td><td>Web</td><td>GUI structure of the web environment, with defined states</td><td>Defined per state. such as searching, selecting items. navigating pages. and finalizing a purchase</td><td>GPT-4</td><td>Single-agent</td><td>Uses a state-space exploration approach, allowing it to handle novel situations with flexible backtracking</td><td>https://github com/Mayer123/ LASER</td></tr><tr><td>WebVoyager 269]</td><td>Web</td><td>Screenshots with numerical labels interactive elements</td><td>Standard UI operations</td><td>GPT-4V</td><td>Single-agent</td><td>Integrates visual and textual cues within real-world, rendered web pages, enhancing its ability to navigate com plex web structures</td><td>https: //github.com/ MinorJerry/ WebVoyager</td></tr><tr><td>AutoWeb-GLM 270]</td><td>Web</td><td>Simplified HTML OCR for text recognition</td><td>UI operations such as clicking, typing, scrolling, and selecting, and advanced APIs like jumping to specific URLs</td><td>ChatGLM3-6B 499</td><td>Single-agent</td><td>Its HTML simplification method for efficient webpage comprehension and its bilingual benchmark</td><td>https://github.com/THUDM/ AutoWebGLM</td></tr><tr><td>OpenAgents 271]</td><td>Web</td><td>DOM elements</td><td>Standard operations. browser-based actions controlled. API calls for tool execution, and structured data manipulation</td><td>GPT-4 and Claude [163]</td><td>Multi-agent architecture, with distinct agents (Data Agent, Plugins Agent, and Web Agent)</td><td>Democratizes access to lan guage agents by providing an open-source, multi-agent framework optimized for real-world tasks</td><td>https://github.<br> com/xlang-al <br> OpenAgents</td></tr><tr><td>SeeAct [17]</td><td>Web</td><td>Screenshot images and HTML structure</td><td>Standard UI operations</td><td>GPT-4V</td><td>Single-agent</td><td>Its use of GPT-4V's multimodal capabilities to integrate both visual anc HTML information, allowing for more accurate task per formance on dynamic web content</td><td>https: //github.com/ OSU-NLP-Group SeeAct</td></tr><tr><td>DUAL-VCR 272]</td><td>Web</td><td>HTML elements and screenshots</td><td>Standard UI operations</td><td>Flan-T5-base 498</td><td>Two-stage single-agent architecture</td><td>Dual-view contextualization</td><td>/</td></tr><tr><td>Agent-E 273]</td><td>Web</td><td>DOM structure and change observation</td><td>Standard UI operations</td><td>GPT-4 Turbo</td><td>Hierarchical multi-agent architecture, composed of a planner agent and a browser navigation agent</td><td>Hierarchical architecture and adaptive DOM perception</td><td>https: //github.com/ EmergenceAI/ Agent-E</td></tr><tr><td>Search-Agent 274]</td><td>Web</td><td>Screenshot and text descriptions</td><td>Standard UI operations</td><td>GPT-4</td><td>Single-agent with search</td><td>Novel inference-time search algorithm that enhances the agent's ability to per form multi-step planning and decision-making</td><td>https: //jykoh.com/ search-agents</td></tr><tr><td>R2D2 288]</td><td>Web</td><td>DOM</td><td>Standard UI operations</td><td>GPT-40</td><td>Single-agent</td><td>Dynamically constructs an internal web environment representation for more robust decision-making. The integration of a replay buffer and error analysis reduces navigation errors and improves task completion rates.</td><td>https: //github.com/ AmenRa/retriv</td></tr></table>

<!-- Media -->

Beyond predictive modeling, integrating API interactions into web navigation offers enhanced flexibility and efficiency. The Hybrid Agent [199] combines web browsing and API interactions, dynamically switching between methods based on task requirements. By utilizing API calls for structured data interaction, the Hybrid Agent reduces the time and complexity involved in traditional web navigation, achieving higher accuracy and efficiency in task performance. This hybrid architecture underscores the benefits of integrating both structured API data and human-like browsing capabilities in AI agent systems.


<!-- Meanless: 32 -->

<!-- Media -->

JOURNAL OF LATEX CLASS FILES, DECEMBER 2024

TABLE 11: Overview of LLM-brained GUI agent frameworks on web platforms (Part II).

<table><tr><td>Agent</td><td>Platform</td><td>Perception</td><td>Action</td><td>Model</td><td>Architecture</td><td>Highlight</td><td>Link</td></tr><tr><td>ScribeAgent 285]</td><td>Web</td><td>HTML-DOM</td><td>Standard UI operations</td><td>Single-agent architecture</td><td>Specialized fine-tuning approach using production-scale workflow data to outperform general-purpose LLMs like GPT-4 in web navigation tasks</td><td>https://github.com/ <br> colonylabs/ScribeAgent</td><td></td></tr><tr><td>286]</td><td>Web</td><td>Screenshots</td><td>Standard UI Operations</td><td>Claude Sonnet 1631, Qwen2VL-7B <br> 231], and <br> LLaVa-1.6 [217]</td><td>A multi-agent architecture involving a task proposer to suggest tasks, an agent policy to perform tasks, and an autonomous evaluator to assess success and provide feedback using RL.</td><td>Autonomous skill discovery in real-world environments using task proposers and reward-based evaluation</td><td>https://yanqval.github.io/PAE/</td></tr><tr><td>WebPilot 275]</td><td>Web</td><td>Accessibility trees (actrees) and dynamic observations</td><td>Standard UI operations</td><td>GPT-4</td><td>Multi-agent architecture, with Global Optimization and Local Optimization</td><td>Dual optimization strategy (Global and Local) with Monte Carlo Tree Search (MCTS) [500], allowing dynamic adaptation to complex, real-world web environments</td><td>https://yaoz720.github.io/ WebPilot/</td></tr><tr><td>Hybrid Agent 199]</td><td>Web</td><td>Accessibility trees and screenshots</td><td>Standard UI operations, API calls, and generating code</td><td>GPT-4</td><td>Multi-agent system, combining both API and browsing capabilities</td><td>Hybrid Agent seamlessly integrates web browsing and API calls</td><td>https://github.com/yuegis/ API-Based-Agent</td></tr><tr><td>AgentOccam 276]</td><td>Web</td><td>HTML</td><td>Standard UI operations</td><td>GPT-4</td><td>Single-agent</td><td>Simple design that optimizes the observation and action spaces</td><td>/</td></tr><tr><td>NNetnav 277</td><td>Web</td><td>DOM</td><td>Standard UI operations</td><td>GPT-4</td><td>Single-agent</td><td>Trains web agents using synthetic demonstrations, eliminating the need for expensive human input</td><td>https: //github.com/ MurtyShikhar/ Nnetnav</td></tr><tr><td>NaviQAte 2781</td><td>Web</td><td>Screenshots</td><td>Standard UI operations</td><td>GPT-4</td><td>Single-agent system</td><td>Frames web navigation as a question-and-answer task</td><td>7</td></tr><tr><td>OpenWeb-Agent 279</td><td>Web</td><td>HTML and screenshots</td><td>UI operations. Web APIs, and self-generated code</td><td>GPT-4 and AutoWebGLM 270</td><td>Modular single-agent</td><td>Modular design that allows developers to seamlessly integrate various models to automate web tasks</td><td>https://github.com/THUDM/ OpenWebAgent/</td></tr><tr><td>Steward 280</td><td>Web</td><td>HTML and screenshots</td><td>Standard UI operations</td><td>GPT-4</td><td>Single-agent</td><td>Ability to automate web interactions using natural language instructions</td><td>/</td></tr><tr><td>WebDreamer 282]</td><td>Web</td><td>Screenshots combined with SoM. and HTML</td><td>Standard UI operations and navigation actions</td><td>GPT-40</td><td>Model-based single-agent architecture</td><td>Pioneers the use of LLMs as world models for planning in complex web environments</td><td>https: //github.com/ OSU-NLP-Group/ WebDreamer</td></tr><tr><td>Agent Q 281]</td><td>Web</td><td>DOM for textual input. screenshots for visual feedback</td><td>UI interactions. querying the user for help</td><td>LLaMA-3 70B 91 for policy learning and execution. GPT-V for visual feedback</td><td>Single-agent with MCTS and RL</td><td>Combination of MCTS-guided search and self-critique mechanisms enables improvement in reasoning and task execution</td><td>https: //github.com/ sentient-engineerir agent-q</td></tr></table>

<!-- Media -->

Addressing the challenges of complex web structures and cross-domain interactions, AutoWebGLM [270] offers an efficient solution by simplifying HTML to focus on key webpage components, thereby improving task accuracy. Using reinforcement learning and rejection sampling for fine-tuning, AutoWebGLM excels in complex navigation tasks on both English and Chinese sites. Its bilingual dataset and structured action-perception modules make it practical for cross-domain web interactions, emphasizing the importance of efficient handling in diverse web tasks.

ECLAIR [291] represents a pioneering application that replaces traditional RPA with a foundation model-powered GUI agent for enterprise automation. Unlike conventional RPA, which relies on manually programmed rules and rigid scripts, ECLAIR dynamically learns workflows from video demonstrations and textual SOPs (Standard Operating Procedures), significantly reducing setup time and improving adaptability. It operates on enterprise web applications, leveraging GPT- 4V and CogAgent [505] to perceive GUI elements, plan actions, and execute workflows, and validate automatically. By eliminating the high maintenance costs and execution brittleness of RPA, ECLAIR introduces a more flexible and scalable approach to GUI automation. We show a comparison of such agent-based vs. RPA automation in Figure 22. This work establishes an important foundation for LLM-powered GUI automation, demonstrating how multimodal foundation models can bridge the gap between process mining, RPA, and fully autonomous enterprise workflows.

In summary, recent frameworks for web GUI agents have made substantial progress by integrating multimodal inputs, predictive models, and advanced task-specific optimizations. These innovations enable robust solutions for real-world tasks, enhancing the capabilities of web-based GUI agents and marking significant steps forward in developing intelligent, adaptive web automation.


<!-- Meanless: 33 -->

### 6.2 Mobile GUI Agents

<!-- Media -->

JOURNAL OF LATEX CLASS FILES, DECEMBER 2024

TABLE 12: Overview of LLM-brained GUI agent frameworks on web platforms (Part III).

<table><tr><td>Agent</td><td>Platform</td><td>Perception</td><td>Action</td><td>Model</td><td>Architecture</td><td>Highlight</td><td>Link</td></tr><tr><td>Auto-Intent 284]</td><td>Web</td><td>HTML structure</td><td>Standard UI Operations</td><td>GPT-3.5, GPT-4, Llama-3 [91] for action inference; Mistral-7B [501] and Flan-T5XL 498] for intent prediction</td><td>Single-agent with self-exploration</td><td>Introduces a unique self-exploration strategy to generate semantically diverse intent hints</td><td>/</td></tr><tr><td>AdaptAgent 283]</td><td>Web</td><td>GUI screen-shots with HTML/DOM structures</td><td>Standard UI Operations and Playwright scripts</td><td>GPT-40 and Co-gAgent [15]</td><td>Single-agent</td><td>Adapts to unseen tasks with just 1-2 multimodal human demonstrations</td><td>/</td></tr><tr><td>WEPO 287]</td><td>Web</td><td>HTML and DOM</td><td>Standard UI Operations</td><td>Llama3-8B 91 Mistral-7B [501]. and Gemma-2B 502]</td><td>Single-agent architecture.</td><td>Incorporates a distance-based sampling mechanism tailored to the DOM tree structure, enhancing preference learning by distinguishing between salient and non-salient web elements with DPO 5031</td><td>/</td></tr><tr><td>AgentSymbioticWeb <br> 289]</td><td></td><td>Accessible tree structure of web elements</td><td>Standard UI operations</td><td>Large LLMs: GPT-40, Claude- 3.5. Small LLMs: LLaMA-3 [91]. DeepSeek-R1 504]</td><td>Multi-agent iterative architecture</td><td>Introduces an iterative, symbiotic learning process between large and small LLMs for web automation. Enhances both data synthesis and task performance through speculative data synthesis, multi-task learning, and privacy-preserving hybrid modes.</td><td>/</td></tr><tr><td>LiteWebAgent 292]</td><td>Web</td><td>DOM, Screen-shots</td><td>Standard UI operations, Playwright script</td><td>Any LLM and MLLM</td><td>Single-agent</td><td>First open-source. production-ready web agent integrating tree search for multi-step task execution.</td><td>https://github.com/PathOnAI/ LiteWebAgent</td></tr><tr><td>ECLAIR 291]</td><td>Web</td><td>Screenshots</td><td>Standard UI operations</td><td>GPT-4V,GPT- 40, CogAgent 505</td><td>Single-agent architecture</td><td>Eliminates the high setup costs, brittle execution, and burdensome maintenance associated with traditional RPA by learning from video and text documentation.</td><td>https: //github.com/ HazyResearch/ eclair-agents</td></tr><tr><td>Dammu et al., 293]</td><td>Web</td><td>DOM elements, Webpage accessibility attributes</td><td>Standard UI operations</td><td>Not specified</td><td>Single-agent architecture</td><td>User-aligned task execution where the agent adapts to individual user preferences in an ethical manner.</td><td>/</td></tr><tr><td>Plan-and-Act 294]</td><td>Web</td><td>HTML</td><td>Standard UI operations</td><td>LLaMA-3.3- 70B-Instruct 91]</td><td>Two-stage modular architecture: PLANNER + EXECUTOR</td><td>Decouples planning from execution in LLM-based GUI agents and introduces a scalable synthetic data generation pipeline to fine-tune each component</td><td>/</td></tr><tr><td>SkillWeaver 295]</td><td>Web</td><td>GUI screen-shots and Accessibility Tree</td><td>Standard UI operations and high-level skill APIs</td><td>GPT-40</td><td>Single-agent</td><td>Introduces a self-improvement framework for web agents that autonomously discover. synthesize. and refine reusable skill APIs through exploration</td><td>https: //github.com/ OSU-NLP-Group/ SkillWeaver</td></tr><tr><td>ASI 2961</td><td>Web</td><td>Webpage Accessibility Tree</td><td>Standard GUI actions</td><td>Claude-3.5- Sonnet</td><td>Single-agent</td><td>Introduces programmatic skills that are verified through execution to ensure quality and are used as callable actions to improve efficiency</td><td>https://github.com/zorazrw/ agent-skill-inductio</td></tr><tr><td>Rollback Agent 297]</td><td>Web</td><td>Accessibility trees</td><td>Standard GUI actions</td><td>Multi-agent ar chitecture</td><td>Multi-module. ReAct-inspired agent architecture</td><td>Introduces a modular rollback mechanism that enables multi-step rollback to avoid dead-end states</td><td>/</td></tr></table>

<!-- Media -->

The evolution of mobile GUI agents has been marked by significant advancements, leveraging multimodal models, complex architectures, and adaptive planning to address the unique challenges of mobile environments. These agents have progressed from basic interaction capabilities to sophisticated systems capable of dynamic, context-aware operations across diverse mobile applications. We first provide an overview of mobile GUI agent frameworks in Tables 13 14 and 15

Wang et al., 323 pioneer the use of LLMs to enable conversational interaction with mobile UIs, establishing one of the earliest foundations for mobile GUI agents. Their approach involves directly prompting foundation models such as PaLM using structured representations of Android view hierarchies, which are transformed into HTML-like text to better align with the LLM's training distribution. The authors define and evaluate four core tasks, including Screen Summarization, Screen QA, Screen Question Generation, and Instruction-to-UI Mapping-demonstrating that strong performance can be achieved with as few as two prompt examples per task. Emphasizing practicality and accessibility, the work enables rapid prototyping without model fine-tuning, and stands out as a seminal effort in prompt-based evaluation of LLM-powered GUI agents for mobile applications.


<!-- Meanless: JOURNAL OF LATEX CLASS FILES, DECEMBER 2024<br>34 -->

<!-- Media -->

<!-- figureText: 1. Workflow construction and demonstration<br>2. Execution<br>3. Evaluation<br>RPA<br>Hard<br>Deterministic<br>Manual<br>Human Design<br>Hardcode Workflow<br>Validation<br>Step 1<br>Step 1<br>Step n<br>OG<br>Execution logs<br>Human<br>Agent-based<br>Easy<br>Flexible<br>Automatic<br>Human Demonstration<br>Validation<br>LLM<br>Textual Documentation<br>Log<br>GUI Agent<br>Execution logs<br>Video Recording -->

<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2025/11/2025_11_04__12_39_04_c286b4.jpg"/>

Fig. 22: Comparison of RPA and agent based automation. Figure adapted from [291].

<!-- Media -->

Early efforts focused on enabling human-like GUI interactions without requiring backend system access. One such pioneering framework is AppAgent [18], which utilizes GPT- 4V's multimodal capabilities to comprehend and respond to both visual and textual information. By performing actions like tapping and swiping using real-time screenshots and structured XML data, AppAgent can interact directly with the GUI across a variety of applications, from social media to complex image editing. Its unique approach of learning through autonomous exploration and observing human demonstrations allows for rapid adaptability to new apps, highlighting the effectiveness of multimodal capabilities in mobile agents.

Building upon this foundation, AppAgent-V2 [304] advances the framework by enhancing visual recognition and incorporating structured data parsing. This enables precise, context-aware interactions and the ability to perform complex, multi-step operations across different applications. AppAgent-V2 also introduces safety checks to handle sensitive data and supports cross-app tasks by tracking and adapting to real-time interactions. This progression underscores the importance of advanced visual recognition and structured data processing in improving task precision and safety in real-time mobile environments.

Parallel to these developments, vision-centric approaches emerged to further enhance mobile task automation without relying on app-specific data. For instance, Mobile-Agent [158] leverages OCR, CLIP [509], and Grounding DINO [183] for visual perception. By using screenshots and visual tools, Mobile-Agent performs operations ranging from app navigation to complex multitasking, following instructions iteratively and adjusting for errors through a self-reflective mechanism. This vision-based method positions Mobile-Agent as a versatile and adaptable assistant for mobile tasks.

To address challenges in long-sequence navigation and complex, multi-app scenarios, Mobile-Agent-v2 [306] introduces a multi-agent architecture that separates planning, decision-making, and reflection. By distributing responsibilities among three agents, this framework optimizes task progress tracking, retains memory of task-relevant information, and performs corrective actions when errors occur. Integrated with advanced visual perception tools like Grounding DINO [183] and Qwen-VL-Int4 [210], Mobile-Agent-v2 showcases significant improvements in task completion rates on both Android and Harmony OS, highlighting the potential of multi-agent systems for handling complex mobile tasks.

In addition to vision-centric methods, some frameworks focus on translating GUI states into language to enable LLM-based action planning. VisionTasker 299] combines vision-based UI interpretation with sequential LLM task planning by processing mobile UI screenshots into structured natural language. Supported by YOLO-v8 [207] and PaddleOCP ${}^{28}$ for widget detection, VisionTasker allows the agent to automate complex tasks across unfamiliar apps, demonstrating higher accuracy than human operators on certain tasks. This two-stage design illustrates a versatile and adaptable framework, setting a strong precedent in mobile automation.

Similarly, DroidBot-GPT [300] showcases an innovative approach by converting GUI states into natural language prompts, enabling LLMs to autonomously decide on action sequences. By interpreting the GUI structure and translating it into language that GPT models can understand, DroidBot-GPT generalizes across various apps without requiring app-specific modifications. This adaptability underscores the transformative role of LLMs in handling complex, multi-step tasks with minimal custom data.

To enhance action prediction and context awareness, advanced frameworks integrate perception and action systems within a multimodal LLM. CoCo-Agent [301] exemplifies this by processing GUI elements like icons and layouts through its Comprehensive Event Perception and Comprehensive Action Planning modules. By decomposing actions into manageable steps and leveraging high-quality data from benchmarks like Android in the Wild (AITW) [358] and META-GUI [357], CoCo-Agent demonstrates its ability to automate mobile tasks reliably across varied smartphone applications.

Further advancing this integration, CoAT [298] introduces a chain-of-action-thought process to enhance action prediction and context awareness. Utilizing sophisticated models such as GPT-4V and set-of-mark tagging, CoAT addresses the limitations of traditional coordinate-based action recognition. By leveraging the Android-In-The-Zoo (AITZ) dataset it builds, CoAT provides deep context awareness and improves both action prediction accuracy and task completion rates, highlighting its potential for accessibility and user convenience on Android platforms.

---

<!-- Footnote -->

28. https://github.com/PaddlePaddle/PaddleOCR

<!-- Footnote -->

---


<!-- Meanless: 35 -->

<!-- Media -->

JOURNAL OF LATEX CLASS FILES, DECEMBER 2024

TABLE 13: Overview of LLM-brained GUI agent frameworks on mobile platforms (Part I).

<table><tr><td>Agent</td><td>Platform</td><td>Perception</td><td>Action</td><td>Model</td><td>Architecture</td><td>Highlight</td><td>Link</td></tr><tr><td>Wang et al., 323]</td><td>Android Mobile</td><td>Android view hierarchy structure</td><td>(1) Screen Question Generation, (2) Screen Summarization, (3) Screen Question Answering, and (4) Mapping Instruction to UI Action</td><td>PaLM [507</td><td>Single-agent</td><td>The first paper to study Screen Question Generation and Screen QA using LLMs</td><td>https: //github.com/ google-research google-research tree/master/ llm4mobile</td></tr><tr><td>VisionTasker 299]</td><td>Android mobile devices</td><td>UI screenshots with widget detection and text extraction</td><td>UI operations such as tapping, swiping, and entering text</td><td>ERNIE Bot 508</td><td>Single-agent with vision-based UI understanding and sequential task planning</td><td>Vision-based UI understanding approach, which allows it to interpret UI semantics directly from screenshots with out view hierarchy dependencies</td><td>https: //github.com/ AkimotoAyako/ VisionTasker</td></tr><tr><td>DroidBot-GPT [300]</td><td>Android mobile devices</td><td>Translates the GUI state information of Android applications into natural language prompts</td><td>UI operations, including actions like click, scroll, check, and edit</td><td>GPT</td><td>Single-agent</td><td>Automates Android applica tions without modifications to either the app or the model</td><td>https: //github.com/ MobileLLM/ DroidBot-GPT</td></tr><tr><td>CoCo-Agent 301]</td><td>Android mobile devices</td><td>GUI screen-shots, OCR layouts, and historical actions</td><td>GUI actions, such as clicking, scrolling, and typing</td><td>CLIP 509] for vision encoding and LLaMA-2- chat-7B for language processing</td><td>Single-agent</td><td>Its dual approach of Comprehensive Environment Perception and Conditional Action Prediction</td><td>https://github.<br> com/xbmxb/ <br> CoCo-Agent</td></tr><tr><td></td><td>Android mobile devices</td><td>GUI screen-shots</td><td>GUI operations</td><td>BLIP-2 vision encoder 206 with a FLAN-Alpaca [78]</td><td>Single-agent with chain-of-action</td><td>Its direct interaction with GUI elements. Its chain-of-action mechanism enables it to leverage both past and planned actions</td><td>https://github.<br> com/co0elf/ <br> Auto-GUI</td></tr><tr><td>MobileGPT 310]</td><td>Android mobile devices</td><td>Simplified HTML representation</td><td>Standard UI operations and navigation actions</td><td>GPT-4-turbo for screen understanding and reasoning. GPT-3.5-turbo for slot-filling sub-task parameters</td><td>Single-agent architecture augmented by a hierarchical memory structure</td><td>Introduces a human-like app memory that allows for task decomposition into modular sub-tasks</td><td>https: //mobile-gpt. github.io</td></tr><tr><td>MM-Navigator 303]</td><td>Mobile iOS and Android</td><td>Smartphone screenshots with associated set-of-mark tags</td><td>Clickable UI operations</td><td>GPT-4V</td><td>Single-agent</td><td>Using set-of-mark prompting with GPT-4V for precise GUI navigation on smartphones</td><td>https://github.com/zzxslp/ MM-Navigator</td></tr><tr><td>AppAgent</td><td>Android mobile devices</td><td>Real-time screenshots and XML files detailing the interactive elements</td><td>User-like actions. like Tap, Long press, Swipe, Text input, Back and Exit</td><td>GPT-4V</td><td>Single-agent</td><td>Its ability to perform tasks on any smartphone app us ing a human-like interaction method</td><td>https:// appagent-official github.io/</td></tr><tr><td>AppAgent-V2 304]</td><td>Android mobile devices</td><td>GUI screen-shots witl annotated elements, OCR for detecting text and icons. Structured XML metadata</td><td>Standard UI Operations: Tap, text input, long press, swipe, back, and stop</td><td>GPT-4</td><td>Multi-phase architecture witl Exploration Phase and Deployment Phase</td><td>Enhances adaptability and precision in mobile environ ments by combining structured data parsing with visual features</td><td>/</td></tr><tr><td>FedMobileAge <br> 314]</td><td>entAndroid mobile devices</td><td>GUI Screen-shots</td><td>Standard UI operations</td><td>Qwen2-VL-Instruct-7B 231]</td><td>Multi-agent federated learning</td><td>Introduces preserving learning for mobile automation, enabling large-scale without centralized human annotation. privacy-federated training</td><td></td></tr></table>

<!-- Media -->

Addressing the need for efficient handling of multi-step tasks with lower computational costs, AutoDroid [156] combines LLM-based comprehension with app-specific knowledge. Using an HTML-style GUI representation and a memory-based approach, AutoDroid reduces dependency on extensive LLM queries. Its hybrid architecture of cloud and on-device models enhances responsiveness and accessibility, making AutoDroid a practical solution for diverse mobile tasks. AutoDroid-V2 [305] enhances its predecessor AutoDroid, by utilizing on-device language models to generate and execute multi-step scripts for user task automation. By transforming dynamic and complex GUI elements of mobile apps into structured app documents, it achieves efficient and accurate automation without depending on cloud-based resources. The script-based approach reduces computational overhead by minimizing query frequency, thereby improving task efficiency and addressing the limitations of stepwise agents. This advancement enables privacy-preserving and scalable task automation on mobile platforms.


<!-- Meanless: 36 -->

<!-- Media -->

JOURNAL OF LATEX CLASS FILES, DECEMBER 2024

TABLE 14: Overview of LLM-brained GUI agent frameworks on mobile platforms (Part II).

<table><tr><td>Agent</td><td>Platform</td><td>Perception</td><td>Action</td><td>Model</td><td>Architecture</td><td>Highlight</td><td>Link</td></tr><tr><td>Prompt2Task 315]</td><td>Android mobile devices</td><td>GUI structure and layout hierarchy, full-page textual descriptions OCR-based text extraction</td><td>Standard UI operations</td><td>GPT-4</td><td>Multi-agent architecture</td><td>Enables UI automation through free-form textual prompts, eliminating the need for users to script automation tasks.</td><td>https: //github.com/ PromptRPA Prompt2TaskDataset</td></tr><tr><td>ClickAgent 312]</td><td>Android Mobile Devices</td><td>Screenshots</td><td>Standard UI operations</td><td>InternVL-2.0 379], TinyClick 337], SeeClick 25</td><td>Single-agent</td><td>Combines MLLM reasoning with a dedicated UI location model to enhance UI interaction accuracy</td><td>https://github.com/Samsung/ ClickAgent</td></tr><tr><td>AutoDroid 156]</td><td>Android mobile devices</td><td>Simplified HTML-style representation</td><td>Standard UI operations</td><td>GP f-3.5, GPT-4, and Vicuna-7B 510]</td><td>Single-agent architecture</td><td>Its use of app-specific knowledge and a multi-granularity query optimization module to reduce the computational cost</td><td>https: //autodroid-sys. github.io/</td></tr><tr><td>AutoDroid-V2 305]</td><td>Android Mobile Devices</td><td>Structured GUI Representations</td><td>Multi-step scripts of standard UI operations and API calls</td><td>LIama-3.1-8B 91]</td><td>Script-based architecture.</td><td>Converts GUI task automation into a script generation problem, enhancing efficiency and task success rates.</td><td>/</td></tr><tr><td>CoAT [298]</td><td>Android mobile devices</td><td>Screenshot-based context and semantic information</td><td>Standard UI operations</td><td>GPT-4V</td><td>Single-agent architecture</td><td>The integration of a chain-of-action-thought process, which explicitly maps each action to screen descriptions, reasoning steps, and anticipated outcomes</td><td>https: //github.com/ ZhangL-HKU/ CoAT</td></tr><tr><td>Mobile-Agent [158]</td><td>Mobile Android</td><td>Screenshots with detection</td><td>Standard UI operations</td><td>GPT-4V with Grounding DINO 1831 and CLIP 509 for icon detection</td><td>Single-agent</td><td>Vision-centric approach that eliminates dependency on system-specific data</td><td>https://github.com/X-PLUG/ MobileAgent</td></tr><tr><td>Mobile-Agent-v2 306</td><td>Mobile Android OS and Harmony OS</td><td>Screenshots with text, icon recognition, and description</td><td>Standard UI operations on mobile phones</td><td>GPT-4V with Grounding DINO 183] and Qwen-VL-Int4 511]</td><td>Multi-agent architecture with Planning Agent, Decision Agent, and Reflection Agent</td><td>Multi-agent architecture enhances task navigation for long-sequence operations</td><td>https://github.com/X-PLUG/ MobileAgent</td></tr><tr><td>Mobile-Experts 307</td><td>Mobile Android</td><td>Interface memory and procedural memory</td><td>Standard UI operations and code-combined tool formulation</td><td>VLMs</td><td>Multi-agent framework with double-layer planning</td><td>Code-combined tool formulation method and double-layer planning mechanism for collaborative task execution</td><td>/</td></tr><tr><td>LiMAC 308</td><td>Mobile Android</td><td>Screenshots and corresponding widget trees</td><td>Standard UI operations</td><td>Lightweight transformer and fine-tuned VLMs</td><td>Single-agent</td><td>Balances computational efficiency and natural language understanding</td><td>/</td></tr><tr><td>MobA 309</td><td>Mobile Android</td><td>GUI structures. screenshots with annotation</td><td>Standard UI operations and API function calls</td><td>GPT-4</td><td>Two-level agent: a Global Agent and a Local Agent</td><td>Two-level agent system that separates task planning and execution into two specialized agents</td><td>https://github.com/OpenDFM/ MobA</td></tr><tr><td>Mobile-Agent-E 311]</td><td>Mobile Android</td><td>GUI screen-shots, OCR for detecting text and icons</td><td>Standard UI operations and APIs</td><td>GPT-4o, Claude- 3.5-Sonnet. Gemini-1.5-Pro</td><td>Hierarchical Multi-Agent System</td><td>Hierarchical multi-agent framework that separates planning from execution for improved long-term reasoning and self-evolution, enabling the system to learn reusable tips and shortcuts</td><td>https: //x-plug.github. io/MobileAgent</td></tr></table>

<!-- Media -->

MobileGPT [310] automates tasks on Android devices using a human-like app memory system that emulates the cognitive process of task decomposition-Explore, Select, Derive, and Recall. This approach results in highly efficient and accurate task automation. Its hierarchical memory structure supports modular, reusable, and adaptable tasks and sub-tasks across diverse contexts. MobileGPT demonstrates superior performance over state-of-the-art systems in task success rates, cost efficiency, and adaptability, highlighting its potential for advancing mobile task automation.

In a more advanced distributed setting, FedMobileAgent 314] employs a federated learning framework to train mobile automation agents using self-sourced data from users' phone interactions. It addresses the high cost and privacy concerns associated with traditional human-annotated datasets by introducing Auto-Annotation, which leverages vision-language models (VLMs) to infer user intentions from screenshots and actions. The system enables decentralized training through federated learning while preserving user privacy, and its adaptive aggregation method enhances model performance under non-IID data conditions. Experimental results on several mobile benchmarks demonstrate that FedMobileAgent achieves performance comparable to human-annotated models at a fraction of the cost.

In summary, mobile GUI agents have evolved significantly, progressing from single-agent systems to complex, multi-agent frameworks capable of dynamic, context-aware operations. These innovations demonstrate that sophisticated architectures, multimodal processing, and advanced planning strategies are essential in handling the diverse challenges of mobile environments, marking significant advancements in mobile automation capabilities.


<!-- Meanless: 37 -->

<!-- Media -->

JOURNAL OF LATEX CLASS FILES, DECEMBER 2024

TABLE 15: Overview of LLM-brained GUI agent frameworks on mobile platforms (Part III).

<table><tr><td>Agent</td><td>Platform</td><td>Perception</td><td>Action</td><td>Model</td><td>Architecture</td><td>Highlight</td><td>Link</td></tr><tr><td>ReachAgent 313]</td><td>Android mobile devices</td><td>GUI Screen-XML document</td><td>Standard UI operations</td><td>MobileVLM 353]</td><td>Single-agent, two-stage training</td><td>Divides tasks into subtasks: "Page Reaching" (navigating to the correct screen) and "Page Operation" (performing actions on the screen). using RL with preference-based training to improve long-term task success.</td><td>/</td></tr><tr><td>Mobile-Agent-V 316]</td><td>Mobile Android</td><td>Video guidance, XML lierarchy</td><td>Standard UI operations</td><td>GPT-40</td><td>Multi-agent system</td><td>Introduces video-guided learning, allowing the agent to acquire operational knowledge efficiently.</td><td>https://github.com/X-PLUG/ MobileAgent</td></tr><tr><td>MobileSteward 317]</td><td>Mobile Android</td><td>XML layouts, Screenshots</td><td>Standard UI interactions, Code execution</td><td>GPT-4V, GPT-4o</td><td>App-oriented multi-agent framework</td><td>Introduces an app-oriented multi-agent framework with self-evolution, overcoming the complexity of cross-app interactions by dynamically recruiting specialized agents.</td><td>https://github com/XiaoMi/ MobileSteward</td></tr><tr><td>AppAgentX 258]</td><td>Mobile Android</td><td>Screenshots</td><td>Standard UI operations</td><td>GPT-40</td><td>Single-agent architecture</td><td>Introduces an evolutionary mechanism that enables dynamic learning from past interactions and replaces inefficient low-level operations with high-level actions</td><td>https: //appagentx. github.io/</td></tr><tr><td>CHOP 318]</td><td>Mobile Android</td><td>Screenshots</td><td>Standard UI operations</td><td>GPT-40</td><td>Multi-agent architecture</td><td>Introduces a basis subtask framework, where subtasks are predefined based on human task decomposition patterns, ensuring better exe-cutability and efficiency.</td><td>https://github.com/Yuqi-Zhou/ CHOP</td></tr><tr><td>OS-Kairos 319]</td><td>Mobile Android</td><td>GUI shots</td><td>Standard UI operations</td><td>OS-Atlas-Pro- 7B and GPT-40</td><td>Single-agent with critic-in-the-loop design</td><td>Introduces an adaptive interaction framework where each GUI action is paired with a confidence score, dynamically deciding between autonomous execution and human intervention</td><td>https: //github.com/ Wuzheng02/ OS-Kairos</td></tr><tr><td>V-Droid 320]</td><td>Mobile Android</td><td>Android Accessibility Tree</td><td>Standard UI operations</td><td>LLaMA-3.1-8B-Instruct [91]</td><td>Verifier-Driven Single-Agent Architecture</td><td>Introduces a novel verifier-driven architecture where the LLM does not generate actions directly but instead scores and selects from a finite set of extracted actions, improving task success rates and significantly reducing latency</td><td>/</td></tr><tr><td>LearnAct 321]</td><td>Mobile Android</td><td>GUI screen-shots, UI trees, and demonstration trajectories</td><td>Standard GUI actions</td><td>Gemini-1.5-Pro, UI-TARS-7BSFT, Qwen2-VL- 7B</td><td>Multi-agent</td><td>Introduces a structured. demonstration-based learning pipeline for mobile GUI agents. It addresses long-tail generalization via few-shot demonstrations, achieving substantial performance gains on complex real-world mobile tasks</td><td>https: //Igy0404.github io/LearnAct</td></tr><tr><td>AndroidGen 322]</td><td>Mobile Android</td><td>XML UI structure</td><td>Standard GUI actions</td><td>GLM-4-9B 499 / LLaMA-3-70B 91]</td><td>Multi-module single-agent</td><td>Innovatively addresses data scarcity for Android agents through a self-improving architecture, a zero human-annotation training pipeline. and effective generalization from easy to hard tasks</td><td>https://github.com/THUDM/ AndroidGen</td></tr><tr><td>Agent-Initiated Interaction 324]</td><td>Android Mobile</td><td>Accessibility tree and screenshots</td><td>Standard GUI operations</td><td>Gemini 1.5</td><td>Single-agent architecture</td><td>Pioneers agent-initiated interaction in mobile UI automation</td><td>https: //github.com/ google-research google-research tree/master/ android interaction</td></tr><tr><td>Latent State Estimation 325]</td><td>Android Mobile</td><td>Accessibility tree</td><td>Standard GUI operations</td><td>PaLM 2</td><td>Two-module design with Reasoner and Grounder</td><td>First to formalize the estimation of latent UI states using LLMs to support UI automation</td><td>7</td></tr></table>

<!-- Media -->

### 6.3 Computer GUI Agents

Computer GUI agents have evolved to offer complex automation capabilities across diverse operating systems, addressing challenges such as cross-application interaction, task generalization, and high-level task planning. They have led to the development of sophisticated frameworks capable of handling complex tasks across desktop environments. These agents have evolved from simple automation tools to intelligent systems that leverage multimodal inputs, advanced architectures, and adaptive learning to perform multi-application tasks with high efficiency and adaptability. We provide an overview of computer GUI agent frameworks in Table 16 and 17


<!-- Meanless: 38 -->

<!-- Media -->

JOURNAL OF LATEX CLASS FILES, DECEMBER 2024

TABLE 16: Overview of LLM-brained GUI agent frameworks on computer platforms (Part I)..

<table><tr><td>Agent</td><td>Platform</td><td>Perception</td><td>Action</td><td>Model</td><td>Architecture</td><td>Highlight</td><td>Link</td></tr><tr><td>UFO [19]</td><td>Windows computer</td><td>Screenshots with annotated controls, and properties</td><td>Standard UI operations with additional customized operations</td><td>GPT-Vision</td><td>Dual-agent architecture, consisting of a HostAgent (for application selection and global planning) and an AppAgent (for specific task execution within applications)</td><td>Its dual-agent system that seamlessly navigates and interacts with multiple applications to fulfill complex user requests in natural language on Windows OS</td><td>https: //github.com/ microsoft/UFO</td></tr><tr><td>UFO2 334j</td><td>Windows desktops</td><td>GUI screen-shots and textual control properties list</td><td>Unified GUI-API action layer</td><td>GPT-40 (and GPT-4V, o1, Gemini-Flash); Vision grounding via OmniParser-v2</td><td>Centralized HostAgent with application-specialized AppAgents</td><td>Transforms a conventional CUA into an OS-native, pluggable AgentOS with deep Windows integration, hybrid GUI-API actions, vision + UIA perception, speculative multi-action planning, retrieval-augmented knowledge, and a non-intrusive PiP virtual desktop</td><td>https: //github.com/ microsoft/UFO/</td></tr><tr><td>ScreenAgent 366]</td><td>Linux and Windows desktop</td><td>Screenshots</td><td>Standard UI operations</td><td>ScreenAgent model</td><td>Single-agent</td><td>Integrated planning-acting-reflecting pipeline that simulates a continuous thought process</td><td>https: //github.com/ niuzaisheng/ ScreenAgent</td></tr><tr><td>OS-Copilot [162]</td><td>Linux and MacOS computer</td><td>Unified interface that includes mouse and keyboard control, rol. API calls, and Bash or P Python interpreters</td><td>Standard operations, Bash and Python commands, as well as API calls</td><td>GPT-4</td><td>Multi-component architecture involving a planner, configura-tor, actor, and critic modules</td><td>Self-directed learning capability, allowing it to adapt to new applications by autonomously generating and refining tools</td><td>https: <br> //os-copilot. <br> github.io/</td></tr><tr><td>Cradle [161]</td><td>Windows computer</td><td>Complete screen videos with Grounding DINO 183] and SAM 182 for object detection and localization</td><td>Keyboard and mouse actions</td><td>GPT-4</td><td>Modular single-agent architecture</td><td>Its generalizability across various digital environments. allowing it to operate without relying on internal APIs</td><td>https: //baai-agents. github.io/ Cradle/</td></tr><tr><td>Agent S 326]</td><td>Ubuntu and Windows computer</td><td>Screenshots and accessibility tree</td><td>Standard UI operations and system-level controls</td><td>GPT-4 and Claude-3.5 Sonnet 1631</td><td>Multi-agent architecture comprising a Manager and Worker structure</td><td>Experience-augmented hierarchical planning</td><td>https://github.com/simular-ai/ Agent-S</td></tr><tr><td>GUI Narrator 327]</td><td>Windows computer</td><td>High-resolution screenshots</td><td>Standard UI operations</td><td>GPT-4 and QwenVL-7B 511]</td><td>Two-stage architecture, detecting the cursor location and selecting keyframes. then generating action captions</td><td>Uses the cursor as a focal point to improve understanding of high-resolution GUI actions</td><td>https://showlab.github.io/ GUI-Narrator</td></tr><tr><td>PC Agent 329]</td><td>Windows Computer</td><td>Screenshots and based tracking</td><td>Standard UI Operations</td><td>Qwen2-VL-72B-Instruct 231] and Molmo 512</td><td>A planning agent for decision-making combined with a arounding agent for executing actions.</td><td>Human cognition transfer framework, which transforms raw interaction data into cognitive trajectories to enable complex computer tasks.</td><td>https: //gair-nlp.github io/PC-Agent/</td></tr></table>

<!-- Media -->

One significant development in this area is the introduction of multi-agent architectures that enhance task management and execution. For instance, the UI-Focused Agent, UFO [19] represents a pioneering framework specifically designed for the Windows operating system. UFO redefines UI-focused automation through its advanced dual-agent architecture, leveraging GPT-Vision to interpret GUI elements and execute actions autonomously across multiple applications. The framework comprises a HostAgent, responsible for global planning, task decomposition, and application selection, and an AppAgent, tasked with executing assigned subtasks within individual applications, as illustrated in Figure 23. This centralized structure enables UFO to manage complex, multi-application workflows such as aggregating information and generating reports. Similar architectural approach has also been adopted by other GUI agent frameworks [307], [309], [493]. By incorporating safeguards and customizable actions, UFO ensures efficiency and security when handling intricate commands, positioning itself as a cutting-edge assistant for Windows OS. Its architecture, exemplifies dynamic adaptability and robust task-solving capabilities across diverse applications, demonstrating the potential of multi-agent systems in desktop automation.

UFO ${}^{2}$ 334, the successor to UFO, elevates GUI automation from a vision-only prototype to a deeply integrated, Windows-native AgentOS (Figure 24). It coordinates tasks through a centralized HostAgent, which delegates subtasks to application-specialized AppAgents. A hybrid perception pipeline that fuses Windows UI Automation (UIA) metadata with OmniParser-v2 visual grounding delivers robust control identification even for custom widgets. Via a unified GUI-API action layer, AppAgents preferentially invoke high-level application APIs and fall back to pixel-level clicks only when necessary, cutting both latency and brittleness. A picture-in-picture virtual desktop cleanly isolates agent execution from the user's main session, enabling non-intrusive


<!-- Meanless: 39 -->

<!-- Media -->

<!-- figureText: HostAgent<br>User Request<br>Sub-tasks<br>Assignment<br>AppAgent 1<br>AppAgent 2<br>AppAgent 3<br>AppAgent n<br>P<br>X<br>W<br>PowerPoint<br>Edge Browser<br>Excel<br>Word -->

<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2025/11/2025_11_04__12_39_04_212553.jpg"/>

Fig. 23: The multi-agent architecture employed in UFO [19]. Figure adapted from the original paper.

<!-- Media -->

<!-- Media -->

JOURNAL OF LATEX CLASS FILES, DECEMBER 2024

TABLE 17: Overview of LLM-brained GUI agent frameworks on computer platforms (Part II).

<table><tr><td>Agent</td><td>Platform</td><td>Perception</td><td>Action</td><td>Model</td><td>Architecture</td><td>Highlight</td><td>Link</td></tr><tr><td>Zero-shot Agent 328]</td><td>Computer</td><td>HTML code and DOM</td><td>Standard UI operations</td><td>PaLM-2 507]</td><td>Single-agent</td><td>Zero-shot capability in performing computer control tasks</td><td>https: <br> //github.com/ <br> google-research <br> google-research <br> tree/master <br> zero_shot <br> structured <br> reflection</td></tr><tr><td>PC-Agent 330]</td><td>Windows computers</td><td>UI tree, Screen-shots</td><td>Standard UI operations</td><td>GPT-40</td><td>Hierarchica Multi-Agent</td><td>PC-Agent's hierarchica multi-agent design enables efficient decomposition of complex PC tasks. Its Active Perception Module enhances fine-grained GUI understanding by combining accessibility structures, OCR, and intention grounding.</td><td>https://github com/X-PLUG MobileAgent tree/main PC-Agent</td></tr><tr><td>PwP 331]</td><td>VSCode-based IDE Computers</td><td>Screenshots, File system access, Terminal outputs</td><td>Standard UI interactions, File operations, Bash commands, Tools in VSCode</td><td>GPT-40, Claude- 3.5 Sonnet, Gemini-1.5</td><td>Single-agent architecture</td><td>Shifts software engineering agents from API-based tool interactions to direct GUI based computer use, allowing agents to interact with an IDE as a human developer would.</td><td>https:// <br> programmingwithpixels <br> com</td></tr><tr><td>COLA 332]</td><td>Windows computers</td><td>GUI structure, properties and screenshots</td><td>Standard UI operations and system APIs</td><td>GPT-40</td><td>Hierarchical Multi-Agent</td><td>A dynamic task scheduling mechanism with a plug-and play agent pool, enabling adaptive handling of GUI tasks</td><td>https://github.com/Alokia/ COLA-demo</td></tr><tr><td>STEVE 333]</td><td>Windows Desktop</td><td>screen-shots and A11y Tree</td><td>Standard UI operations</td><td>Qwen2-VL 231 and GPT-40</td><td>Single-agent</td><td>Introduces a scalable step verification pipeline using GPT-40 to generate binary labels for agent actions, and applies KTO optimization to incorporate both positive and negative actions into agent learning</td><td>https://github com/FanbinLu/ STEVE</td></tr><tr><td>TaskMind 335]</td><td>Windows Computer</td><td>Standard GUI actions</td><td>GPT-3.5 / GPT-4</td><td>Single-agent architecture</td><td>Introduces novel task graph representation with cognitive dependencies. enabling LLMs to better generalize demonstrated GUI tasks</td><td>https://github.com/ Evennaire/TaskMind</td><td></td></tr></table>

<!-- Media -->

multitasking. Runtime performance is further boosted by retrieval-augmented help documents and execution logs, coupled with speculative multi-action planning that executes several steps per single LLM invocation. Tested on 20+ real Windows applications, ${\mathbf{{UFO}}}^{2}$ exceeds Operator [513] and other CUAs by more than 10 percentage points in success rate while halving LLM calls. Because the framework is model-agnostic, swapping GPT-40 for a stronger LLM such as o1 yields additional gains without code changes.

<!-- Media -->

<!-- figureText: (a) Traditional CUAs<br>☒<br>(b) AgentOS<br>Deep OS Integration<br>Screenshot<br>Shallow OS Integration<br>GUI Actions<br>Screenshot<br>GUI actions<br>Arryinfo<br>APIs<br>Knowledge<br>Picture-in-Picture UX<br>-<br>Locking -->

<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2025/11/2025_11_04__12_39_04_e4a0a7.jpg"/>

Fig. 24: The comparison of traditional CUAs and the Desktop AgentOS UFO2. Figure adapted from the original paper.

<!-- Media -->

Building upon the theme of adaptability and generalist capabilities, Cradle [161] pushes the boundaries of general computer control by utilizing VLMs for interacting with various software, ranging from games to professional applications, without the need for API access. Cradle employs GPT-40 to interpret screen inputs and perform low-level actions, making it versatile across different types of software environments. Its six-module structure, covering functions such as information gathering and self-reflection, enables the agent to execute tasks, reason about actions, and utilize past interactions to inform future decisions. Cradle's capacity to function in dynamic environments, including complex software, marks it as a significant step toward creating generalist agents with broad applicability across desktop environments.


<!-- Meanless: JOURNAL OF LATEX CLASS FILES, DECEMBER 2024<br>40 -->

Extending the capabilities of computer GUI agents to multiple operating systems, OS-Copilot [162] introduces a general-purpose framework designed to operate across Linux and macOS systems. Its notable feature, FRIDAY, showcases the potential of self-directed learning by adapting to various applications and performing tasks without explicit training for each app. Unlike application-specific agents, FRIDAY integrates APIs, keyboard and mouse controls, and command-line operations, creating a flexible platform that can autonomously generate and refine tools as it interacts with new applications. OS-Copilot's ability to generalize across unseen applications, validated by its performance on the GAIA benchmark, provides a foundational model for OS-level agents capable of evolving in complex environments. This demonstrates promising directions for creating adaptable digital assistants that can handle diverse desktop environments and complex task requirements.

In the emerging field of LLM-powered GUI agents for desktop environments, Programming with Pixels (PwP) 331 introduces a compelling alternative to traditional tool-based software engineering agents. Rather than relying on predefined API calls, PwP enables agents to interact directly with an IDE using visual perception, keyboard inputs, and mouse clicks, mimicking the way human developers operate within an IDE. This approach allows for generalization beyond predefined APIs, providing a highly expressive environment where agents can execute a wide range of software engineering tasks, including debugging, UI generation, and code editing. Evaluations conducted on PwP-Bench demonstrate that computer-use agents, despite lacking direct access to structured APIs, can match or even surpass traditional tool-based approaches in certain scenarios.

In summary, computer GUI agents have evolved significantly, progressing from single-task automation tools to advanced multi-agent systems capable of performing complex, multi-application tasks and learning from interactions. Frameworks like UFO, Cradle, and OS-Copilot illustrate the potential of adaptable, generalist agents in desktop automation, paving the way for the evolution of more intelligent and versatile AgentOS frameworks.

### 6.4 Cross-Platform GUI Agents

Cross-platform GUI agents have emerged as versatile solutions capable of interacting with various environments, from desktop and mobile platforms to more complex systems. These frameworks prioritize adaptability and efficiency, leveraging both lightweight models and multi-agent architectures to enhance cross-platform operability. In this subsection, we first We overview cross-platform GUI agent frameworks in Table 18 then explore key frameworks that exemplify the advancements in cross-platform GUI automation.

A significant stride in this domain is represented by Au-toGLM [336], which bridges the gap between web browsing and Android control by integrating large multimodal models for seamless GUI interactions across platforms. AutoGLM introduces an Intermediate Interface Design that separates planning and grounding tasks, improving dynamic decision-making and adaptability. By employing a self-evolving online curriculum with reinforcement learning, the agent learns incrementally from real-world feedback and can recover from errors. This adaptability and robustness make AutoGLM ideal for real-world deployment in diverse user applications, setting a new standard in cross-platform automation and offering promising directions for future research in foundation agents.

While some frameworks focus on integrating advanced models for cross-platform interactions, others emphasize efficiency and accessibility. TinyClick [337] addresses the need for lightweight solutions by focusing on single-turn interactions within GUIs. Utilizing the Florence-2-Base Vision-Language Model, TinyClick executes tasks based on user commands and screenshots with only 0.27 billion parameters. Despite its compact size, it achieves high accuracy-73% on Screenspot [25] and 58.3% on OmniAct [459] —outperforming larger multimodal models like GPT-4V while maintaining efficiency. Its multi-task training and MLLM-based data augmentation enable precise UI element localization, making it suitable for low-resource environments and addressing latency and resource constraints in UI grounding and action execution.

In addition to lightweight models, multi-agent architectures play a crucial role in enhancing cross-platform GUI interactions. OSCAR 340 exemplifies this approach by introducing a generalist GUI agent capable of autonomously navigating and controlling both desktop and mobile applications. By utilizing a state machine architecture, OSCAR dynamically handles errors and adjusts its actions based on real-time feedback, making it suitable for automating complex workflows guided by natural language. The integration of standardized OS controls, such as keyboard and mouse inputs, allows OSCAR to interact with applications in a generalized manner, improving productivity across diverse GUI environments. Its open-source design promotes broad adoption and seamless integration, offering a versatile tool for cross-platform task automation and productivity enhancement.

Expanding on the concept of multi-agent systems, AgentStore 341 introduces a flexible and scalable framework for integrating heterogeneous agents to automate tasks across operating systems. The key feature of AgentStore is the MetaAgent, which uses the innovative AgentToken strategy to dynamically manage a growing number of specialized agents. By enabling dynamic agent enrollment, the framework fosters adaptability and scalability, allowing both specialized and generalist capabilities to coexist. This multi-agent architecture supports diverse platforms, including desktop and mobile environments, leveraging multimodal perceptions such as GUI structures and system states. AgentStore's contributions highlight the importance of combining specialization with generalist capabilities to overcome the limitations of previous systems.


<!-- Meanless: JOURNAL OF LATEX CLASS FILES, DECEMBER 2024<br>41 -->

<!-- Media -->

TABLE 18: Overview of LLM-brained cross-platform GUI agent frameworks.

<table><tr><td>Agent</td><td>Platform</td><td>Perception</td><td>Action</td><td>Model</td><td>Architecture</td><td>Highlight</td><td>Link</td></tr><tr><td>AutoGLM 336]</td><td>Web and Mobile Android</td><td>Screenshots with SoM annotation and OCR</td><td>Standard UI operations, Native API interactions, and AI-driven actions</td><td>ChatGLM 499</td><td>Single-agent architecture</td><td>Self-evolving online curriculum RL framework, which enables continuous improvement by interacting with real-world environments</td><td>https:// xiao9905.github io/AutoGLM/</td></tr><tr><td>TinyClick 337]</td><td>Web, Mobile, and Windows platforms</td><td>GUI screen-shots</td><td>Standard UI operations, Native API interactions, and AI-driven actions</td><td>Florence-2- Base VLM 514</td><td>Single-agent, with single-turn tasks</td><td>Compact size (0.27B parameters) with high performance</td><td>https: //huggingface co/Samsung/ TinyClick</td></tr><tr><td>OSCAR 340]</td><td>Desktop and Mobile</td><td>Screenshots</td><td>Standard UI operations</td><td>GPT-4</td><td>Single-agent architecture</td><td>Ability to adapt to real-time feedback and dynamically adjust its actions</td><td>/</td></tr><tr><td>AgentStore 341]</td><td>Desktop and mobile environments</td><td>GUI structures and properties. accessibility trees, screenshots and terminal output etc</td><td>Standard UI operations, API calls</td><td>GPT-40 an InternVL2-8B 379]</td><td>Multi-agent architecture</td><td>Dynamically integrate a wide variety of heterogeneous agents, enabling both spe cialized and generalist capabilities</td><td>https: //chengyou-jia. github.io/ AgentStore-Home/</td></tr><tr><td>MMAC-Copilot 249]</td><td>Windows OS Desktop, mobile applications, and game environments</td><td>Screenshots</td><td>Standard UI operations. Native APIs, and Collaborative multi-agent actions</td><td>GPT-4V, SeeClick 25 and Genimi Vision different agents</td><td>Multi-agent architecture with Planner, Programmer Viewer, N Mentor Video Analyst, and Librarian</td><td>Collaborative multi-agent architecture where agents specialize in specific tasks</td><td>/</td></tr><tr><td>AGUVIS 236</td><td>Web, desktop, and mobile</td><td>Image-based observations</td><td>Standard UI operations</td><td>Fine-tuned Qwen2-VL 231]</td><td>Single-agent architecture</td><td>Pure vision-based approach for GUI interaction, bypassing textual UI representations and enabling robust cross-platform generalization</td><td>https: //aguvis-project. github.io</td></tr><tr><td>Ponder Press 342]</td><td>Web, Android, iOS Mobile, Windows, and macOS</td><td>Purely visual inputs</td><td>Standard UI operations</td><td>GPT-40 and Claude Sonnet high-level task decomposition, a fine-tuned Qwen2-VL-Instruct 231 for GUI element grounding</td><td>Divide-and-conquer architecture</td><td>Purely vision-based GUI agent that does not require non-visual inputs</td><td>https: //invinciblewyq. github.io/ ponder-press-page</td></tr><tr><td>InfiGUIAgent 343]</td><td>Mobile, Web, Desktop</td><td>Raw screen-shots</td><td>Standard UI operations.</td><td>Qwen2-VL-2B 231]</td><td>Single-agent architecture enhanced by hierarchical reasoning.</td><td>Introduces native reasoning skills, such as hierarchical and expectation-reflection reasoning, enabling advanced and adaptive task handling.</td><td>https: //github.com/ RealIm-Labs/ InfiGUIAgent</td></tr><tr><td>Learn-by-Interact [338]</td><td>Web, code development, and desktops</td><td>GUI screen-shots with SoM and accessibility tree</td><td>Standard UI interactions and code execution</td><td>Claude-3.5- Sonnet, Gemini- 1.5-Pro CodeGemma- 7B, CodeStral- 22B</td><td>Multi-agent</td><td>Introduces autonomous data synthesis process, eliminating the need for human-labeled agentic data</td><td>/</td></tr><tr><td>CollabUIAgen 339]</td><td>sMobile Android, Web</td><td>Screenshots, UI trees</td><td>Standard UI operations</td><td>Qwen2-7B 231], GPT-4</td><td>Multi-agent system</td><td>A multi-agent reinforcement learning framework that introduces a Credit Re-Assignment (CR) strategy, using LLMs instead of environment specific rewards to enhance performance and generalization.</td><td>https: //github.com/ THUNLP-MT/ CollabUIAgents</td></tr><tr><td>Agent S2 344]</td><td>Ubuntu, Windows, Android</td><td>GUI screenshot</td><td>Standard UI operations and system APIs</td><td>Claude-3.7- Sonnet, Claude- 3.5-Sonnet, GPT-40 (for Manager and Worker roles), UI-TARS-72B-DPO, Tesseract OCR, and UNO (for grounding experts)</td><td>Compositional multi-agent architecture with a Manager for planning, a Worker for execution, and a Mixture of Grounding experts</td><td>Features a Mixture of Grounding technique and Proactive Hierarchical Planning, enabling more accurate grounding and adaptive replanning in long-horizon tasks</td><td>https://github.com/simular-ai/ Agent-S</td></tr><tr><td>GuidNav 345]</td><td>Android and Web</td><td>GUI shots screen-</td><td>Standard UI operations and system APIs</td><td>GPT-40, Gemini 2.0 Flash, Qwen-VL-Plus</td><td>Single-agent</td><td>Introduces a novel process reward model that provides fine-grained, step-level feedback to enhance GUI task accuracy and success</td><td>/</td></tr><tr><td>ScaleTrack 346]</td><td>Web, Android Mobile, and Desktop Computers</td><td>GUI shots screen-</td><td>Standard GUI operations</td><td>Qwen2-VL-7B</td><td>Single-agent</td><td>First GUI agent framework to introduce backtracking—learning not only the next action but also historical action sequences</td><td>/</td></tr></table>

<!-- Media -->


<!-- Meanless: JOURNAL OF LATEX CLASS FILES, DECEMBER 2024<br>42 -->

Further advancing cross-platform GUI interaction, MMAC-Copilot [249] employs a multi-agent, multimodal approach to handle tasks across 3D gaming, office, and mobile applications without relying on APIs. By utilizing specialized agents like Planner, Viewer, and Programmer, MMAC-Copilot collaborates to adapt to the complexities of visually rich environments. Using GPT-4V for visual recognition and OCR for text analysis, it achieves high task completion rates in visually complex environments. The framework's integration with VIBench, a benchmark for non-API applications, underscores its real-world relevance and adaptability. MMAC-Copilot's robust foundation for dynamic interaction across platforms extends applications to industries like gaming, healthcare, and productivity.

AGUVIS 236] leverages a pure vision approach to automate GUI interactions, overcoming limitations of text-based systems like HTML or accessibility trees. Its platform-agnostic design supports web, desktop, and mobile applications while reducing inference costs. AGUVIS employs a two-stage training process: the first focuses on GUI grounding, and the second integrates planning and reasoning within a unified model. This approach delivers state-of-the-art performance in both offline and online scenarios, streamlining decision-making and execution.

Agent S2 [344] builds upon its predecessor, Agent S [326], by introducing a hierarchical and compositional framework for GUI agents that integrates generalist models with specialized grounding modules. Departing from monolithic architectures, it employs a Mixture of Grounding (MoG) strategy to delegate fine-grained grounding tasks to expert modules, and adopts Proactive Hierarchical Planning (PHP) to dynamically revise action plans based on evolving observations. Relying solely on GUI screenshots, Agent S2 generalizes effectively across Ubuntu, Windows, and Android platforms. It demonstrates strong scalability and consistently outperforms larger monolithic models by strategically distributing cognitive responsibilities. The design of Agent S2 underscores the advantages of modular architectures for handling long-horizon, high-fidelity GUI interactions.

In summary, cross-platform GUI agents exemplify the future of versatile automation, offering solutions ranging from lightweight models like TinyClick to sophisticated multi-agent systems such as MMAC-Copilot. Each framework brings unique innovations, contributing to a diverse ecosystem of GUI automation tools that enhance interaction capabilities across varying platforms, and marking significant advancements in cross-platform GUI automation.

### 6.5 Takeaways

The landscape of GUI agent frameworks has seen notable advancements, particularly in terms of multi-agent architectures, multimodal inputs, and enhanced action sets. These developments are laying the groundwork for more versatile and powerful agents capable of handling complex, dynamic environments. Key takeaways from recent advancements include:

1) Multi-Agent Synergy: Multi-agent systems, such as those in UFO [19] and MMAC-Copilot [249], represent a significant trend in GUI agent development. By assigning specialized roles to different agents within a framework, multi-agent systems can enhance task efficiency, adaptability, and overall performance. As agents take on more complex tasks across diverse platforms, the coordinated use of multiple agents is proving to be a powerful approach, enabling agents to handle intricate workflows with greater precision and speed.

2) Multimodal Input Benefits: While some agents still rely solely on text-based inputs (e.g., DOM structures or HTML), incorporating visual inputs, such as screenshots, has shown clear performance advantages. Agents like WebVoyager [269] and SeeAct [17] highlight how visual data, combined with textual inputs, provides a richer representation of the environment state, helping agents make better-informed decisions. This integration of multimodal inputs is essential for accurate interpretation in visually complex or dynamic environments where text alone may not capture all necessary context.

3) Expanding Action Sets Beyond UI Operations: Recent agents have expanded their action sets beyond standard UI operations to include API calls and AI-driven actions, as seen in Hybrid Agent 199 and AutoWebGLM 270]. Incorporating diverse actions allows agents to achieve higher levels of interaction and task completion, particularly in environments where data can be directly retrieved or manipulated through API calls. This flexibility enhances agent capabilities, making them more efficient and adaptable across a wider range of applications.

4) Emerging Techniques for Improved Decision-Making: Novel approaches such as world models in WMA [266] and search-based strategies in Search-Agent 274 represent promising directions for more advanced decision-making. World models allow agents to simulate action outcomes, reducing unnecessary interactions and improving efficiency, especially in long-horizon tasks. Similarly, search-based algorithms like best-first and MCTS help agents explore action pathways more effectively, enhancing their adaptability in complex, real-time environments.

5) Toward Cross-Platform Generalization: Cross-platform frameworks, such as AutoGLM [336] and OSCAR [340], underscore the value of generalizability in GUI agent design. These agents are pioneering efforts to create solutions that work seamlessly across mobile, desktop, and web platforms, moving closer to the goal of a one-stop GUI agent that can operate across multiple ecosystems. Cross-platform flexibility will be crucial for agents that aim to assist users consistently across their digital interactions.

6) Pure Vision-Based Agent: To enable universal GUI control, pure vision-based frameworks have emerged as a prominent solution. These agents rely solely on screenshots for decision-making, eliminating the need for access to metadata such as widget trees or element properties. Notable work like AGUVIS [236] exemplifies this approach. While pure vision-based methods offer greater generalizability and bypass system API limitations, they require strong "grounding" capabilities to accurately locate and interact with UI elements—an ability often lacking in many foundational models. Fine-tuning models specifically for visual grounding and GUI understanding, or integrating GUI parsing techniques like OmniParser [184], can address this challenge and enhance the agent's ability to perform precise interactions.


<!-- Meanless: JOURNAL OF IATEX CLASS FILES, DECEMBER 2024<br>43 -->

The field of GUI agents is moving towards multi-agent architectures, multimodal capabilities, diverse action sets, and novel decision-making strategies. These innovations mark significant steps toward creating intelligent, adaptable agents capable of high performance across varied and dynamic environments. The future of GUI agents lies in the continued refinement of these trends, driving agents towards broader applicability and more sophisticated, human-like interactions across platforms.

## 7 Data for Optimizing LLM-Brained GUI AGENTS

In the previous section, we explored general frameworks for LLM-brained GUI agents, most of which rely on foundational LLMs such as GPT-4V and GPT-4o. However, to elevate these agents' performance and efficiency, optimizing their "brain", the underlying model is crucial. Achieving this often involves fine-tuning foundational models using large-scale, diverse, and high-quality contextual GUI datasets [515], which are specifically curated to enable these models to excel in GUI-specific tasks. Collecting such datasets, particularly those rich in GUI screenshots, metadata, and interactions, necessitates an elaborate process of data acquisition, filtering, and preprocessing, each requiring substantial effort and resources [516].

As GUI agents continue to gain traction, researchers have focused on assembling datasets that represent a broad spectrum of platforms and capture the diverse intricacies of GUI environments. These datasets are pivotal in training models that can generalize effectively, thanks to their coverage of varied interfaces, workflows, and user interactions. To ensure comprehensive representation, innovative methodologies have been employed to collect and structure these data assets. In the sections that follow, we detail an end-to-end pipeline for data collection and processing tailored to training GUI-specific LLMs. We also examine significant datasets from various platforms, providing insights into their unique features, the methodologies used in their creation, and their potential applications in advancing the field of LLM-brained GUI agents.

### 7.1 Data Collection

Data is pivotal in training a purpose-built GUI agent, yet gathering it requires substantial time and effort due to the task's complexity and the varied environments involved.

#### 7.1.1 Data Composition and Sources

The essential data components for GUI agent training closely align with the agent's perception and inference requirements discussed in Sections 5.2.2 and 5.4. At a high level, this data comprises:

1) User Instructions: These provide the task's overarching goal, purpose, and specific details, typically in natural language, offering a clear target for the agent to accomplish, e.g., "change the font size of all text to 12".

2) Environment Perception: This typically includes GUI screenshots, often with various visual augmentations, as well as optional supplementary data like widget trees and UI element properties to enrich the context. This should encompass both the static assessment of environment states (Section 5.2.2) and the dynamic environment feedback that captures post-action changes (Section 5.2.3), thereby providing sufficient contextual information.

3) Task Trajectory: This contains the critical action sequence required to accomplish the task, along with supplementary information, such as the agent's plan. A trajectory usually involves multiple steps and actions to navigate through the task.

While user instructions and environmental perception serve as the model's input, the expected model output is the task trajectory. This trajectory's action sequence is then grounded within the environment to complete the task.

For user instructions, it is crucial to ensure that they are realistic and reflective of actual user scenarios. Instructions can be sourced in several ways: (i) directly from human designers, who can provide insights based on real-world applications; (ii) extracted from existing, relevant datasets if suitable data is available; (iii) sourcing from public materials, such as websites, application help documentation, and other publicly available resources; and (iv) generated by LLMs, which can simulate a broad range of user requests across different contexts. Additionally, LLMs can be employed for data augmentation [517], increasing both the quality and diversity of instructions derived from the original data.

For gathering environment perception data, various toolkits—such as those discussed in Section 5.2.2 can be used to capture the required GUI data. This can be done within an environment emulator (e.g., Android Studio Emulator ${}^{29}$ Selenium WebDriver ${}^{30}$ Windows Sandbox ${}^{31}$ ) or by directly interfacing with a real environment to capture the state of GUI elements, including screenshots, widget trees, and other metadata essential for the agent's operation.

Collecting task trajectories, which represent the agent's action sequence to complete a task, is often the most challenging aspect. Task trajectories need to be accurate, executable, and well-validated. Collection methods include (i) using programmatically generated scripts, which define action sequences for predefined tasks, providing a highly controlled data source; (ii) employing human annotators, who complete tasks in a crowdsourced manner with each step recorded, allowing for rich, authentic action data; and (iii) leveraging model or agent bootstrapping [518], where an existing LLM or GUI agent attempts to complete the task and logs its actions, though this method may require additional validation due to potential inaccuracies. All these methods demand considerable effort, reflecting the complexities of gathering reliable, task-accurate data for training GUI agents.


<!-- Meanless: JOURNAL OF IATEX CLASS FILES, DECEMBER 2024<br>44 -->

<!-- Media -->

<!-- figureText: Existing<br>Public<br>State-Action Data Collection<br>Dataset Resources<br>LLM<br>Human<br>Instantiation Task: Change the font size of the third paragraph in a Word document draft.docx to 20.<br>Step 1: Select the third paragraph for editing.<br>Step 2: Click on the "Font Size" dropdown in the Word toolbar. Action: Click(Toolbar("Font Size"))<br>Step 3: From the dropdown menu, select "20" as the font size. Action: Click(Button("20"))<br>Step 4: Ensure the third paragraph's font size is now set to 20. Action: VerifyFontSize(Text, 20)<br>Action: SelectText(Paragraph(3))<br>\{Botton - Insert, ...\} \{Botton - Chart, ...\}<br>\{Botton - Insert, ...\} \{Botton - Chart, ...\}<br>\{Botton - Insert, ...\} \{Botton - Chart, ...\}<br>\{Botton - Insert, ...\}<br>\{Botton - Chart, ...\}<br>UI Properties<br>品品… UI Properties<br>UI Properties<br><icon/><br>UI Properties<br>Widget Tree<br>Widget Tree<br>Widget Tree<br>Widget Tree<br>Prototypical Instructions<br>Task Instantiation<br>Data Filtering Data Augmentation<br>Screenshot<br>Screenshot<br>Screenshot<br>Screenshot<br>Task Execution<br>(Evaluation)<br>1<br>1<br>GUI Environments<br>LLM<br>Human<br>GUI Agent<br>Human -->

<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2025/11/2025_11_04__12_39_04_3f908b.jpg"/>

Fig. 25: A complete pipeline for data collection for training a GUI agent model.

<!-- Media -->

#### 7.1.2 Collection Pipeline

Figure 25 presents a complete pipeline for data collection aimed at training a GUI agent model. The process begins with gathering initial user instructions, which may come from various aforementioned sources. These instructions are typically prototypical, not yet tailored or grounded to a specific environment [374]. For instance, an instruction like "how to change the font size?" from a general website lacks specificity and doesn't align with the concrete requests a user might make within a particular application. To address this, an instantiation step is required [374], where instructions are contextualized within a specific environment, making them more actionable. For example, the instruction might be refined to "Change the font size of the third paragraph in a Word document of draft.docx to 20.", giving it a clear, environment-specific goal. This instantiation process can be conducted either manually by humans or programmatically with an LLM.

Following instantiation, instructions may undergo a filtering step to remove low-quality data, ensuring only relevant and actionable instructions remain. Additionally, data augmentation techniques can be applied to expand and diversify the dataset, improving robustness. Both of these processes can involve human validation or leverage LLMs for efficiency.

Once instruction refinement is complete, task trajectories and environment perceptions are collected simultaneously. As actions are performed within the environment, each step is logged, providing a record of the environment's state and the specific actions taken. After a full task trajectory is recorded, an evaluation phase is necessary to identify and remove any failed or inaccurate sequences, preserving the quality of the dataset. By iterating this pipeline, a high-quality dataset of GUI agent data can be compiled, which is crucial for training optimized models.

In the following sections, we review existing GUI agent datasets across various platforms, offering insights into current practices and potential areas for improvement.

### 7.2 Web Agent Data

Web-based GUI agents demand datasets that capture the intricate complexity and diversity of real-world web interactions. These datasets often encompass varied website structures, including DOM trees and HTML content, as well as multi-step task annotations that reflect realistic user navigation and interaction patterns. Developing agents that can generalize across different websites and perform complex tasks requires comprehensive datasets that provide rich contextual information. We provide an overview of web-based GUI agents dataset in Table 19

Building upon this need, several significant datasets have been developed to advance web-based GUI agents. Unlike traditional datasets focusing on narrow, predefined tasks, Mind2Web [212] represents a significant step forward by emphasizing open-ended task descriptions, pushing agents to interpret high-level goals independently. It offers over 2,350 human-annotated tasks across 137 diverse websites, capturing complex interaction patterns and sequences typical in web navigation. This setup aids in evaluating agents' generalization across unseen domains and serves as a benchmark for language grounding in web-based GUIs, enhancing adaptability for real-world applications.

Similarly, WebVLN [347] expands on web GUI tasks by combining navigation with question-answering. It provides agents with text-based queries that guide them to locate relevant web pages and extract information. By leveraging both HTML and visual content from websites, WebVLN aligns with real-world challenges of web browsing. This dataset is particularly valuable for researchers aiming to develop agents capable of complex, human-like interactions in GUI-driven web spaces.

Moreover, WebLINX 348 focuses on conversational GUI agents, particularly emphasizing real-world web navigation through multi-turn dialogue. Featuring over 2,300 expert demonstrations across 155 real-world websites, WebLINX creates a rich environment with DOM trees and screenshots for training and evaluating agents capable of dynamic, user-guided navigation tasks. This dataset promotes agent generalization across new sites and tasks, with comprehensive action and dialogue data that provide insights into enhancing agent responsiveness in realistic web-based scenarios.

---

<!-- Footnote -->

29. https://developer.android.com/studio

30. https://www.selenium.dev/

31. https://learn.microsoft.com/en-us/windows/security/ application-security/application-isolation/windows-sandbox/

windows-sandbox-overview

<!-- Footnote -->

---


<!-- Meanless: 45 -->

MultiUI [220] is a large-scale dataset designed to enhance GUI agents' text-rich visual understanding. It comprises 7.3 million multimodal instruction samples collected from 1 million websites, covering key web UI tasks such as element grounding, action prediction, and interaction modeling. Unlike traditional datasets that rely on raw HTML, MultiUI utilizes structured accessibility trees to generate high-quality multimodal instructions. Models trained on MultiUI demonstrate substantial performance improvements, achieving a 48% gain on VisualWebBench [213] and a 19.1% increase in element accuracy on Mind2Web [212].

<!-- Media -->

JOURNAL OF LATEX CLASS FILES, DECEMBER 2024

TABLE 19: Overview of datasets for optimizing LLMs tailored for web GUI agents.

<table><tr><td>Dataset</td><td>Platform</td><td>Source</td><td>Content</td><td>Scale</td><td>Collection Method</td><td>Highlight</td><td>Link</td></tr><tr><td>Mind2Web 212]</td><td>Web</td><td>Crowdsourced</td><td>Task descriptions, action sequences, webpage snapshots</td><td>2,350 from websites 137</td><td>Human demonstrations</td><td>Develops generalist web agents with diverse user interactions on real-world websites</td><td>https: //osu-nlp-group. github.io/ Mind2Web/</td></tr><tr><td>Mind2Web-Live 349]</td><td>Web</td><td>Sampled and reannotated from the Mind2Web 212</td><td>Textual descriptions, intermediate evaluation states. action sequences, and metadata, GUI screenshots</td><td>542 tasks, with 4,550 detailed annotation steps.</td><td>Annotated by human experts.</td><td>Emphasis on dynamic evaluation using “key nodes”, which represent critical intermediate states in web tasks.</td><td>https: //huggingface. co/datasets/ iMeanAI/ Mind2Web-Live</td></tr><tr><td>WebVLN 347]</td><td>Web</td><td>Human-designed, LLM-generated</td><td>Text instructions, plans, GUI screen-shots, HTML content</td><td>8,990 navigation paths, 14,825 QA pairs</td><td>WebVLN simulator, LLM-generated QA pairs</td><td>Vision-and-language navigation for human-like web browsing</td><td>https://github.com/WebVLN/ WebVLN</td></tr><tr><td>WebLINX 348]</td><td>Web</td><td>From human experts</td><td>Conversational interactions, action sequences, DOM and screenshots</td><td>2,337 demonstrations with over 100,000 interactions</td><td>Annotated by human experts</td><td>The first large-scale dataset designed to evaluate agents in real-world conversational web navigation</td><td>https: //mcgill-nlp. github.io/ weblinx/</td></tr><tr><td>AgentTrek 350]</td><td>Web</td><td>Web tutorials</td><td>Task metadata. step-by-step instructions, action sequences, visual observations, reproducible native traces</td><td>4,902 trajectories</td><td>VLM agent guided by tutorials, with Playwright capturing the traces</td><td>Synthesizes high-quality trajectory data by leveraging web tutorials</td><td>/</td></tr><tr><td>MultiUI 220]</td><td>Web</td><td>Combination of human-designed instructions and automated extraction from web structures</td><td>Textual task descriptions, plans, action sequences. GUI screenshots. accessibility trees, bounding box annotations</td><td>7.3 million instruction samples from 1 million websites</td><td>LLMs and Playwright</td><td>Supports a broad range of UI-related tasks, including GUI understanding, action prediction, and element grounding.</td><td>https: //neulab.github. io/MultiUI/</td></tr><tr><td>Explorer 290]</td><td>Web</td><td>Popular URLs with systematic web exploration by LLMs</td><td>Textual task descriptions. Action sequences, GUI screenshots, Accessibility trees, HTML content</td><td>94K successful web trajectories. 49K unique URLs, 720K screenshots</td><td>Generated by a multi-agent LLM pipeline</td><td>Largest-scale web trajectory dataset to date; dynamically explores web pages to create contextually relevant tasks</td><td>/</td></tr><tr><td>InSTA [351]</td><td>Web</td><td>Automatically generated by LLMs across 1M websites from Common Crawl</td><td>Web navigation tasks in natural language, task plans and action sequences, HTML-based observations converted to markdown, and evaluations from LLM-based judges</td><td>150,000 tasks across 150,000 websites</td><td>Generated by LLMs using the Playwright API and filtered by LLM-based judges</td><td>Presents fully automated three-stage data generation pipeline—task generation, action execution, and evaluation—using only language models without any human annotations</td><td>https:// data-for-agents github.io</td></tr></table>

<!-- Media -->

InSTA [351] is an Internet-scale dataset for training GUI-based web agents, generated entirely through an automated LLM pipeline without human annotations. It covers 150k diverse websites sourced from Common Crawl and includes rich web navigation tasks, trajectories in Playwright API calls, and evaluations using LLM-based judges. The dataset highlights strong generalization capabilities and data efficiency, significantly outperforming human-collected datasets like Mind2Web [212] and WebLINX 348 in zero-shot and low-resource settings. InSTA represents a key advancement in scalable data curation for LLM-powered GUI agents, offering unprecedented coverage across real-world web interfaces.

Collectively, these datasets represent essential resources that enable advancements in web agent capabilities, supporting the development of adaptable and intelligent agents for diverse web applications.


<!-- Meanless: JOURNAL OF IATEX CLASS FILES, DECEMBER 2024<br>46 -->

### 7.3 Mobile Agent Data

Mobile platforms are critical for GUI agents due to the diverse range of apps and unique user interactions they involve. To develop agents that can effectively navigate and interact with mobile interfaces, datasets must offer a mix of single and multi-step tasks, focusing on natural language instructions, UI layouts, and user interactions. We first overview mobile GUI agents dataset in Tables 20 and 21.

An early and foundational contribution in this domain is the Rico dataset [355], which provides over 72,000 unique UI screens and 10,811 user interaction traces from more than 9,700 Android apps. Rico has been instrumental for tasks such as UI layout similarity, interaction modeling, and perceptual modeling, laying the groundwork for mobile interface understanding and GUI agent development.

Building upon the need for grounding natural language instructions to mobile UI actions, PIXELHELP [144] introduces a dataset specifically designed for this purpose. It includes multi-step instructions, screenshots, and structured UI element data, enabling detailed analysis of how verbal instructions can be converted into mobile actions. This dataset has significant applications in accessibility and task automation, supporting agents that autonomously execute tasks based on verbal cues.

Further expanding the scope, the Android in the Wild (AITW) dataset [358] offers one of the most extensive collections of natural device interactions. Covering a broad spectrum of Android applications and diverse UI states, AITW captures multi-step tasks emulating real-world device usage. Collected through interactions with Android emulators, it includes both screenshots and action sequences, making it ideal for developing GUI agents that navigate app interfaces without relying on app-specific APIs. Due to its scale and diversity, AITW has become a widely used standard in the field.

In addition, META-GUI 357 provides a unique dataset for mobile task-oriented dialogue systems by enabling direct interaction with mobile GUIs, bypassing the need for API-based controls. This approach allows agents to interact across various mobile applications using multi-turn dialogues and GUI traces, broadening their capabilities in real-world applications without custom API dependencies. The dataset's support for complex interactions and multi-turn dialogue scenarios makes it valuable for building robust conversational agents.

Recently, MobileViews 364 emerged as the largest mobile screen dataset to date, offering over 600,000 screenshot-view hierarchy pairs from 20,000 Android apps. Collected with an LLM-enhanced app traversal tool, it provides a high-fidelity resource for mobile GUI agents in tasks such as screen summarization, tappability prediction, and UI component identification. Its scale and comprehensive coverage of screen states make MobileViews a key resource for advancing mobile GUI agent capabilities.

Collectively, mobile platforms currently boast the richest set of datasets due to their versatile tools, emulator support, and diverse use cases, reflecting the demand for high-quality, adaptive GUI agents in mobile applications.

### 7.4 Computer Agent Data

In contrast to mobile and web platforms, the desktop domain for GUI agents has relatively fewer dedicated datasets, despite its critical importance for applications like productivity tools and enterprise software. However, notable efforts have been made to support the development and evaluation of agents designed for complex, multi-step desktop tasks. We show related dataset for computer GUI agents in Table 22

A significant contribution in this area is ScreenA-gent [366], a dedicated dataset and model designed to facilitate GUI control in Linux and Windows desktop environments. ScreenAgent provides a comprehensive pipeline that enables agents to perform multi-step task execution autonomously, encompassing planning, action, and reflection phases. By leveraging annotated screenshots and detailed action sequences, it allows for high precision in UI element positioning and task completion, surpassing previous models in accuracy. This dataset is invaluable for researchers aiming to advance GUI agent capabilities in the desktop domain, enhancing agents' decision-making accuracy and user interface interactions.

The LAM [367] is specifically designed to train and evaluate Large Action Models (LAMs) for GUI environments, bridging natural language task understanding and action execution. It comprises two core components: Task-Plan data, detailing user tasks with step-by-step plans, and Task-Action data, translating these plans into executable GUI actions. Sourced from application documentation, WikiHow articles, and Bing search queries, the dataset is enriched and structured using GPT-4. Targeting the Windows OS, with a focus on automating tasks in Microsoft Word, it includes 76,672 task-plan pairs and 2,688 task-action trajectories, making it one of the largest collections for GUI-based action learning. Data quality is ensured through a robust validation pipeline that combines LLM-based instantiation, GUI interaction testing, and manual review. Each entry is complemented with GUI screenshots and metadata, enabling models to learn both high-level task planning and low-level execution. The dataset's modular design supports fine-tuning for specific GUI tasks and serves as a replicable framework for building datasets in other environments, marking a significant contribution to advancing GUI-based automation.

Although the desktop domain has fewer datasets compared to mobile and web, efforts like ScreenAgent and LAMs highlight the growing interest and potential for developing sophisticated GUI agents for computer systems.

### 7.5 Cross-Platform Agent Data

Cross-platform datasets play a pivotal role in developing versatile GUI agents that can operate seamlessly across mobile, computer, and web environments. Such datasets support generalizability and adaptability, enabling agents to handle varied interfaces and tasks in real-world applications. We provide an overview of related dataset for cross-platform GUI agents in Table 23 and 24.

One significant contribution is ScreenAI [375], which extends the scope of data collection to include both mobile and desktop interfaces. Covering tasks such as screen annotation, question-answering, and navigation, ScreenAI offers hundreds of millions of annotated samples. Its comprehensive scale and mixed-platform coverage make it a robust foundation for GUI agents that need to manage complex layouts and interactions across diverse interfaces. By emphasizing element recognition and screen summarization, ScreenAI advances the development of multi-platform GUI agents capable of handling varied visual structures.


<!-- Meanless: 47 -->

<!-- Media -->

JOURNAL OF LATEX CLASS FILES, DECEMBER 2024

TABLE 20: Overview of datasets for optimizing LLMs tailored for mobile GUI agents (Part I).

<table><tr><td>Dataset</td><td>Platform</td><td>Source</td><td>Content</td><td>Scale</td><td>Collection Method</td><td>Highlight</td><td>Link</td></tr><tr><td>VGA [354]</td><td>Android Mobile</td><td>Rico 355</td><td>GUI screenshots, task descriptions, action sequences, bounds, layout, and functions of GUI elements</td><td>63.8k instances, 22.3k instruction-following data pairs, 41.4k conversation data pairs</td><td>Generated by GPT-4 models</td><td>Prioritizes visual content to reduce inaccuracies</td><td>https: //github.com/ Linziyang1999/ Vision% 2DGUI% 2Dassistant</td></tr><tr><td>Rico 355]</td><td>Android Mobile</td><td>Gathered from real Android apps on Google Play Store</td><td>Textual data, screenshots. action sequences, UI structure, annotated l representations</td><td>72,219 unique UI screens, 10,811 user interaction traces</td><td>automated exploration</td><td>Crowdsourcing, Comprehensive dataset for mobile UI design, interaction modeling, layout generation</td><td>http://www.interactionmining org/</td></tr><tr><td>PixelHelp 144]</td><td>Android Mobile</td><td>Human, web "How-to", Rico UI corpus synthetic</td><td>Natural language instructions, action sequences, GUI screenshots. structured UI data</td><td>187 multi-step instructions, 295,476 synthetic singlestep commands</td><td>Human annotation and synthetic generation</td><td>Pioneering method for grounding natural language instructions to executable mobile UI actions</td><td>https: //github.com/ google-research google-research tree/master/ seg2act</td></tr><tr><td>MoTIF 356]</td><td>Android Mobile</td><td>Human-written</td><td>Natural language instructions, action sequences, GUI screenshots. structured UI data</td><td>6,100 tasks across 125 Android apps</td><td>Human annotation</td><td>Task feasibility prediction for interactive GUI in mobile apps</td><td>https: //github.com/ aburns4/MoTIF</td></tr><tr><td>META-GUI 357]</td><td>Android Mobile</td><td>SMCalFlow 519</td><td>Dialogues, action sequences, screenshots. Android view hierarchies</td><td>1,125 dialogues and 4,684 turns</td><td>Human annotation</td><td>Task-oriented dialogue system for mobile GUI without relying on back-end APIs</td><td>https: //x-lance.github. io/META-GU</td></tr><tr><td>AITW 358</td><td>Android Mobile</td><td>Human-generated instructions. LLM-generated prompts</td><td>Natural language instructions, screenshots. observation-action pairs</td><td>715,142 episodes and 30,378 unique instructions</td><td>Human raters using Android emulators</td><td>Large-scale dataset for device control research with extensive app and UI diversity</td><td>https: //github.com/ google-research google-research tree/master/ android in the wild</td></tr><tr><td>GUI-Xplore 365]</td><td>Mobile Android</td><td>Combination of automated exploration and manual design</td><td>Exploration videos. textual tasks, QA pairs, view hierarchies, GUI screen-shots, action sequences, and GUI transition graphs</td><td>312 apps, 115 hours of video, 32,569 QA pairs, 41,293 actions, about 200 pages per app</td><td>Automated and human exploration</td><td>Introduces an exploration-based pretraining paradigm that provides rich app-specific priors through video data</td><td>https: //github.com/ 921112343/ GUI-Xplore</td></tr></table>

<!-- Media -->

Building upon the need for evaluating visual foundation models across environments, VisualAgentBench [374] is a groundbreaking cross-platform benchmark designed to assess GUI agents in both mobile and web settings. It emphasizes interaction-focused tasks, using environments like Android Virtual Device and WebArena-Lite [412] to evaluate and improve agent responses to GUI layouts and user interface actions. The dataset's innovative collection method, which combines program-based solvers and large multimodal model bootstrapping, facilitates robust training trajectories that enhance adaptability and error recovery in GUI agent tasks.

Furthermore, GUI-World [371] spans multiple platforms, including desktop, mobile, and XR environments, with over 12,000 annotated videos. Designed to address the challenges of dynamic and sequential GUI tasks, GUI-World allows researchers to benchmark GUI agent capabilities across diverse interfaces. By providing detailed action sequences and QA pairs, it sets a high standard for evaluating agents in complex, real-world scenarios.

Additionally, xLAM [372] contributes significantly to actionable agent development by providing a unified dataset format designed to support multi-turn interactions, reasoning, and function-calling tasks. Sourced from datasets like WebShop 425], ToolBench [520], and AgentBoard [521], xLAM standardizes data formats across diverse environments, addressing the common issue of inconsistent data structures that hinder agent training and cross-environment compatibility. By offering a consistent structure, xLAM enhances the adaptability and error detection capabilities of GUI agents, allowing for more seamless integration and performance across different applications.

OS-Genesis [369] adopts a reverse task synthesis approach for the Android and web platforms. It leverages GPT- 4o to interactively explore the environment and generate instructions in a reverse manner. This process constructs high-quality, diverse GUI trajectories without relying on human annotations or predefined tasks. By eliminating these dependencies, OS-Genesis achieves scalable and efficient training for GUI agents while significantly enhancing the diversity and quality of the generated data.


<!-- Meanless: 48 -->

<!-- Media -->

JOURNAL OF LATEX CLASS FILES, DECEMBER 2024

TABLE 21: Overview of datasets for optimizing LLMs tailored for mobile GUI agents (Part II).

<table><tr><td>Dataset</td><td>Platform</td><td>Source</td><td>Content</td><td>Scale</td><td>Collection Method</td><td>Highlight</td><td>Link</td></tr><tr><td>GUI Odyssey 359</td><td>Android Mobile</td><td>Human designers, GPT-4</td><td>Textual tasks, plans, action sequences, GUI screenshots</td><td>7,735 episodes across 201 apps</td><td>Human demonstrations</td><td>Focuses on cross-app navigation tasks on mobile devices</td><td>https: //github.com/ OpenGVLab/ GUI-Odyssey</td></tr><tr><td>Amex 360</td><td>Android Mobile</td><td>Human-designed, ChatGPT-generated</td><td>Text tasks, action sequences, high-res screenshots with multi-level annotations</td><td>104,000 screen-shots, 1.6 million interactive elements, 2,946 instructions</td><td>Human annotations, autonomous scripts</td><td>Multi-level, large-scale annotations supporting complex mobile GUI tasks</td><td>https: //yuxiangchai. github.io/ AMEX/</td></tr><tr><td>Ferret-UI 352]</td><td>iOS, Android Mobile</td><td>Spotlight dataset, GPT-4</td><td>Text tasks, action plans, GUI element annotations, bounding boxes</td><td>40,000 elementary tasks, 10,000 advanced tasks</td><td>GPT-generated</td><td>Benchmark for UI-centric tasks with adjustable screen aspect ratios</td><td>https: //github.com/ apple/ml-ferret</td></tr><tr><td>AITZ [298]</td><td>Android Mobile</td><td>AITW 358</td><td>Screen-action pairs, action descriptions</td><td>18,643 screen-action pairs across 70+ apps, episodes</td><td>GPT- 4V, icon detection models</td><td>Structured "Chain-of-Action-Thought" enhancing GUI navigation</td><td>https: //github.com/ IMNearth/CoAT</td></tr><tr><td>Octo-planner [361]</td><td>Android Mobile</td><td>GPT-4 generated</td><td>Text tasks, decomposed plans, action sequences</td><td>1,000 points</td><td>GPT-4 generated</td><td>Optimized for task planning with GUI actions</td><td>https: //huggingface. co/NexaAIDev/ octopus-planning</td></tr><tr><td>E-ANT 362]</td><td>Android tiny-apps</td><td>Human behaviors</td><td>Task descriptions, screenshots. action sequences. page element data</td><td>40,000+ traces, 10,000 action intents</td><td>Human annotation</td><td>First large-scale Chinese dataset for GUI navigation with real human interactions</td><td>/</td></tr><tr><td>Mobile3M 353]</td><td>Android Mobile</td><td>Real-world interactions, simulations</td><td>UI screenshots. XML documents. action sequences</td><td>3,098,786 pages, 20,138,332 actions</td><td>Simulation algorithm</td><td>Large-scale Chinese mobile GUI dataset with unique navigation graph</td><td>https: //github.com/ Meituan-AutoMD MobileVLM</td></tr><tr><td>AndroidLab 363]</td><td>Android Mobile</td><td>Human design, LLM self-exploration. academic datasets</td><td>Text instructions, action sequences, XML data, screen-shots</td><td>10.5k traces. 94.3k steps</td><td>Human annotation, LLM self-exploration</td><td>XML-based interaction data with unified action space</td><td>https://github.com/THUDM/ Android-Lab</td></tr><tr><td>MobileViews 364]</td><td>Android Mobile</td><td>LLM-enhanced app traversal tool</td><td>Screenshot-view hierarchy pairs</td><td>600,000 screenshots. VH pairs from 20,000+ apps</td><td>LLM-enhanced crawler</td><td>Largest open-source mobile screen dataset</td><td>https: //huggingface. co/datasets/ mllmTeam/ MobileViews</td></tr><tr><td>FedMABench 452</td><td>Android Mobile</td><td>AndroidControl 515], AITW 358</td><td>Textual task descriptions. action sequences. and GUI screenshots</td><td>6 dataset series with over 30 subsets</td><td>Inferred from existing Android datasets</td><td>The first dataset designed to benchmark federated mobile GUI agents</td><td>https://github.com/wwh0411/ FedMABench</td></tr></table>

<!-- Media -->

Collectively, these cross-platform datasets contribute to building multi-platform GUI agents, paving the way for agents that can seamlessly navigate and perform tasks across different interfaces, fostering more generalized and adaptable systems.

### 7.6 Takeaways

Data collection and curation for LLM-powered GUI agents is an intensive process, often requiring substantial human involvement, particularly for generating accurate action sequences and annotations. While early datasets were limited in scale and task diversity, recent advancements have led to large-scale, multi-platform datasets that support more complex and realistic GUI interactions. Key insights from these developments include:

1) Scale and Diversity: High-quality, large-scale data is essential for training robust GUI agents capable of handling diverse UI states and tasks. Datasets like MobileViews 364 and ScreenAI 375 illustrate the importance of vast and varied data to accommodate the dynamic nature of mobile and desktop applications, enhancing the agent's resilience across different environments.

2) Cross-Platform Flexibility: Cross-platform datasets such as VisualAgentBench 374 and GUI-World 371 underscore the value of generalizability, enabling agents to perform consistently across mobile, web, and desktop environments. This cross-platform adaptability is a crucial step towards creating one-stop solutions where a single GUI agent can operate seamlessly across multiple platforms.

3) Automated Data Collection: Al-driven data collection tools, as exemplified by OmniParser [184] and Mo-bileViews 364, showcase the potential to significantly reduce manual efforts and accelerate scalable dataset creation. By automating the annotation process, these tools pave the way for more efficient data pipelines, moving towards a future where AI supports AI by expediting data gathering and labeling for complex GUI interactions.


<!-- Meanless: 49 -->

JOURNAL OF IATEX CLASS FILES, DECEMBER 2024

<!-- Media -->

TABLE 22: Overview of datasets for optimizing LLMs tailored for computer GUI agents.

<table><tr><td>Dataset</td><td>Platform</td><td>Source</td><td>Content</td><td>Scale</td><td>Collection Method</td><td>Highlight</td><td>Link</td></tr><tr><td>ScreenAgent 366]</td><td>Linux, Windows OS</td><td>Human-designed</td><td>GUI screenshots, action sequences</td><td>273 task sessions. 3,005 training screenshots, 898 test screenshots</td><td>Human annotation</td><td>VLM-based agent across multiple desktop environments</td><td>https: //github.com/ niuzaisheng/ ScreenAgent</td></tr><tr><td>LAM 367]</td><td>Windows OS</td><td>Application documentation, WikiHow articles, Bing search queries</td><td>Task descriptions in natural language, step-by-step plans, action sequences, GUI screenshots</td><td>76,672 task-plan pairs, 2,192 task-action trajectories</td><td>Instantiated using GPT- 4, with actions tested and validated in the Windows environment using UFO 19</td><td>Provides structured pipeline for collecting validating, and augmenting data, enabling high-quality training for action-oriented AI models.</td><td>https://github.com/microsoft/ UFO/tree/main/ dataflow</td></tr><tr><td>DeskVision 368]</td><td>Windows macOS, and Linux desktops</td><td>Internet</td><td>GUI screenshots with annotated bounding boxes for UI elements and detailed region captions</td><td>54,855 screenshots with 303,622 UI element annotations</td><td>UI elements detected using OmniParser and PaddleOCR</td><td>The first large-scale, open-source dataset focusing on real-world desktop GUI scenarios across operating systems</td><td>/</td></tr></table>

TABLE 23: Overview of datasets for optimizing LLMs tailored for cross-platform GUI agents (Part I).

<table><tr><td>Dataset</td><td>Platform</td><td>Source</td><td>Content</td><td>Scale</td><td>Collection Method</td><td>Highlight</td><td>Link</td></tr><tr><td>Visual-AgentBench [374]</td><td>Android Mobile, Web</td><td>VAB-Mobile: Android Virtual Device, VAB-WebArena-Lite: WebArena 413]</td><td>Task instructions, action sequences, screen observations</td><td>VAB-Mobile: 1,213 trajectories, 10,175 steps; VAB-WebArena-Lite: 1,186 trajectories, 9,522 steps</td><td>Program-based solvers. agent bootstrapping, human demonstrations</td><td>Systematic evaluation of VLM as a visual foundation agent across multiple scenarios</td><td>https://github.com/THUDM/ VisualAgentBench</td></tr><tr><td>GUICourse 219]</td><td>Android Mobile, Web</td><td>Web scraping, simulation, manual design</td><td>GUI screenshots, action sequences, OCR tasks, QA pairs</td><td>10 million website page-annotation pairs, 67,000 action instructions</td><td>LLM-based auto-annotation, crowd-sourcing</td><td>Dataset suite for enhancing VLM GUI navigation on web and mobile platforms</td><td>https://github.com/yiye3/ GUICourse</td></tr><tr><td>GUI-World [371]</td><td>OS, Mobile, Web, XR</td><td>Student workers, YouTube instructional videos</td><td>GUI videos, human-annotated keyframes, captions, data, action sequences</td><td>12,000 videos, 83,176 frames</td><td>Human annotation</td><td>Designed dynamic, sequential GUI tasks with video data</td><td>https: //gui-world. github.io/</td></tr><tr><td>ScreenAl 375]</td><td>Android, iOS, Desktop/Web</td><td>Crawling apps and webpages, synthetic QA</td><td>Screen annotation, screen QA, navigation, summarization</td><td>Annotation: hundreds of millions; QA: tens of millions; Navigation: millions</td><td>Model, human annotation</td><td>Comprehensive pretraining and fine-tuning for GUI tasks across platforms</td><td>https://github.com/google% 2Dresearch% 2Ddatasets/ screen annotation</td></tr><tr><td>OmniParser 184]</td><td>Web, Desktop, Mobile</td><td>Popular webpages</td><td>UI screenshots, bounding boxes, icon descriptions, OCR-derived text</td><td>67,000+ screenshots. 7,000 icon-description pairs</td><td>Finetuned detection model, OCR, human descriptions</td><td>Vision-based parsing of UI screen-shots into structured elements</td><td>https://github.com/microsoft/ OmniParser</td></tr></table>

<!-- Media -->

4) Unified Data Formats and Protocols: xLAM's unified data format is an essential innovation that improves compatibility across diverse platforms [372], addressing a significant bottleneck in cross-platform GUI agent development. Establishing standardized protocols or action spaces for data collection, particularly given the varied data formats, action spaces, and environment representations across platforms, will be vital in furthering agent generalization and consistency.

In summary, the evolving landscape of datasets for LLM-powered GUI agents spans multiple platforms, with each dataset addressing unique challenges and requirements specific to its environment. These foundational resources are key to enabling agents to understand complex UIs, perform nuanced interactions, and improve generalization across diverse applications. The push towards cross-platform adaptability, automated data collection, and standardized data formats will continue to shape the future of GUI agents.


<!-- Meanless: 50 -->

<!-- Media -->

JOURNAL OF LATEX CLASS FILES, DECEMBER 2024

TABLE 24: Overview of datasets for optimizing LLMs tailored for cross-platform GUI agents (Part II).

<table><tr><td>Dataset</td><td>Platform</td><td>Source</td><td>Content</td><td>Scale</td><td>Collection Method</td><td>Highlight</td><td>Link</td></tr><tr><td>Web-Hybrid 216]</td><td>Web, Android Mobile</td><td>Web-synthetic data</td><td>Screenshots, text-based referring expressions, coordinates on GUIs</td><td>10 million GUI elements, 1.3 million screenshots</td><td>Rule-based synthesis, LLMs for referring expressions</td><td>Largest dataset for GUI visual grounding</td><td>https: //osu-nlp-group. github.io/ UGround/</td></tr><tr><td>GUIDE 370]</td><td>Computer and Web</td><td>Direct submissions from businesses and survey responses</td><td>Task descriptions, GUI screenshots. action sequences, CoT reasoning, spatial grounding</td><td>N/A</td><td>Collected through NEXTAG, an automated annotation tool</td><td>Integrates images, action sequences, task descriptions, and spatial grounding into a unified dataset</td><td>https://github.com/superagi/ GUIDE</td></tr><tr><td>xLAM 372]</td><td>Web and tools used</td><td>Synthesized data, and existing dataset</td><td>Textual tasks, action sequences, function-calling data</td><td>60,000 data points</td><td>Collected using AI models with human verification steps</td><td>Provides a unified format across diverse environments, enhancing generalizability and error detection for GUI agents</td><td>https: //github.com/ SalesforceAlResearch xLAM</td></tr><tr><td>Insight-UI 373]</td><td>iOS, Android, Windows, Linux, Web</td><td>Common Crawl corpus</td><td>Textual tasks, plans, action sequences, GUI screenshots</td><td>434,000 episodes, 1,456,000 images</td><td>Automatic simulations performed by a browser API</td><td>Instruction-free paradigm and entirely auto-generated</td><td>/</td></tr><tr><td>OS-Genesis 369]</td><td>Web and Android</td><td>Reverse tas synthesis, where the GUI environment is explored interactively without predefined tasks or human annotations.</td><td>High-level instructions, low-level instructions, action sequences, and environment states.</td><td>1,000 synthesized trajectories.</td><td>Model-based interaction-driven approach with GPT- 40.</td><td>Reverses the conventional task-driven collection process by enabling exploration-first trajectory synthesis.</td><td>https: //giushisun. github.io/ OS-Genesis-Home/</td></tr><tr><td>Navi-plus 377]</td><td>Web and Android</td><td>AndroidControl 515] and Mind2Web 212]</td><td>Task descriptions, GUI action trajectories, low-level step instructions, screenshots, and followup ASK/SAY interaction pairs</td><td>/</td><td>LLM-automated with human validation</td><td>Introduces a Self-Correction GUI Navigation task featuring the novel ASK action for recovering missing information</td><td>/</td></tr><tr><td>Explorer <br> 376]</td><td>Web and Android</td><td>Automated traversal of real websites and Android apps</td><td>UI screenshots, bounding boxes of interactable elements, screen similarity labels, and user actions</td><td>KhanAcademy (Web): 2,841 interactables, 378 screen similarity samples; Spotify (Android): 1,207 interactables, 451 screen similarity samples</td><td>Automated tools, HTML parsing, Accessibility Tree</td><td>Platform-independent, supports auto-labeling, and enables trace recording and voice-controlled GUI navigation</td><td>https://github.com/varnelis/ Explorer</td></tr></table>

<!-- Media -->

## 8 Models for Optimizing LLM-Brained GUI AGENTS

LLMs act as the "brain" of GUI agents, empowering them to interpret user intents, comprehend GUI screens, and execute actions that directly impact their environments. While several existing foundation models are robust enough to serve as this core, they can be further fine-tuned and optimized to evolve into Large Action Models (LAMs)-specialized models tailored to improve the performance and efficiency of GUI agents. These LAMs bridge the gap between general-purpose capabilities and the specific demands of GUI-based interactions.

In this section, we first introduce the foundation models that currently form the backbone of GUI agents, highlighting their strengths and limitations. We then delve into the concept of LAMs, discussing how these models are fine-tuned with GUI-specific datasets to enhance their adaptability, accuracy, and action-orientation in GUI environments. Through this exploration, we illustrate the progression from general-purpose LLMs to purpose-built LAMs, laying the foundation for advanced, intelligent GUI agents.


<!-- Meanless: JOURNAL OF LATEX CLASS FILES, DECEMBER 2024<br>51 -->

### 8.1 Foundation Models

Foundation models serve as the core of LLM-powered GUI agents, providing the essential capabilities for understanding and interacting with graphical user interfaces. Recent advancements in both close-source and open-source MLLMs have significantly enhanced the potential of GUI agents, offering improvements in efficiency, scalability, and multimodal reasoning. This subsection explores these foundation models, highlighting their innovations, contributions, and suitability for GUI agent applications. For a quick reference, Table 25 presents an overview of the key models and their characteristics.

#### 8.1.1 Close-Source Models

While proprietary models are not openly available for customization, they offer powerful capabilities that can be directly utilized as the "brain" of GUI agents.

Among these, GPT-4V [378] and GPT-4o [93] are most commonly used in existing GUI agent frameworks due to their strong abilities, as discussed in Section 6. GPT- 4V represents a significant advancement in multimodal AI, combining text and image analysis to expand the functionality of traditional LLMs. Its ability to understand and generate responses based on both textual and visual inputs makes it well-suited for GUI agent tasks that require deep multimodal reasoning. Although its deployment is limited due to safety and ethical considerations, GPT-4V underscores the potential of foundation models to revolutionize GUI agent development with enhanced efficiency and flexibility.

Similarly, GPT-4o offers a unified multimodal autoregressive architecture capable of processing text, audio, images, and video. This model excels in generating diverse outputs efficiently, achieving faster response times at lower costs compared to its predecessors. Its rigorous safety and alignment practices make it reliable for sensitive tasks, positioning it as a robust tool for intelligent GUI agents that require comprehensive multimodal comprehension.

The Gemini model family [92] advances multimodal AI modeling by offering versions tailored for high-complexity tasks, scalable performance, and on-device efficiency. Notably, the Nano models demonstrate significant capability in reasoning and coding tasks despite their small size, making them suitable for resource-constrained devices. Gemini's versatility and efficiency make it a compelling choice for powering GUI agents that require both performance and adaptability.

Emphasizing industry investment in GUI automation, Claude 3.5 Sonnet (Computer Use) introduces a pioneering approach by utilizing a vision-only paradigm for desktop task automation [163], [164]. It leverages real-time screenshots to observe the GUI state and generate actions, eliminating the need for metadata or underlying GUI structure. This model effectively automates GUI tasks by interpreting the screen, moving the cursor, clicking buttons, and typing text. Its unique architecture integrates a ReAct-based [252] reasoning paradigm with selective observation, reducing computational overhead by observing the environment only when necessary. Additionally, Claude 3.5 maintains a history of GUI screenshots, enhancing task adaptability and enabling dynamic interaction with software environments in a humanlike manner. Despite challenges in handling dynamic interfaces and error recovery, this model represents a significant step forward in creating general-purpose GUI agents. Its development highlights substantial industry investment in this area, indicating a growing focus on leveraging LLMs for advanced GUI automation.

The Operator model [165], [513], developed by OpenAI, represents a new frontier in Computer-Using Agents (CUA), akin to Claude 3.5 Sonnet (Computer Use). Designed to interact with GUI environments through LLM-powered reasoning and vision capabilities, Operator builds upon GPT-40, integrating reinforcement learning to navigate and execute tasks across digital interfaces such as browsers, forms, and applications. By perceiving screenshots, interpreting UI elements, and performing actions via a virtual cursor and keyboard, Operator enables the automation of complex GUI-based workflows, including online purchases, email management, and document editing. Notably, Operator excels in understanding and manipulating digital environments, establishing itself as a powerful tool for human-computer interaction automation. Its exceptional performance on various benchmarks underscores its leading capabilities in GUI-based task automation.

#### 8.1.2 Open-Source Models

Open-source models provide flexibility for customization and optimization, allowing developers to tailor GUI agents with contextual data and deploy them on devices with limited resources.

The Qwen-VL series [210] is notable for its fine-grained visual understanding and multimodal capabilities. With a Vision Transformer-based visual encoder and the Qwen-7B language model [511], it achieves state-of-the-art results on vision-language benchmarks while supporting multilingual interactions. Its efficiency and open-source availability, along with quantized versions for resource efficiency, make it suitable for developing GUI agents that require precise visual comprehension.

Building upon this, Qwen2-VL [231] introduces innovations like Naive Dynamic Resolution and Multimodal Rotary Position Embedding, enabling efficient processing of diverse modalities including extended-length videos. The scalable versions of Qwen2-VL balance computational efficiency and performance, making them adaptable for both on-device applications and complex multimodal tasks in GUI environments.

InternVL-2 [379], [380] combines a Vision Transformer with a Large Language Model to handle text, images, video, and medical data inputs. Its progressive alignment strategy and availability in various sizes allow for flexibility in deployment. By achieving state-of-the-art performance in complex multimodal tasks, InternVL-2 demonstrates powerful capabilities that are valuable for GUI agents requiring comprehensive multimodal understanding.

Advancing efficient integration of visual and linguistic information, CogVLM [381] excels in cross-modal tasks with a relatively small number of trainable parameters. Its ability to deeply integrate visual and language features while preserving the full capabilities of large language models makes it a cornerstone for GUI agent development, especially in applications where resource efficiency is critical.


<!-- Meanless: JOURNAL OF LATEX CLASS FILES, DECEMBER 2024<br>52 -->

<!-- Media -->

TABLE 25: Overview of foundation models for LLM-brained GUI agents.

<table><tr><td>Model</td><td>Modality</td><td>Model Size</td><td>Architecture</td><td>Training Methods</td><td>Highlights</td><td>Open-Source</td><td>Link</td></tr><tr><td></td><td>Text, audio, image, and video</td><td>-</td><td>Multimodal autoregressive architecture</td><td>Pre-trained on a mix of public data, further trained for align ment with human preferences and safety considerations</td><td>Unified multimodal architecture that seamlessly processes and generates outputs across text audio, image, and video, offering faster and more cost-effective operation than its predecessors</td><td>No</td><td>/</td></tr><tr><td>GPT-4V 378]</td><td>Text and image</td><td>-</td><td>-</td><td>Pre-trained on a large dataset of text and image data, followed by fine-tuning with reinforcement learning from human feedback (RLHF)</td><td>Notable for its multimodal ca pabilities, allowing it to analyze and understand images along side text</td><td>No</td><td>/</td></tr><tr><td>Gemini 9</td><td>Text, image, audio, and video</td><td>Nano versions: 1.8B/3.25B</td><td>Enhanced Transformer decoders</td><td>Large-scale pre-training on multimodal data, followed by supervised fine-tuning, reward modeling, and RLHF</td><td>Achieves state-of-the-art performance across multimodal tasks. including a groundbreaking 90% on the MMLU benchmark, and demonstrates capacity for on-device deployment with small model sizes</td><td>No</td><td>/</td></tr><tr><td>Claude 3.5 Sonnet (Computer Use) 163. 164]</td><td>Text and image</td><td>-</td><td>ReAct-based reasoning</td><td>-</td><td>Pioneering role in GUI automation as the first public beta model to utilize a vision-only paradigm for desktop task automation</td><td>No</td><td>/</td></tr><tr><td>Operator [165], [513]</td><td>Text and Image</td><td>-</td><td>Built on GPT-4o</td><td>Supervised learning and reinforcement learning</td><td>Trained to use a computer like a human, achieving remarkable performance on benchmarks</td><td>No</td><td>/</td></tr><tr><td>Qwen-VL 210]</td><td>Text and image</td><td>9.6B</td><td>A Vision Transformer (ViT) 522] as the visual encoder, with a large language model based on the Qwen-7B architecture</td><td>Two stages of pre-training and a final stage of instruction fine-tuning</td><td>Achieves state-of-the-art per formance on vision-language benchmarks and supports fine-grained visual understanding</td><td>Yes</td><td>httpss://github. com/QwenLM/ Qwen-VL</td></tr><tr><td>Qwen2-VL 231]</td><td>Text, image, and video</td><td>2B/7B/72B</td><td>ViT [522] as the vision encoder, paired with the Qwen2 series of language models</td><td>The ViT is trained with image text pairs; all parameters are unfrozen for broader multimodal learning with various datasets; fine-tuning the LLM on instruction datasets</td><td>Introduces Naive Dynamic Res olution for variable resolution image processing and Multimodal Rotary Position Embedding for enhanced multimodal integration</td><td>Yes</td><td>httpss://github. com/QwenLM/ Qwen2-VL</td></tr><tr><td>InternVL-2 379], [380]</td><td>Text, image, video, and medical data</td><td>1B/2B/4B/ 8B/26B/40B</td><td>ViT as the vision encoder and a LLM as the language component</td><td>Progressive alignment strategy starting with coarse data and moving to fine data</td><td>Demonstrates powerful capabilities in handling complex multimodal tasks with various model sizes</td><td>Yes</td><td>httpss://internvl. github.io/blog/ 2024-07-02-InternVL-2 0/</td></tr><tr><td>CoqVLM 381]</td><td>Text and image</td><td>17B</td><td>A ViT encoder, a two-layer MLP adapter, a pre-trained large language model, and a visual expert module</td><td>Stage 1 focuses on image captioning; Stage 2 combines image captioning and referring expression comprehension tasks</td><td>Achieves deep integration of visual and language features while preserving the full capabilities of large language models</td><td>Yes</td><td>httpss://github. com/THUDM/ CogVLM</td></tr><tr><td></td><td>Text and image</td><td>7B/13B</td><td>Decoder-only architecture based on the Vicuna model, combined with a visual encoder</td><td>A combination of supervised training and additional instruction tuning</td><td>Ability to handle free-form region inputs via its hybrid region representation, enabling versatile spa tial understanding and grounding</td><td>Yes</td><td>httpss: //github.com/ apple/ml-ferret</td></tr><tr><td></td><td>Text and image</td><td>7B/13B</td><td>A vision encoder (CLIP ViT-L/14), a language decoder (Vicuna)</td><td>Pre-training using filtered image-text pairs, fine-tuning with a multimodal instruction-following dataset</td><td>Its lightweight architecture enables quick experimentation demonstrating capabilities close to GPT-4 in multimodal reason ing</td><td>Yes</td><td>httpss://llava-vl. github.io</td></tr><tr><td>LLaVA-1.5 383]</td><td>Text and image</td><td>7B/13B</td><td>A vision encoder (CLIP-ViT) and an encoder-decoder LLM architecture (e.g., Vicuna or LLaMA)</td><td>Pre-training on vision-language alignment with image-text pairs; visual instruction tuning with specific task-oriented data</td><td>Notable for its data efficiency and scaling to high-resolution image inputs</td><td>Yes</td><td>httpss://llava-vl github.io</td></tr><tr><td>BLIP-2 206]</td><td>Text and image</td><td>3.4B/12.1B</td><td>A frozen image encoder. a lightweight Querying Transformer to bridge the modality gap, and a frozen large language model</td><td>Vision-language representation learning: trains the Q-Former with a frozen image encoder; Vision-to-language generative learning: connects the Q-Former to a frozen LLM to enable image-to-text generation</td><td>Achieves state-of-the-art performance on various vision-language tasks with a compute efficient strategy by leveraging frozen pre-trained models</td><td>Yes</td><td>httpss://github. com/salesforce LAVIS/tree main/projects/ blip2</td></tr><tr><td>Phi-3.5- Vision 234]</td><td>Text and image</td><td>4.2B</td><td>Image encoder: CLIP ViT-L/14 to process visual inputs, and transformer decoder based on the Phi-3.5 mini model for textual outputs</td><td>Pre-training on a combination of interleaved image-text datasets synthetic OCR data, chart/table comprehension data, and text-only data; supervised fine-tuning using large-scale multimodal and text datasets; Direct Preference Optimization (DPO) to improve alignment, safety, and multimodal task performance</td><td>Excels in reasoning over visual and textual inputs, demonstrat ing competitive performance on single-image and multi-image tasks while being compact</td><td>Yes</td><td>httpss://github. com/microsoft/ Phi-3CookBook tree/main</td></tr></table>

<!-- Media -->

Enhancing spatial understanding and grounding, Ferret 382 offers an innovative approach tailored for GUI agents. By unifying referring and grounding tasks within a single framework and employing a hybrid region representation, it provides precise interaction with graphical interfaces. Its robustness against object hallucinations and efficient architecture make it ideal for on-device deployment in real-time GUI applications.


<!-- Meanless: JOURNAL OF LATEX CLASS FILES, DECEMBER 2024<br>53 -->

The LLaVA model [217] integrates a visual encoder with a language decoder, facilitating efficient alignment between modalities. Its lightweight projection layer and modular design enable quick experimentation and adaptation, making it suitable for GUI agents that require fast development cycles and strong multimodal reasoning abilities. Building on this, LLaVA-1.5 383 introduces a novel MLP-based cross-modal connector and scales to high-resolution image inputs, achieving impressive performance with minimal training data. Its data efficiency and open-source availability pave the way for widespread use in GUI applications requiring detailed visual reasoning.

BLIP-2 [206] employs a compute-efficient strategy by leveraging frozen pre-trained models and introducing a lightweight Querying Transformer. This design allows for state-of-the-art performance on vision-language tasks with fewer trainable parameters. BLIP-2's modularity and efficiency make it suitable for resource-constrained environments, highlighting its potential for on-device GUI agents.

Finally, Phi-3.5-Vision 234 achieves competitive performance in multimodal reasoning within a compact model size. Its innovative training methodology and efficient integration of image and text understanding make it a robust candidate for GUI agents that require multimodal reasoning and on-device inference without the computational overhead of larger models.

In summary, both close-source and open-source foundation models have significantly advanced the capabilities of LLM-powered GUI agents. While proprietary models offer powerful out-of-the-box performance, open-source models provide flexibility for customization and optimization, enabling tailored solutions for diverse GUI agent applications. The innovations in multimodal reasoning, efficiency, and scalability across these models highlight the evolving landscape of foundation models, paving the way for more intelligent and accessible GUI agents.

### 8.2 Large Action Models

While general-purpose foundation LLMs excel in capabilities like multimodal understanding, task planning, and tool utilization, they often lack the specialized optimizations required for GUI-oriented tasks. To address this, researchers have introduced Large Action Models (LAMs)—foundation LLMs fine-tuned with contextual, GUI-specific datasets (as outlined in Section 7) to enhance their action-driven capabilities. These models represent a significant step forward in refining the "brain" of GUI agents for superior performance.

In the realm of GUI agents, LAMs provide several transformative advantages:

1) Enhanced Action Orientation: By specializing in action-oriented tasks, LAMs enable accurate interpretation of user intentions and generation of precise action sequences. This fine-tuning ensures that LAMs can seamlessly align their outputs with GUI operations, delivering actionable steps tailored to user requests.

2) Specialized Planning for Long, Complex Tasks: LAMs excel in devising and executing intricate, multi-step workflows. Whether the tasks span multiple applications or involve interdependent operations, LAMs leverage their training on extensive action sequence datasets to create coherent, long-term plans. This makes them ideal for productivity-focused tasks requiring sophisticated planning across various tools.

3) Improved GUI Comprehension and Visual Grounding: Training on datasets that incorporate GUI screenshots allows LAMs to advance their abilities in detecting, localizing, and interpreting UI components such as buttons, menus, and forms. By utilizing visual cues instead of relying solely on structured UI metadata, LAMs become highly adaptable, performing effectively across diverse software environments.

4) Efficiency through Model Size Reduction: Many LAMs are built on smaller foundational models—typically around 7 billion parameters—that are optimized for GUI-specific tasks. This compact, purpose-driven design reduces computational overhead, enabling efficient operation even in resource-constrained environments, such as on-device inference.

As illustrated in Figure 26, the process of developing a purpose-built LAM for GUI agents begins with a robust, general-purpose foundation model, ideally with VLM capabilities. Fine-tuning these models on comprehensive, specialized GUI datasets—including user instructions, widget trees, UI properties, action sequences, and annotated screen-shots-transforms them into optimized LAMs, effectively equipping them to serve as the "brain" of GUI agents.

This optimization bridges the gap between planning and execution. A general-purpose LLM might provide only textual plans or abstract instructions in response to user queries, which may lack precision. In contrast, a LAM-empowered GUI agent moves beyond planning to actively and intelligently execute tasks on GUIs. By interacting directly with application interfaces, these agents perform tasks with remarkable precision and adaptability. This paradigm shift marks the evolution of GUI agents from passive task planners to active, intelligent executors.

### 8.3 LAMs for Web GUI Agents

In the domain of web-based GUI agents, researchers have developed specialized LAMs that enhance interaction and navigation within web environments. These models are tailored to understand the complexities of web GUIs, including dynamic content and diverse interaction patterns. We present an analysis of LAMs tailored for web GUI agents in Table 26

Building upon the need for multimodal understanding, WebGUM [153] integrates HTML understanding with visual perception through temporal and local tokens. It leverages Flan-T5 [498] for instruction fine-tuning and ViT [522] for visual inputs, enabling it to process both textual and visual information efficiently. This multimodal grounding allows WebGUM to generalize tasks effectively, significantly outperforming prior models on benchmarks like MiniWoB++ [146] and WebShop [425]. With its data-efficient design and capacity for multi-step reasoning, WebGUM underscores the importance of combining multimodal inputs in enhancing GUI agent performance.


<!-- Meanless: JOURNAL OF LATEX CLASS FILES, DECEMBER 2024<br>54 -->

<!-- Media -->

<!-- figureText: Step 1: Select the note on Slide 1. Step 2: Delete the selected note. Step 3: Repeat for each slide until all notes are delete.<br>Foundation LLM<br>Textual Reply<br>...<br>Fine-tuning Dataset<br>User Instruction<br>Widget Tree<br>UI Properties<br>Action Sequences<br>Screenshots<br>\{Botton - Chart, ...\}<br>Fine-tuning<br>\{Botton - Insert, ...\}<br>00<br>Task: Delete all notes on the slide deck<br>\{Menu - Design, ...\}<br><icon/><br>\{ListItem - Art, ...\}<br>...<br>Optimized LAM<br>Action Execution<br>GUI Agent -->

<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2025/11/2025_11_04__12_39_04_eb54b4.jpg"/>

Fig. 26: The evolution from foundation LLMs to GUI agent-optimized LAM with fine-tuning.

<!-- Media -->

Addressing the challenge of multi-step reasoning and planning in GUI environments, researchers have introduced frameworks that incorporate advanced search and learning mechanisms. For instance, Agent Q [281] employs MCTS combined with self-critique mechanisms and Direct Preference Optimization (DPO) [503] to improve success rates in complex tasks such as product search and reservation booking. By fine-tuning the LLaMA-3 70B model [91] to process HTML DOM representations and generate structured action plans, thoughts, and environment-specific commands, this framework showcases the power of integrating reasoning, search, and iterative fine-tuning for autonomous agent development.

Leveraging smaller models for efficient web interaction, GLAINTEL [385] demonstrates that high performance can be achieved without large computational resources. Utilizing the Flan-T5 [498] model with 780M parameters, it focuses on dynamic web environments like simulated e-commerce platforms. The model incorporates RL to optimize actions such as query formulation and navigation, effectively integrating human demonstrations and unsupervised learning. Achieving results comparable to GPT-4-based methods at a fraction of the computational cost, GLAINTEL underscores the potential of reinforcement learning in enhancing web-based GUI agents for task-specific optimization.

To enable continuous improvement and generalization across varied web domains, OpenWebVoyager [387] combines imitation learning with an iterative exploration-feedback-optimization cycle. Leveraging large multimodal models like Idefics2-8B [523], it performs autonomous web navigation tasks. By training on diverse datasets and fine-tuning using trajectories validated by GPT-4 feedback, the agent addresses real-world complexities without relying on synthetic environments. This approach significantly advances GUI agent frameworks by demonstrating the capability to generalize across varied web domains and tasks.

Moreover, tackling challenges such as sparse training data and policy distribution drift, WebRL 388 introduces a self-evolving curriculum and robust reward mechanisms for training LLMs as proficient web agents. By dynamically generating tasks based on the agent's performance, WebRL fine-tunes models like Llama-3.1 [91] and GLM-4 [499], achieving significant success rates in web-based tasks within the WebArena environment. This framework outperforms both proprietary APIs and other open-source models, highlighting the effectiveness of adaptive task generation and sustained learning improvements in developing advanced GUI agents.

These advancements in LAMs for web GUI agents illustrate the importance of integrating multimodal inputs, efficient model designs, and innovative training frameworks to enhance agent capabilities in complex web environments.

### 8.4 LAMs for Mobile GUI Agents

Mobile platforms present unique challenges for GUI agents, including diverse screen sizes, touch interactions, and resource constraints. Researchers have developed specialized LAMs to address these challenges, enhancing interaction and navigation within mobile environments. We present an overview of LAMs tailored for mobile GUI agents in Table 27 and 28


<!-- Meanless: JOURNAL OF LATEX CLASS FILES, DECEMBER 2024<br>55 -->

<!-- Media -->

TABLE 26: An overview of GUI-optimized models on web platforms.

<table><tr><td>Model</td><td>Platform</td><td>Foundation Model</td><td>Size</td><td>Input</td><td>Output</td><td>Dataset</td><td>Highlights</td><td>Link</td></tr><tr><td>Agent Q 281]</td><td>Web</td><td>LLaMA-3 70B 91J</td><td>70B</td><td>HTML DOM representations</td><td>Plans, thoughts, actions, and actio explanations</td><td>WebShop benchmark and OpenTable dataset</td><td>Combines Monte Carlo Tree Search (MCTS) with self-critique mechanisms, leveraging reinforcement learning to achieve exceptional performance</td><td>https://github.com/sentient% 2Dengineering/ agent-q</td></tr><tr><td>GLAINTEL 385]</td><td>Web</td><td>Flan-T5 498]</td><td>780M</td><td>User instructions and observations of webpage state</td><td>GUI actions</td><td>1.18M real-world products, 12,087 crowd-sourced natural language intents, 1,010 human demonstrations</td><td>Efficient use of smaller LLMs, and integration of RL and human demonstrations for robust performance</td><td>/</td></tr><tr><td>WebN-T5 386</td><td>Web</td><td>T5 87]</td><td>-</td><td>HTML and DOM with screenshots</td><td>Hierarchical navigation plans and GUI interactions</td><td>MiniWoB++, 13,000 human-made demonstrations</td><td>Combines supervised learning and reinforcement learning to address limitations of previous models in memorization and generalization</td><td>/</td></tr><tr><td>OpenWeb- We Voyager 387]</td><td></td><td>ldefics2- 8b-instruct 523]</td><td>8B</td><td>GUI screenshots, accessibility trees</td><td>Actions on GUI, planning and thought, answers queries</td><td>Mind2Web WebVoyager datasets and generated queries for real-world web navigation</td><td>Combining imitation learning with a feedback loop for continuous improvement</td><td>https: //github.com/ MinorJerry/ OpenWebVoyager</td></tr><tr><td>WebRL 388</td><td>Web</td><td>Llama-3.1 91 and GLM-4 524]</td><td>8B/9B/ 70B</td><td>Task instructions, action history, HTML content</td><td>Actions. element identifiers. explanations or notes</td><td>WebArena-Lite</td><td>Introduces a self-evolving online curriculum reinforcement learning framework, which dynamically generates tasks based on past failures and adapts to the agent's skill level</td><td>https://github.com/THUDM/ WebRL</td></tr><tr><td>WebGUM 153]</td><td>Web</td><td>Flan-T5 498] and Vision Transformer (ViT) 522]</td><td>3B</td><td>HTML, screenshots, interaction history. instructions</td><td>Web navigation actions and freeform text</td><td>MiniWoB++ and WebShop benchmarks</td><td>Integrates temporal and local multimodal perception, combining HTML and visual tokens, and uses an instruction-finetuned language model for enhanced reasoning and task generalization</td><td>https://console.cloud.google com/storage/ browser/ gresearch/ weblim</td></tr></table>

<!-- figureText: Screenshot State + Action<br>Policy Model<br><icon/><br>SYoufabe<br>Exploration<br>CLICK(50, 60)<br>CLICK(30, 120)<br>VEM<br>State-Action Values<br>PPO<br>X<br>Policy Model -->

<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2025/11/2025_11_04__12_39_04_1bfd21.jpg"/>

Fig. 27: The PPO training process of VEM 399. Figure adapted from the original paper.

<!-- Media -->

Focusing on detailed UI understanding, MobileVLM [353] introduces an advanced vision-language model designed specifically for mobile UI manipulation tasks. Built on Qwen-VL-Chat [210], it incorporates mobile-specific pretraining tasks for intra- and inter-UI comprehension. By leveraging the Mobile3M dataset—a comprehensive corpus of 3 million UI pages and interaction traces organized into directed graphs-the model excels in action prediction and navigation tasks. MobileVLM's novel two-stage pretraining framework significantly enhances its adaptability to mobile UIs, outperforming existing VLMs in benchmarks like ScreenQA [532] and Auto-UI [302]. This work highlights the effectiveness of tailored pretraining in improving mobile GUI agent performance.

Addressing the need for robust interaction in dynamic environments, DigiRL 264 presents a reinforcement learning-based framework tailored for training GUI agents in Android environments. By leveraging offline-to-online RL, DigiRL adapts to real-world stochasticity, making it suitable for diverse, multi-step tasks. Unlike prior models reliant on imitation learning, DigiRL autonomously learns from interaction data, refining itself to recover from errors and adapt to new scenarios. The use of a pre-trained Vision-Language Model with 1.3 billion parameters enables efficient processing of GUI screenshots and navigation commands. Its performance on the AITW dataset demonstrates a significant improvement over baseline methods, positioning DigiRL as a benchmark in the development of intelligent agents optimized for complex GUI interactions.

Both Digi-Q [398] and VEM [399] investigate the use of offline RL to enhance the performance of GUI agents without requiring direct interaction with the environment. Digi-Q employs temporal-difference learning to train a Q-function offline and derives policies through a Best-of-N selection strategy based on the predicted Q-values. Similarly, VEM introduces an environment-free RL framework tailored for training LLM-powered GUI agents using PPO. It directly estimates state-action values from offline data by fine-tuning with annotated value data from GPT-40, thereby enabling policy training without real-time execution in a GUI environment. At inference time, only the policy model is utilized. Figure 27 illustrates the overall architecture of VEM. The study further demonstrates that offline RL with structured credit assignment can achieve performance comparable to interactive RL models. Overall, VEM offers a scalable and layout-agnostic approach for training GUI agents while minimizing interaction costs. Both works underscore the potential of offline RL for GUI agent training.


<!-- Meanless: 56 -->

<!-- Media -->

JOURNAL OF LATEX CLASS FILES, DECEMBER 2024

TABLE 27: An overview of GUI-optimized models on mobile platforms (Part I).

<table><tr><td>Model</td><td>Platform</td><td>Foundation Model</td><td>Size</td><td>Input</td><td>Output</td><td>Dataset</td><td>Highlights</td><td>Link</td></tr><tr><td>Mobile- <br> VLM <br> 353]</td><td>Mobile Android</td><td>Qwen-VL-Chat 210]</td><td>9.8B</td><td>Screenshots and structured XML documents</td><td>Action predictions. navigation steps, and element locations</td><td>Mobile3M, includes 3 million UI pages, 20+ million actions, and XML data structured as directed graphs</td><td>Mobile-specific pretraining tasks that enhance intra- and inter-UI understanding, with a uniquely large and graph-structured Chinese UI dataset (Mobile3M)</td><td>https://github.com/XiaoMi/ mobilevlm</td></tr><tr><td>Octo-planner 361]</td><td>Mobile devices</td><td>Phi-3 N Mini 234]</td><td>3.8B</td><td>User queries and available function descriptions</td><td>Execution steps</td><td>1,000 data samples generated using GPT-4</td><td>Optimized for resource-constrained devices to ensure low latency, privacy, and offline functionality</td><td>https: //huggingface. co/NexaAlDev/ octopus-planning</td></tr><tr><td>DigiRL 264]</td><td>Mobile Android</td><td>AutoUI 302]</td><td>1.3B</td><td>Screenshots</td><td>GUI actions</td><td>AiTW</td><td>Offline-to-online reinforcement learning, bridging gaps in static and dynamic environments</td><td>https: //github.com/ DigiRL-agent/ digirl</td></tr><tr><td>LVG <br> 390]</td><td>Mobile Android</td><td>Swin Transformer [525] and BERT 82]</td><td>-</td><td>UI screenshots and free-form language expressions</td><td>Bounding box coordinates</td><td>UIBert dataset and synthetic dataset</td><td>Unifies detection and grounding tasks through layout-guided contrastive learning</td><td>/</td></tr><tr><td>Ferret-UI 352</td><td>Android and iPhone platforms</td><td>Ferret | 382]</td><td>7B/13B</td><td>Raw screen pixels, sub-images divided for finer resolution, bounding boxes and regional annotations</td><td>Widget bounding boxes, text from OCR tasks descriptions of UI elements or overall screen functionality, UI interaction actions</td><td>Generated from RICO (for Android) and AMP (for iPhone)</td><td>Multi-platform support with high-resolution adaptive image encoding</td><td>https://github.com/apple/ ml-ferret/tree/ main/ferretui</td></tr><tr><td>Octopus 391]</td><td>Mobile devices</td><td>CodeLlama- 7B 5261 Google Gemma 2B 502</td><td>7B, 2B</td><td>API documentation examples</td><td>Function names with arguments for API calls</td><td>RapidAPI Hub</td><td>Use of conditional masking to enforce correct output formatting</td><td>/</td></tr><tr><td>Octopus v2 392]</td><td>Edge devices</td><td>Gemma-2B 502]</td><td>2B</td><td>User queries and descriptions of available functions</td><td>Function calls with precise parameters</td><td>20 Android APIs. with up to 1,000 data points generated for training</td><td>Functional tokenization strategy. which assigns unique tokens to function calls, significantly reducing the context length required for accurate prediction</td><td>/</td></tr><tr><td>Octopus v3 393]</td><td>Edge devices</td><td>CLIP-based model and a causal language model</td><td>Less than 1 billion parameters</td><td>Queries and commands, images and functional tokens</td><td>Functional tokens for actions</td><td>Leveraged from Octopus v2 [392]</td><td>Introduction of functional tokens for multimodal applications enables the representation of any function as a token, enhancing the model's flexibility</td><td>/</td></tr><tr><td>Octopus v4 394]</td><td>Serverless cloud-based platforms and edge devices</td><td>17 models</td><td>Varies</td><td>User queries</td><td>Domain-specific answers, actions</td><td>Synthetic datasets similar to Octopus v2</td><td>Graph-based framework integrating multiple specialized models for optimized performance</td><td>https://github.com/NexaAI/ octopus-v4</td></tr></table>

<!-- Media -->

To enhance GUI comprehension and reduce reliance on textual data, VGA 354 employs fine-tuned vision-language models that prioritize image-based cues such as shapes, colors, and positions. Utilizing the RICO [355] dataset for training, VGA is tailored for Android GUIs and employs a two-stage fine-tuning process to align responses with both visual data and human intent. The model excels in understanding GUI layouts, predicting design intents, and facilitating precise user interactions. By outperforming existing models like GPT- 4V in GUI comprehension benchmarks, VGA sets a new standard for accuracy and efficiency in mobile GUI agents.

In the context of lightweight and efficient models, UINav [395] demonstrates a practical system for training neural agents to automate UI tasks on mobile devices. It balances accuracy, generalizability, and computational efficiency through macro actions and an error-driven demonstration collection process. UINav uses a compact encoder-decoder architecture and SmallBERT [528] for text and screen element encoding, making it suitable for on-device inference. A key innovation is its ability to generalize across diverse tasks and apps with minimal demonstrations, addressing key challenges in UI automation with a versatile framework.

UI-R1 [401] introduces a RL-based training paradigm aimed at enhancing GUI action prediction for multimodal large language models (MLLMs). The resulting model, UI-R1-3B, fine-tunes Qwen2.5-VL-3B using a novel rule-based reward function that jointly evaluates action type correctness and click coordinate accuracy, while also enabling o1-style [533] chain-of-thought (CoT) reasoning through structured <think> tags. UI-R1 relies on only 136 high-quality samples selected via a three-stage filtering strategy. Despite this limited supervision, UI-R1-3B achieves significant improvements on both in-domain and out-of-domain benchmarks. By leveraging Group Relative Policy Optimization (GRPO) [534], the framework aligns policy optimization with the goals of GUI grounding and task execution. UI-R1 establishes a scalable and data-efficient approach for training GUI agents via RL and paves the way for lightweight yet effective agent design. Its methodology has also been successfully extended to cross-platform agents [410], [411], demonstrating strong generalization capabilities.


<!-- Meanless: 57 -->

<!-- Media -->

JOURNAL OF LATEX CLASS FILES, DECEMBER 2024

TABLE 28: An overview of GUI-optimized models on mobile platforms (Part II).

<table><tr><td>Model</td><td>Platform</td><td>Foundation Model</td><td>Size</td><td>Input</td><td>Output</td><td>Dataset</td><td>Highlights</td><td>Link</td></tr><tr><td>VGA <br> 354]</td><td>Mobile Android</td><td>LLaVA-v1.6- mistral-7B 217]</td><td>7B</td><td>GUI screenshots with with positional, visual, and hierarchical data</td><td>Actions and function calls, descriptions of GUI components, navigation and task planning</td><td>63.8k-image dataset constructed from the RICO</td><td>Minimizes hallucinations in GUI comprehension by employing an image-centric fine-tuning approach, ensuring balanced attention between text and visual content</td><td>https: //github.com/ Linziyang1999/ VGA% 2Dvisual% 2DGUI% 2Dassistant</td></tr><tr><td>MobileFlow Mobile <br> 159]</td><td>phones</td><td>Qwen-VL-Chat [210]</td><td>21B</td><td>GUI screenshots with OCR textual information and bounding boxes</td><td>GUI actions and question answering</td><td>70k manually labeled business-specific data spanning 10 business sectors, and datasets like RefCOCO, ScreenQA, Flickr30K</td><td>Hybrid visual encoder capable of variable-resolution input and Mixture of Experts (MoE) [527] for enhanced performance and efficiency</td><td>7</td></tr><tr><td>UINav <br> 395]</td><td>Mobile Android</td><td>SmallBERT 528]</td><td>Agent model: 320k, Referee model: 430k, Small-BERT model: 17.6MB</td><td>UI elements, utterance, screen representation</td><td>Predicted actions and element to act upon</td><td>43 tasks across 128 Android apps and websites, collecting 3,661 demonstrations</td><td>Introduces a macro action framework and an error-driven demonstration collection process, significantly reducing training effort while enabling robust task performance with small, efficient models suitable for mobile devices</td><td>/</td></tr><tr><td>AppVLM <br> 397]</td><td>Android mobile devices</td><td>Paligemma- 3B-896 529]</td><td>3B</td><td>Annotated screen-shots with bounding boxes and UI labels</td><td>GUI actions</td><td>AndroidControl 515], AndroidWorld 440]</td><td>A lightweight model that achieves near-GPT-40 performance in Android control tasks while being ${10} \times$ faster and more resource-efficient.</td><td>/</td></tr><tr><td>VSC-RL 396]</td><td>Mobile Android</td><td>AutoUI 302 Gemini-1.5- Pro</td><td>/</td><td>Screenshots</td><td>GUI actions</td><td>AitW</td><td>Addresses sparse-reward, long-horizon tasks for RL by autonomously breaking a complicated goal into subgoals</td><td>https:// ai-agents-2030 github.io/ VSC-RL</td></tr><tr><td>Diai-Q <br> 398]</td><td>Mobile Android</td><td>LLaVA-1.5 383</td><td>7B</td><td>GUI screenshots</td><td>GUI actions, Q-values</td><td>AitW 358]</td><td>Introduces a VLM-based Q-function for GUI agent training, enabling reinforcement learning without online interactions.</td><td>https: //github.com/ DigiRL-agent/ digiq</td></tr><tr><td>VEM <br> 399]</td><td>Mobile Android</td><td>Qwen2VL 231]</td><td>7B</td><td>GUI screenshots</td><td>GUI actions, Q-values</td><td>AitW 358]</td><td>Unlike traditional RL methods that require environment interactions, VEM enables training purely on offline data with a Value Environment Model.</td><td>https://github.com/microsoft/ GUI-Agent-RL</td></tr><tr><td>MP- <br> GUI <br> 400]</td><td>Mobile Android</td><td>InternViT- 300M and InternLM2.5- 7B-chat 530]</td><td>8B</td><td>GUI screenshots</td><td>Natural language output, element grounding, captioning, and semantic navigation</td><td>680K mixed-modality dataset</td><td>Introduces a tri-perceiver architecture that models textual, graphical, and spatial modalities to enhance GUI reasoning</td><td>https://github.com/Big Taige/ MP-GUI</td></tr><tr><td>UI-R1 <br> 401]</td><td>Mobile Android</td><td>Qwen2.5- VL-3B</td><td>3B</td><td>GUI screenshots</td><td>Reasoning text and GUI actions</td><td>ScreenSpo and An-droidControl</td><td>Introduces a rule-based reinforcement learning approach using GRPO to enhance reasoning and action prediction in GUI tasks with only 136 examples</td><td>https: //github.com/ III6gg/UI-R1</td></tr><tr><td>ViMo <br> 402]</td><td>Mobile Android</td><td>Pre-trained Stable Diffusion model [531]</td><td>/</td><td>Current GUI image. user action (in natural language), GUI text representation</td><td>GUI text representation of the next state and reconstructed full GUI image (visual prediction of the next screen)</td><td>Android Control and AITW</td><td>First GUI world model that predicts future visual GUI states</td><td>https:// ai-agents-2030. github.io/ViMo/</td></tr></table>

<!-- Media -->

In addition to action models, ViMo [402] introduces a novel generative visual world model for GUI agents, aimed at improving App agent decision-making by predicting the next GUI state as an image rather than a textual description. A key innovation of ViMo is the Symbolic Text Representation (STR), which replaces GUI text regions with structured placeholders to facilitate accurate and legible text synthesis. This decoupled design allows the system to handle GUI graphics generation using a fine-tuned diffusion model, and text generation through an LLM, thereby achieving high visual fidelity and semantic precision. ViMo significantly boosts both GUI prediction quality and downstream agent performance, with a reported 29.14% relative improvement in GUI generation metrics and enhanced planning accuracy for long-horizon tasks. As a forward simulator, ViMo represents a crucial advancement toward reliable world models for mobile GUI agents, supporting more effective decision evaluation and trajectory planning in visual environments.


<!-- Meanless: 58 -->

### 8.5 LAMs for Computer GUI Agents

<!-- Media -->

JOURNAL OF LATEX CLASS FILES, DECEMBER 2024

TABLE 29: An overview of GUI-optimized models on computer platforms.

<table><tr><td>Model</td><td>Platform</td><td>Foundation Model</td><td>Size</td><td>Input</td><td>Output</td><td>Dataset</td><td>Highlights</td><td>Link</td></tr><tr><td>Screen- <br> Agent <br> 366]</td><td>Linux and Windows desktop</td><td>CogAgent <br> [15]</td><td>18B</td><td>GUI screenshots</td><td>Mouse and keyboard actions</td><td>273 task sessions</td><td>Comprehensive pipeline of planning, acting, and reflecting to handle real computer screen operations autonomously</td><td>https: <br> //github.com/ <br> niuzaisheng/ <br> ScreenAgent</td></tr><tr><td>Octopus <br> 403]</td><td>Desktop</td><td>MPT-7B 535 and CLIP ViT-L/14 509]</td><td>7B</td><td>Visual images, scene graphs containing objects relations, environment messages</td><td>Executable action code and plans</td><td>OctoGibson: 476 tasks with structured initial and goal states; OctoMC: 40 tasks across biomes; OctoGTA: 25 crafted tasks spanning different game settings</td><td>Incorporates reinforcement learning with environmental feedback</td><td>https: //choiszt.github. io/Octopus/</td></tr><tr><td>LAM 367</td><td>Windows OS</td><td>Mistral-7B 501]</td><td>7B</td><td>Task requests in natural language, application environmental data</td><td>Plans, actions</td><td>76,672 task-plan pairs, 2,192 task-action trajectories</td><td>The LAM model bridges the gap between planning and action execution in GUI environments. It introduces a multi-phase training pipeline combining task planning, imitation learning, self-boosting exploration, and reward-based optimization for robust action-oriented performance.</td><td>https://github.com/microsoft/ UFO/tree/main/ dataflow</td></tr><tr><td>ScreenLL <br> 404]</td><td>MDesktop</td><td>LLaVA</td><td>7B, 13B</td><td>GUI screenshots</td><td>Predicted GUI actions</td><td>High-resolution YouTube tutorials</td><td>Introduces a novel stateful screen schema to compactly represent GUI interactions over time, enabling fine-grained understanding and accurate action prediction</td><td>/</td></tr></table>

<!-- Media -->

For desktop and laptop environments, GUI agents must handle complex applications, multitasking, and varied interaction modalities. Specialized LAMs for computer GUI agents enhance capabilities in these settings, enabling more sophisticated task execution. We overview of LAMs for computer GUI agents across in Table 29.

Integrating planning, acting, and reflecting phases, ScreenAgent 366 is designed for autonomous interaction with computer screens. Based on CogAgent [15], it is fine-tuned using the ScreenAgent Dataset, providing comprehensive GUI interaction data across diverse tasks. With inputs as screenshots and outputs formatted in JSON for mouse and keyboard actions, ScreenAgent achieves precise UI element localization and handles continuous multi-step tasks. Its capability to process real-time GUI interactions using a foundation model sets a new benchmark for LLM-powered GUI agents, making it an ideal reference for future research in building more generalized intelligent agents.

Bridging high-level planning with real-world manipulation, Octopus [403] represents a pioneering step in embodied vision-language programming. Leveraging the MPT-7B [535] and CLIP ViT-L/14 [509], Octopus integrates egocentric and bird's-eye views for visual comprehension, generating executable action code. Trained using the OctoVerse suite, its datasets encompass richly annotated environments like OmniGibson, Minecraft, and GTA-V, covering routine and reasoning-intensive tasks. Notably, Octopus innovates through Reinforcement Learning with Environmental Feedback, ensuring adaptive planning and execution. Its vision-dependent functionality offers seamless task generalization in unseen scenarios, underscoring its capability as a unified model for embodied agents operating in complex GUI environments.

Wang et al., [367] present a comprehensive overview of LAMs, a new paradigm in AI designed to perform tangible actions in GUI environments, using UFO [19] at Windows OS as a case study platform. Built on the Mistral-7B [501] foundation, LAMs advance beyond traditional LLMs by integrating task planning with actionable outputs. Leveraging structured inputs from tools like the UI Automation (UIA) API, LAMs generate executable steps for dynamic planning and adaptive responses. A multi-phase training strategy—encompassing task-plan pretraining, imitation learning, self-boosting exploration, and reinforcement learning—ensures robustness and accuracy. Evaluations on real-world GUI tasks highlight LAMs' superior task success rates compared to standard models. This innovation establishes a foundation for intelligent GUI agents capable of transforming user requests into real-world actions, driving significant progress in productivity and automation.

These developments in computer GUI agents highlight the integration of advanced visual comprehension, planning, and action execution, paving the way for more sophisticated and capable desktop agents.

### 8.6 Cross-Platform Large Action Models

To achieve versatility across various platforms, cross-platform LAMs have been developed, enabling GUI agents to operate seamlessly in multiple environments such as mobile devices, desktops, and web interfaces. We provide an analysis of LAMs tailored for cross-platform GUI agents in Table 30 and 31

CogAgent [15] stands out as an advanced visual language model specializing in GUI understanding and navigation across PC, web, and Android platforms. Built on CogVLM [381], it incorporates a novel high-resolution cross-module to process GUI screenshots efficiently, enabling detailed comprehension of GUI elements and their spatial relationships. Excelling in tasks requiring OCR and GUI grounding, CogAgent achieves state-of-the-art performance on benchmarks like Mind2Web [212] and AITW [358]. Its ability to generate accurate action plans and interface with GUIs positions it as a pivotal step in developing intelligent agents optimized for GUI environments. CogAgent has further evolved into its beta version, GLM-PC [505], offering enhanced control capabilities.


<!-- Meanless: 59 -->

<!-- Media -->

JOURNAL OF LATEX CLASS FILES, DECEMBER 2024

TABLE 30: An overview of GUI-optimized models on cross-platform agents (Part I).

<table><tr><td>Model</td><td>Platform</td><td>Foundation Model</td><td>Size</td><td>Input</td><td>Output</td><td>Dataset</td><td>Highlights</td><td>Link</td></tr><tr><td>RUIG <br> 405]</td><td>Mobile and desktop</td><td>Swin Transformer |525 and BART 88]</td><td>decoder layers</td><td>UI screenshots and text instructions</td><td>Bounding box predictions in linguistic form</td><td>MoTIF dataset and Ri-coSCA dataset for mobile UI data and Common Crawl for desktop UI data</td><td>Innovatively uses policy gradients to improve the spatial decoding in the pixel-to-sequence paradigm</td><td>/</td></tr><tr><td>CogAgen</td><td>PC, web, and Android platforms</td><td>CogVLM- 17B 381]</td><td>18B</td><td>GUI screen-shots combined with OCR-derived text</td><td>Task plans, action sequences, and textual descriptions</td><td>CCS400K, text recognition datasets: 80M synthetic text images, visual grounding datasets and GUI dataset Mind2Web and AiTW</td><td>High-resolution cross-module to balance computational efficiency and high-resolution input processing</td><td>https://github com/THUDM/ CogVLM</td></tr><tr><td>SeeClick</td><td>iOS, Android, macOS, Windows, and web</td><td>Qwen-VL 210]</td><td>9.6B</td><td>GUI screenshots and textual instructions</td><td>GUI actions and element locations for interaction</td><td>300k webpages with text and icons, RICO, and data from LLaVA</td><td>Ability to perform GUI tasks purely from screenshots and its novel GUI grounding pre-training approach</td><td>https://github com/njucckevin/ SeeClick</td></tr><tr><td>ScreenAl 375]</td><td>Mobile, desktop, and tablet Uls</td><td>PaLI-3 [536]</td><td>5B</td><td>GUI screen-shots with OCR text, image captions, and other visual elements</td><td>Text-based answers questions, screen annotations with bounding box coordinates and labels. navigation instructions. summaries of screen content</td><td>262M mobile web screenshots and 54M mobile app screenshots</td><td>Unified representation of UIs and infographics, combining visual and textual elements</td><td>https://github com/kyegomez/ ScreenAl</td></tr><tr><td>V-Zen <br> 408]</td><td>Computers and Web</td><td>Vicuna-7B 510], DINO 183],EVA- 2-CLIP 537]</td><td>7B</td><td>Text, GUI Images</td><td>Action Prediction. GUI Bounding Box</td><td>GUIDE 370]</td><td>Dual-resolution visual encoding for precise GUI grounding and task executior</td><td>https: //github.com/ abdur75648/ V-Zen</td></tr><tr><td>Ferret-UI 406]</td><td>iPhone, Android, iPad, Web, Ap-pleTV</td><td>Vicuna- 13B 510], Gemma- 2B 502 Llama3-8B 91]</td><td>Vicuna- 13B 510], Gemma- 2B 502] Llama3- 8B 91]</td><td>UI screenshots annotated bounding boxes and labels for UI widgets, OCR detected text and bounding boxes text elements. source HTML hierarchy trees for web data</td><td>Descriptions of UI elements, widget classification OCR, tapability and text/widget location. interaction instructions and multi-round interaction-based QA</td><td>Core-set, GroundUI-18k, GUIDE, Spotlight</td><td>Multi-platform support with high-resolution adaptive image encoding</td><td>/</td></tr><tr><td>ShowUI 238]</td><td>Websites, desktops, and mobile phones</td><td>Phi-3.5- Vision 234]</td><td>4.2B</td><td>GUI screen-shots with OCR for text-based UI elements and visual grounding for icons and widgets</td><td>GUI actions. navigation element location</td><td>ScreenSpot, RICO, GUIEnv, GUIAct, AiTW, AiTZ, GUI-World</td><td>Interleaved Vision-Language Action approach, allowing seamless navigation, grounding, and understanding of GUI environments</td><td>https://github com/showlab ShowUI</td></tr><tr><td>OS-ATLAS 232]</td><td>Windows, macOS, Linux, Android, and the web</td><td>InternVL-2 379 and Qwen2-VL 210]</td><td>4B/7B</td><td>GUI screen-shots</td><td>GUI actions</td><td>AndroidControl, SeeClick, and others annotated with GPT-4 over 13 million GUI elements and 2.3 million screenshots</td><td>The first foundation action model designed for generalist GUI agents, supporting cross-platform GUI tasks, and introducing a unified action space</td><td>https://osatlas.github.io/</td></tr><tr><td>xLAM 372</td><td>Diverse environments</td><td>Mistral-7B 501 and DeepSeek-Coder-7B 538</td><td>Range from 1B to 8×22B</td><td>Unified function-calling data formats</td><td>Function calls, thought processes</td><td>Synthetic and augmented data. including over 60,000 high-quality samples generated using APIGen from 3,673 APIs across 21 categories</td><td>Excels in function-calling tasks by leveraging unified and scalable data pipelines</td><td>https: //github.com/ SalesforceAlResear xLAM</td></tr><tr><td>SpiritSigh <br> 384]</td><td>Web, Android, Windows Desktop</td><td>InternVL 379]</td><td>2B, 8B, and 26B</td><td>GUI screen-shots</td><td>GUI actions</td><td>AitW [358], Common-Crawl websites, and custom annotations</td><td>Introduces a Universal Block Parsing (UBP) method to resolve positional ambiguity in high-resolution visual inputs.</td><td>https: //hzhiyuan. github.io/ SpiritSight-Agent</td></tr></table>

<!-- Media -->

Focusing on universal GUI understanding, Ferret-UI 2 406 from Apple is a state-of-the-art multimodal large language model designed to master UI comprehension across diverse platforms, including iPhones, Android devices, iPads, web, and AppleTV. By employing dynamic high-resolution image encoding, adaptive gridding, and high-quality multimodal training data generated through GPT-4, it outperforms its predecessor and other competing models in UI referring, grounding, and interaction tasks. Ferret-UI 2's advanced datasets and innovative training techniques ensure high accuracy in spatial understanding and user-centered interactions, setting a new benchmark for cross-platform UI adaptability and performance.


<!-- Meanless: 60 -->

<!-- Media -->

JOURNAL OF LATEX CLASS FILES, DECEMBER 2024

TABLE 31: An overview of GUI-optimized models on cross-platform agents (Part II).

<table><tr><td>Model</td><td>Platform</td><td>Foundation Model</td><td>Size</td><td>Input</td><td>Output</td><td>Dataset</td><td>Highlights</td><td>Link</td></tr><tr><td>Falcon- <br> UI 373</td><td>iOS, Android, Windows, Linux, Web</td><td>Qwen2-VL- 7B</td><td>7B</td><td>Screenshots of GUI with node information and annotations for visible elements</td><td>GUI actions and coordinates or bounding boxes for interaction elements</td><td>Insight-UI dataset, further fine-tuned on datasets such as AITW, AITZ, Android Control, and Mind2Web</td><td>Decouples GUI context comprehension from instruction-following tasks, leveraging an instruction-free pretraining approach.</td><td>/</td></tr><tr><td>UI-TARS 407]</td><td>Web, Desktop (Windows, ma-cOS), and Mobile (Android)</td><td>Qwen-2-VL 7B and 72B 231]</td><td>7B 72B</td><td>GUI screen-shots</td><td>GUI actions</td><td>GUI screenshots and metadata collected from websites, apps, and operating systems; action trace datasets from various GUI agent benchmarks; 6M GUI tutorials for reasoning enhancement; multiple open-source datasets</td><td>Pure vision-based perception with standardized GUI actions across platforms (Web, Mobile, Desktop).</td><td>https://github.com/bytedance/ UI-TARS</td></tr><tr><td>Magma 409]</td><td>Web, Mobile, Desktop, Robotics</td><td>LLaMA-3- 8B [91], ConvNeXt-Xxlarge 539</td><td>8.6B</td><td>GUI screenshots, textual task descriptions</td><td>GUI actions, robotic manipulation</td><td>UI, robotics data, human instructional videos</td><td>Jointly trains on heterogeneous datasets, enabling generalization across digital and physical tasks</td><td>https: <br> //microsoft. <br> github.io/ <br> Magma/</td></tr><tr><td>GUI-R1 410]</td><td>Windows, Linux, MacOS, Android, and Web</td><td>QwenVL2.5 540]</td><td>3B a 7B</td><td>GUI shots</td><td>Reasoning text and GUI actions</td><td>Mixture of 3K high-quality samples</td><td>first framework to apply rule-based reinforcement learning (RFT) to high-level GUI tasks across platforms.</td><td>https://github.com/ritzz-ai/ GUI-R1.git</td></tr><tr><td>InfiGUI- <br> R1 <br> 411]</td><td>Web, Desktop, and Android</td><td>Qwen2.5- VL-3B-Instruct</td><td>3B</td><td>GUI screenshots, Accessibility Tree</td><td>Reasoning text and GUI actions</td><td>Diverse dataset mixture</td><td>Two-stage training framework Actor2Reasoner: (1) Reasoning Injection via Spatial Reasoning Distillation, and (2) Deliberation Enhancement via Reinforcement Learning with Sub-goal Guidance and Error Recovery Scenario Construction</td><td>https: //github.com/ RealIm-Labs/ InfiGUI-R1</td></tr><tr><td>Task Generalization 389</td><td>Web and Android (Mobile)</td><td>Qwen2-VL- 7B-Instruct 231]</td><td>7B</td><td>GUI screen-shots</td><td>Thoughts and grounded coordinate-based actions</td><td>11 domain datasets with 56K GUI trajectory samples</td><td>Introduces mid-training on diverse non-GUI reasoning tasks (particularly math and code) to substantially enhance GUI agent planning capabilities</td><td>https://github.com/hkust-nlp/ GUIMid</td></tr></table>

<!-- Media -->

Advancing GUI automation, ShowUI 238 introduces a pioneering Vision-Language-Action model that integrates high-resolution visual inputs with textual understanding to perform grounding, navigation, and task planning. Optimized for web, desktop, and mobile environments, ShowUI leverages the Phi- 3.5-vision-instruct backbone and comprehensive datasets to achieve robust results across benchmarks like ScreenSpot [25] and GUI-Odyssey [359]. Its ability to process multi-frame and dynamic visual inputs alongside JSON-structured output actions highlights its versatility. With innovations in interleaved image-text processing and function-calling capabilities, ShowUI sets a new standard for LLM-powered GUI agents.

Addressing the need for a unified action space, OS-ATLAS 232 introduces a foundational action model specifically designed for GUI agents across platforms like Windows, macOS, Linux, Android, and the web. By leveraging a massive multi-platform dataset and implementing a unified action space, OS-ATLAS achieves state-of-the-art performance in GUI grounding and out-of-distribution generalization tasks. Its scalable configurations adapt to varying computational needs while maintaining versatility in handling natural language instructions and GUI elements. As a powerful open-source alternative to commercial solutions, OS-ATLAS marks a significant step toward democratizing access to advanced GUI agents.

Magma [409] is a foundation model for multimodal AI agents that integrates LLMs with vision and action understanding to complete UI navigation and robotic manipulation tasks. Unlike previous models optimized for either UI automation or robotics, Magma jointly trains on a heterogeneous dataset (about 39M samples) spanning UI screenshots, web navigation, robot trajectories, and instructional videos. It employs SoM and Trace-of-Mark techniques, which enhance action grounding and prediction by labeling actionable elements in GUI environments and tracking motion traces in robotic tasks.

UI-TARS [407] is an advanced, vision-based Large Action Model (LAM) optimized for multi-platform GUI agents. Unlike traditional approaches, it relies solely on GUI screenshots for perception, eliminating the need for structured representations. By incorporating a unified action space, UI-TARS enables seamless execution across Web, Windows, macOS, and Android environments. Built on Qwen-2-VL, it is trained on 6 million GUI tutorials, large-scale screenshot datasets, and multiple open-source benchmarks. A key innovation of UI-TARS is its System-2 reasoning capability, which allows it to generate explicit reasoning steps before executing actions, enhancing decision-making in dynamic environments. Additionally, it employs an iterative self-improvement framework, refining its performance through reflection-based learning. Experimental results demonstrate that UI-TARS outperforms existing models, including GPT-40 and Claude, in task execution benchmarks.


<!-- Meanless: JOURNAL OF LATEX CLASS FILES, DECEMBER 2024<br>61 -->

These cross-platform LAMs demonstrate the potential of unified models that can adapt to diverse environments, enhancing the scalability and applicability of GUI agents in various contexts.

### 8.7 Takeaways

The exploration of LAMs for GUI agents has revealed several key insights that are shaping the future of intelligent interaction with graphical user interfaces:

1) Smaller Models for On-Device Inference: Many of the optimized LAMs are built from smaller foundational models, often ranging from 1 billion to 7 billion parameters. This reduction in model size enhances computational efficiency, making it feasible to deploy these models on resource-constrained devices such as mobile phones and edge devices. The ability to perform on-device inference without relying on cloud services addresses privacy concerns and reduces latency, leading to a more responsive user experience.

2) Enhanced GUI Comprehension Reduces Reliance on Structured Data: Models like VGA 354 and OmniParser [184] emphasize the importance of visual grounding and image-centric fine-tuning to reduce dependency on structured UI metadata. By improving GUI comprehension directly from visual inputs, agents become more adaptable to different software environments, including those where structured data may be inaccessible or inconsistent.

3) Reinforcement Learning Bridges Static and Dynamic Environments: The application of reinforcement learning in models like DigiRL [264] demonstrates the effectiveness of bridging static training data with dynamic real-world environments. This approach allows agents to learn from interactions, recover from errors, and adapt to changes, enhancing their robustness and reliability in practical applications.

4) Unified Function-Calling Enhances Interoperability: Efforts to standardize data formats and function-calling mechanisms, as seen in models like xLAM [372], facilitate multi-turn interactions and reasoning across different platforms. This unification addresses compatibility issues and enhances the agent's ability to perform complex tasks involving multiple APIs and services.

5) Inference-Time Computing and Reasoning Models: Recent work highlights the importance of inference-time computing, where models plan, reason, and decompose tasks on the fly without architectural changes. Techniques such as extended context windows and chain-of-thought prompting (e.g., "o1-style" reasoning) enable more robust, long-horizon decision-making. UI-R1 [401], GUI-R1 [410] and InfiGUI-R1 [411] are pioneering efforts in this direction. There is also growing interest in rule-based rewards and cost functions that guide inference-time behavior, integrating explicit heuristics to improve the stability, interpretability, and generalization of GUI agents.

The advancements in LAMs for GUI agents highlight a trend toward specialized, efficient, and adaptable models capable of performing complex tasks across various platforms. By focusing on specialization, multimodal integration, and innovative training methodologies, researchers are overcoming the limitations of general-purpose LLMs. These insights pave the way for more intelligent, responsive, and user-friendly GUI agents that can transform interactions with software applications.

## 9 EVALUATION FOR LLM-BRAINED GUI AGENTS

In the domain of GUI agents, evaluation is crucial for enhancing both functionality and user experience [56], [58] and should be conducted across multiple aspects. By systematically assessing these agents' effectiveness across various tasks, evaluation tonly gauges their performance in different dimensions but also provides a framework for their continuous improvement [541]. Furthermore, it encourages in vation by identifying areas for potential development, ensuring that GUI agents evolve in tandem with advancements in LLMs and align with user expectations.

As illustrated in Figure 28 when a GUI agent completes a task, it produces an action sequence, captures screenshots, extracts UI structures, and logs the resulting environment states. These outputs serve as the foundation for evaluating the agent's performance through various metrics and measurements across diverse platforms. In the subsequent sections, we delve into these evaluation methodologies, discussing the metrics and measurements used to assess GUI agents comprehensively. We also provide an overview of existing benchmarks tailored for GUI agents across different platforms, highlighting their key features and the challenges they address.

### 9.1 Evaluation Metrics

Evaluating GUI agents requires robust and multidimensional metrics to assess their performance across various dimensions, including accuracy, efficiency, and compliance (e.g., safety). In a typical benchmarking setup, the GUI agent is provided with a natural language instruction as input and is expected to auto mously execute actions until the task is completed. During this process, various assets can be collected, such as the sequence of actions taken by the agent, step-wise observations (e.g., DOM or HTML structures), screenshots, runtime logs, final states, and execution time. These assets enable evaluators to determine whether the task has been completed successfully and to analyze the agent's performance. In this section, we summarize the key evaluation metrics commonly used for benchmarking GUI agents. te that different research works may use different names for these metrics, but with similar calculations. We align their names in this section.

1) Step Success Rate: Completing a task may require multiple steps. This metric measures the ratio of the number of steps that are successful over the total steps within a task. A high step success rate indicates precise and accurate execution of granular steps, which is essential for the reliable performance of tasks involving multiple steps [212], [349], [358].

2) Turn Success Rate: A turn indicates a single interaction between the user and the agent. A turn may consist of multiple steps, and completing a task may consist of multiple turns. This metric measures the ratio of turns that successfully address the request in that interaction over all turns. It focuses on the agent's ability to understand and fulfill user expectations during interactive or dialog-based tasks, ensuring the agent's responsiveness and reliability across iterative interactions, particularly in tasks requiring dynamic user-agent communication [155], 348].


<!-- Meanless: JOURNAL OF LATEX CLASS FILES, DECEMBER 2024<br>62 -->

<!-- Media -->

<!-- figureText: 哇<br>Task 1: Change font size to 20 in the PPT page 2<br>Actions<br>Evaluation Measurements<br>Task List<br>Screenshots<br>Trajectory<br>UI structure<br>GUI Agent<br>Env State<br>Metric<br>Score<br>0.65<br>...<br>Task Success Rate<br>Interaction<br>Step Success Rate<br>0.82<br>...<br>...<br>IOS<br>Platform<br>Evaluation Metrics -->

<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2025/11/2025_11_04__12_39_04_cf705e.jpg"/>

Fig. 28: An illustrative example of evaluation of task completion by a GUI agent.

<!-- Media -->

3) Task Success Rate: Task success rate measures the successful task completion over all tasks set in the benchmark. It evaluates whether the final task completion state is achieved while ig ring the intermediate steps. This metric provides an overall measure of end-to-end task completion, reflecting the agent's ability to handle complex workflows holistically [419], [425], [438].

4) Efficiency Score: Efficiency score evaluates how effectively the agent completes tasks while considering resource consumption, execution time, or total steps the agent might take. This metric can be broken down into the following sub-metrics:

- Time Cost: Measures the time taken to complete tasks.

- Resource Cost: Measures the memory/CPU/GPU usage to complete tasks.

- LLM Cost: Evaluates the computational or monetary cost of LLM calls used during task execution.

- Step Cost: Measures the total steps required to complete tasks.

Depending on the specific metrics used, the efficiency score can be interpreted differently in different papers [442], [444].

5) Completion under Policy: This metric measures the rate at which tasks are completed successfully while adhering to policy constraints. It ensures that the agent complies with user-defined or organizational rules, such as security, ethical, safety, privacy, or business guidelines, during task execution. This metric is particularly relevant for applications where compliance is as critical as task success 416 .

6) Risk Ratio: Similar to the previous metric, the risk ratio evaluates the potential risk associated with the agent's actions during task execution. It identifies vulnerabilities, errors, or security concerns that could arise during task handling. A lower risk ratio indicates higher trustworthiness and reliability, while a higher ratio may suggest areas needing improvement to minimize risks and enhance robustness [416].

The implementation of metrics in each GUI agent benchmark might vary depending on the platform and the task formulation. In all tables in this section, we mapped the original metrics used in the benchmarks, which may possess different names, to the categories that we defined above.

### 9.2 Evaluation Measurements

To effectively evaluate GUI agents, various measurement techniques are employed to assess their accuracy and alignment with expected outputs. These measurements validate different aspects of agent performance, ranging from textual and visual correctness to interaction accuracy and system state awareness, using code, models, and even agents as evaluators [26]. Below, we summarize key measurement approaches used in benchmarking GUI agents. Based on these measurements, the evaluation metrics defined beforehand can be calculated accordingly.

1) Text Match: This measurement evaluates whether the text-based outputs of the agent match the expected results. For example, whether a target product name is reached when the agent is browsing an e-commerce website. It can involve different levels of strictness, including:

- Exact Match: Ensures the output perfectly matches the expected result.

- Partial or Fuzzy Match: Allows for approximate matches, which are useful for handling mi r variations such as typos or sy nyms.

- Semantic Similarity: Measures deeper alignment in semantic meaning using techniques like cosine similarity of text embeddings or other semantic similarity measures.

Text Match is widely applied in tasks involving textual selections, data entry, or natural language responses.


<!-- Meanless: JOURNAL OF LATEX CLASS FILES, DECEMBER 2024<br>63 -->

2) Image Match: Image Match focuses on validating whether the agent acts or stops on the expected page (e.g., webpage, app UI), or selects the right image. It involves comparing screenshots, selected graphical elements, or visual outcomes against ground truth images using image similarity metrics or visual question answering (VQA) methods. This measurement is particularly crucial for tasks requiring precise visual identification.

3) Element Match: This measurement checks whether specific widget elements (e.g., those in HTML, DOM, or application UI hierarchies) interacted with by the agent align with the expected elements. These may include:

- HTML Tags and Attributes: Ensuring the agent identifies and interacts with the correct structural elements.

- URLs and Links: Validating navigation-related elements.

- DOM Hierarchies: Confirming alignment with expected DOM structures in dynamic or complex web interfaces.

- UI Controls and Widgets: Verifying interactions with platform-specific controls such as buttons, sliders, checkboxes, dropdown menus, or other GUI components in desktop and mobile applications.

- Accessibility Identifiers: Utilizing accessibility identifiers or resource IDs in mobile platforms like Android and iOS to ensure correct element selection.

- View Hierarchies: Assessing alignment with expected view hierarchies in mobile applications, similar to DOM hierarchies in web applications.

- System Controls and APIs: Ensuring correct interaction with operating system controls or APIs, such as file dialogs, system menus, or tifications in desktop environments.

Element Match ensures robust interaction with user interface components across different platforms during task execution.

4) Action Match: This measurement assesses the accuracy of the agent's actions, such as clicks, scrolls, or keystrokes, by comparing them against an expected sequence. It involves:

- Action Accuracy: Validates that each action (including action type and its arguments) is performed correctly (e.g., clicking the correct button, typing the right input).

- Action Sequence Alignment: Ensures actions occur in the correct order to meet task requirements.

- Location Prediction: Checks that spatial actions, such as mouse clicks or touch gestures, target the intended regions of the interface.

Action Match is vital for evaluating step-wise correctness in task completion.

5) State Information: State Information captures runtime data related to the system's environment during task execution. It provides insights into contextual factors that may influence the agent's behavior, such as:

- Application State: Information about the state of the application being interacted with (e.g., open files, active windows, saved files in given locations).

- System Logs: Detailed logs recording the agent's decisions and interactions.

- Environment Variables: Contextual data about the operating system or runtime environment.

This measurement is valuable for debugging, performance analysis, and ensuring reliability under diverse conditions.

Each of these measurement techniques contributes to a comprehensive evaluation framework, ensuring that the agent $t$ only completes tasks but does so with precision, efficiency, and adaptability. Together, they help build trust in the agent's ability to perform reliably in real-world scenarios while maintaining compliance with policy constraints.

### 9.3 Evaluation Platforms

Evaluating GUI agents requires diverse platforms to capture the varying environments in which these agents operate. The platforms span web, mobile, and desktop environments, each with unique characteristics, challenges, and tools for evaluation. This section summarizes the key aspects of these platforms and their role in benchmarking GUI agents.

1) Web: Web platforms are among the most common environments for GUI agents, reflecting their prevalence in everyday tasks such as browsing, form filling, and data scraping. Key characteristics of web platforms for evaluation include:

- Dynamic Content: Web applications often involve dynamic elements generated through JavaScript, AJAX, or similar tech logies, requiring agents to handle asynchro us updates effectively.

- Diverse Frameworks: The variety of web tech logies (e.g., HTML, CSS, JavaScript frameworks) demands robust agents capable of interacting with a range of interface designs and structures.

- Tools and Libraries: Evaluation often uses tools such as Selenium, Puppeteer, or Playwright to emulate browser interactions, collect runtime information, and compare outcomes against expected results.

- Accessibility Compliance: Metrics like WCAG (Web Content Accessibility Guidelines) adherence can also be evaluated to ensure inclusivity.

2) Mobile: Mobile platforms, particularly Android and iOS, pose unique challenges for GUI agents due to their constrained interfaces and touch-based interactions. Evaluating agents on mobile platforms involves:

- Screen Size Constraints: Agents must adapt to limited screen real estate, ensuring interactions remain accurate and efficient.

- Touch Gestures: Evaluating the agent's ability to simulate gestures such as taps, swipes, and pinches is essential.

- Platform Diversity: Android devices vary significantly in terms of screen sizes, resolutions, and system versions, while iOS offers more standardized conditions.

- Evaluation Tools: Tools like Appium and Espresso (for Android) or XCTest (for iOS) and emulators are commonly used for testing and evaluation.

3) Desktop: Desktop platforms provide a richer and more complex environment for GUI agents, spanning multiple operating systems such as Windows, macOS, and Linux. Evaluations on desktop platforms often emphasize:


<!-- Meanless: JOURNAL OF LATEX CLASS FILES, DECEMBER 2024<br>64 -->

- Application Diversity: Agents must handle a wide range of desktop applications, including productivity tools, web browsers, and custom enterprise software.

- Interaction Complexity: Desktop interfaces often include advanced features such as keyboard shortcuts, drag-and-drop, and context menus, which agents must handle correctly.

- Cross-Platform Compatibility: Evaluations may involve ensuring agents can operate across multiple operating systems and versions.

- Automation Frameworks: Tools such as Windows UI Automation, macOS Accessibility APIs, and Linux's AT-SPI are used to automate and monitor agent interactions.

- Resource Usage: Memory and CPU usage are significant metrics, particularly for long-running tasks or resource-intensive applications.

Each platform presents distinct challenges and opportunities for evaluating GUI agents. Web platforms emphasize scalability and dynamic interactions, mobile platforms focus on touch interfaces and performance, and desktop platforms require handling complex workflows and cross-application tasks. Some benchmarks are cross-platform, requiring agents to be robust, adaptable, and capable of generalizing across different environments.

All the metrics, measurements, and platforms discussed are essential for a comprehensive evaluation of GUI agents across multiple aspects. Most existing benchmarks rely on them for evaluation. In what follows, detail these benchmarks for GUI agents selectively.

### 9.4 Web Agent Benchmarks

Evaluating GUI agents in web environments necessitates benchmarks that capture the complexities and nuances of web-based tasks. Over the years, several benchmarks have been developed, each contributing unique perspectives and challenges to advance the field. We first provide an overview of these benchmarks in Tables 32 33 34 and 35

One of the pioneering efforts in this domain is Mini-WoB++ [145], [146], focusing on assessing reinforcement learning agents on web-based GUI tasks. It introduces realistic interaction scenarios, including clicking, typing, and navigating web elements, and leverages workflow-guided exploration (WGE) to improve efficiency in environments with sparse rewards. Agents are evaluated based on success rates, determined by their ability to achieve final goal states, highlighting adaptability and robustness across various complexities.

Building upon the need for more realistic environments, Mind2Web [212] represents a significant advancement by enabling agents to handle real-world HTML environments rather than simplified simulations. Established after the advent of LLMs [157], it offers a large dataset of over 2,000 tasks spanning multiple domains, presenting challenges from basic actions to complex multi-page workflows. The benchmark emphasizes end-to-end task performance through metrics like Element Accuracy and Task Success Rate, encouraging rigorous evaluation of agents.

Extending Mind2Web's capabilities, MT-Mind2Web 414 introduces conversational web navigation, requiring sophisticated interactions that span multiple turns with both users and the environment. This advanced benchmark includes 720 web navigation conversation sessions with 3,525 instruction and action sequence pairs, averaging five user-agent interactions per session, thereby testing agents' conversational abilities and adaptability.

To further enhance realism, WebArena [412] sets a new standard with its realistic web environment that mimics genuine human interactions. Featuring 812 tasks across multiple domains, it requires agents to perform complex, long-horizon interactions over multi-tab web interfaces. By focusing on functional correctness rather than surface-level matches, WebArena promotes thorough assessment of agents' practical abilities.

Recognizing the importance of multimodal capabilities, VisualWebArena, an extension of WebArena [412], was designed to assess agents on realistic visually grounded web tasks. Comprising 910 diverse tasks in domains like Clas-sifieds, Shopping, and Reddit, it adds new visual functions for measuring open-ended tasks such as visual question answering and fuzzy image matching, thereby challenging agents in multimodal understanding.

Similarly, VideoWebArena [421] focuses on evaluating agents' abilities to comprehend and interact with video content on the web. It presents 74 videos across 2,021 tasks, challenging agents in video-based information retrieval, contextual reasoning, and skill application. This benchmark highlights critical deficiencies in current models, emphasizing the need for advancements in agentic reasoning and video comprehension.

Complementing this, VisualWebBench [213] offers a multimodal benchmark that assesses understanding, OCR, grounding, and reasoning across website, element, and action levels. Spanning 1.5K samples from real-world websites, it identifies challenges such as poor grounding and subpar OCR with low-resolution inputs, providing a crucial evaluation perspective distinct from general multimodal benchmarks.

Beyond the challenges of multimodality, understanding agents' resilience to environmental distractions is crucial. EnvDistraction 422 introduces a benchmark that evaluates the faithfulness of multimodal GUI agents under n-malicious distractions, such as pop-ups and recommendations. The study demonstrates that even advanced agents are prone to such distractions, revealing vulnerabilities that necessitate robust multimodal perception for reliable automation.

Focusing on safety and trustworthiness, ST-WebAgentBench 416 takes a unique approach by emphasizing the management of unsafe behaviors in enterprise settings. It features a human-in-the-loop system and a policy-driven hierarchy, introducing the Completion under Policy (CuP) metric to evaluate agents' compliance with organizational, user, and task-specific policies. This benchmark operates in web environments using BrowserGym 426] and includes 235 tasks with policies addressing various safety dimensions, providing a comprehensive framework for evaluating agents in enterprise scenarios.

Addressing the automation of enterprise software tasks, WorkArena 420 offers a benchmark emphasizing tasks commonly performed within the Service w platform. With 19,912 unique instances across 33 tasks, it highlights the significant performance gap between current state-of-the-art agents and human capabilities in enterprise UI automation, setting a trajectory for future in vation.


<!-- Meanless: JOURNAL OF LATEX CLASS FILES, DECEMBER 2024<br>65 -->

<!-- Media -->

TABLE 32: Overview of web GUI agent benchmarks (Part I).

<table><tr><td>Benchmark</td><td>Platform</td><td>Year</td><td>Live</td><td>Highlight</td><td>Data Size</td><td>Metric</td><td>Measurement</td><td>Link</td></tr><tr><td>MiniWoB++ 145 146]</td><td>Web</td><td>2017</td><td>Yes</td><td>Evaluates agents on basic web interactions like clicking, typing, and form navigation.</td><td>100 web interaction tasks</td><td>Task Rate Success</td><td>Element Match</td><td>https://github.com/Farama% 2DFoundation/ miniwob% 2Dplusplus</td></tr><tr><td>142]</td><td>Web</td><td>2021</td><td>No</td><td>Uses ThingTalk for mapping natural language to web actions, enabling precise web-based task execution in real HTML environments.</td><td>741 instructions</td><td>Task Success Rate</td><td>Text Match, Element Match</td><td>https: <br> //github.com/ <br> xnancy/russ</td></tr><tr><td>WebShop</td><td>Web</td><td>2022</td><td>Yes</td><td>Simulates e-commerce navigation with real-world products, challenging agents with instruction comprehension, multi-page navigation, and strategic exploration.</td><td>12,087 instructions</td><td>Task Success Rate, Step Success Rate"</td><td>Text Match</td><td>https: //webshop-pnlp. github.io/</td></tr><tr><td>Mind2Web 21</td><td>Web</td><td>2023</td><td>No</td><td>Tests adaptability on real-world, dynamic websites across domains.</td><td>2,000 tasks</td><td>Step Success Rate, Task Success Rate</td><td>Element Match, Action Match</td><td>https: //github.com/ OSU-NLP-Group/ Mind2Web</td></tr><tr><td>Mind2Web-Live 349</td><td>Web</td><td>2024</td><td>Yes</td><td>Provides intermediate action tracking for realistic task assessment, along with an updated Mind2Web-Live dataset and tools for an tation.</td><td>542 tasks</td><td>Step Success Rate, Task Success Rate, Efficiency Score</td><td>Element Match, Text Match, trajectory length</td><td>https: //huggingface. co/datasets/ iMeanAI/ Mind2Web-Live</td></tr><tr><td>Mind2Web-Live- <br> Abstracted 278</td><td>Web</td><td>2024</td><td>Yes</td><td>Abstract the descriptions by omitting task-specific details and user input information in Mind2Web-Live, which are more streamlined and less time-consuming to compose.</td><td>104 samples</td><td>Task Success Rate, Efficiency Score</td><td>Text Match, Image Match, Element Match, Path Length</td><td>https: //anonymous. 4open.science/ r/navigate</td></tr><tr><td>WebArena 412</td><td>Web</td><td>2023</td><td>Yes</td><td>Simulates realistic, multi-tab browsing on Docker-hosted websites, focusing on complex, long-horizon tasks that mirror real online interactions.</td><td>812 long-horizon tasks</td><td>Step Success Rate</td><td>Text Match</td><td>https: //webarena.dev/</td></tr><tr><td>VisualWebArena 413]</td><td>Web</td><td>2024</td><td>Yes</td><td>Assesses multimodal agents on visually grounded tasks, requiring both visual and textual interaction capabilities in web environments.</td><td>910 tasks</td><td>Step Success Rate</td><td>Text Match, Image Match</td><td>https: //jykoh.com/vwa</td></tr><tr><td>MT-Mind2Web 414</td><td>Web</td><td>2024</td><td>No</td><td>Introduces conversational web navigation with multiturn interactions, supported by a specialized multi-turn web dataset.</td><td>720 sessions/3525 instructions</td><td>Step Success Rate, Turn Success Rate</td><td>Element Match, Action Match</td><td>https://github.com/magicgh/ self-map</td></tr></table>

<!-- Media -->

BrowserGym [426] builds ecosystem designed for web agent research. It unifies various benchmarks like Mini-WoB(++) [146], WebArena [412], and WorkArena [420] under a single framework, addressing the issue of fragmentation in web agent evaluation. By leveraging standardized observation and action spaces, it enables consistent and reproducible experiments. BrowserGym's extensible architecture make it a vital tool for developing and testing GUI-driven agents powered by LLMs, significantly accelerating in vation in web automation research.

In the realm of interacting with live websites, WebOlym-pus 424 introduces an open platform that enables web agents to interact with live websites through a Chrome extension-based interface. Supporting diverse tasks and integrating a safety monitor to prevent harmful actions, it promotes safer automation of web-based tasks and provides a critical tool for evaluating agent performance in realistic scenarios.

Collectively, these benchmarks have significantly contributed to advancing the evaluation of web-based GUI agents, each addressing different aspects such as realism, multimodality, safety, and enterprise applicability. Their developments reflect the evolving challenges and requirements in creating sophisticated agents capable of complex web interactions.

### 9.5 Mobile Agent Benchmarks

Evaluating GUI agents on mobile platforms presents unique challenges due to the diversity of interactions and the complexity of mobile applications. Several benchmarks have been developed to address these challenges, each contributing to the advancement of mobile agent evaluation. We first provide an analysis for these mobile benchmarks in Tables 36 and 37

An early effort in this domain is PIXELHELP [144], which focuses on grounding natural language instructions to actions on mobile user interfaces. Addressing the significant challenge of interpreting and executing complex, multi-step tasks, PIXELHELP provides a comprehensive dataset pairing English instructions with human-performed actions on a mobile UI emulator. It comprises 187 multi-step instructions across four task categories, offering a robust resource for evaluating models on task accuracy through metrics like Complete Match and Partial Match.


<!-- Meanless: 66 -->

<!-- Media -->

JOURNAL OF LATEX CLASS FILES, DECEMBER 2024

TABLE 33: Overview of web GUI agent benchmarks (Part II).

<table><tr><td>Benchmark</td><td>Platform</td><td>Year</td><td>Live</td><td>Highlight</td><td>Data Size</td><td>Metric</td><td>Measurement</td><td>Link</td></tr><tr><td>MMInA 415</td><td>Web</td><td>2024</td><td>Yes</td><td>Tests multihop, multimodal tasks on real-world websites, requiring agents to handle cross-page information extraction and reasoning for complex tasks.</td><td>1,050 tasks</td><td>Step Success Rate, Task Success Rate</td><td>Text Match, Element Match</td><td>https://mmina.cliangyu.com/</td></tr><tr><td>AutoWebBench 270]</td><td>Web</td><td>2024</td><td>No</td><td>Bilingual web browsing benchmark with 10,000 browsing traces, supporting evaluation across language-specific environments.</td><td>10,000 traces</td><td>Step Success Rate, Efficiency Score</td><td>Element Match, Action Match, Time</td><td>https://github.com/THUDM/ AutoWebGLM</td></tr><tr><td>WorkArena 420]</td><td>Web</td><td>2024</td><td>Yes</td><td>Focuses on real-world enterprise software interactions, targeting tasks frequently performed by knowledge workers</td><td>19,912 unique task instances</td><td>Task Success Rate, Efficiency Score, Completion under Policy, Turn Success Rate</td><td>Element Match, Text Match, Execution-based Validation</td><td>https: //github.com/ ServiceNow/ WorkArena</td></tr><tr><td>VideoWebArena 421]</td><td>Web</td><td>2024</td><td>Yes</td><td>Focuses on long-context multimodal agents using video tutorials for task completion</td><td>74 videos amounting to approximately 4 hours, with 2,021 tasks in total</td><td>Task Success Rate. Intermediate Intent Success Rate, Efficiency Scores</td><td>Element Match, State Information, Exact and Fuzzy Text Matches</td><td>https://github.com/jiang0/ videowebarena</td></tr><tr><td>EnvDistraction 422]</td><td>Web</td><td>2024</td><td>No</td><td>Evaluates the "faithfulness" of multimodal GUI agents by assessing their susceptibility to environmental distractions, such as pop-ups, fake search results, or misleading recommendations</td><td>1,198 tasks</td><td>Task Success Rate</td><td>Text Match, Element Match. State Information</td><td>https://github.com/xbmxb/ EnvDistraction</td></tr><tr><td>WebVLN-v1 347]</td><td>Web</td><td>2024</td><td>No</td><td>Combines navigation and question-answering on shopping sites, integrating visual and textual content for unified web interaction evaluation.</td><td>8,990 paths and 14,825 QA pairs</td><td>Task Success Rate, Efficiency Score</td><td>Element Match, Path Length, Trajectory Length</td><td>https://github.com/WebVLN/ WebVLN</td></tr><tr><td>WEBLINX 348</td><td>Web</td><td>2024</td><td>No</td><td>Focuses on conversational navigation, requiring agents to follow multi-turn user instructions in realistic, dialogue-based web tasks.</td><td>100k interactions</td><td>Turn Success Rate</td><td>Element Match, Text Match, Action Match</td><td>https: //mcgill-nlp. github.io/ weblinx/</td></tr><tr><td>ST-WebAgentBench 416]</td><td>Web</td><td>2024</td><td>Yes</td><td>Evaluates policy-driven safety in web agents, using the Completion under Policy metric to ensure compliance in enterprise-like environments.</td><td>235 tasks</td><td>Task Success Rate, Completion under Policy (CuP), Risk Ratio</td><td>Element Match, Action Match. Text Match</td><td>https: //sites.google. com/view/ st-webagentbench home</td></tr><tr><td>CompWoB 417]</td><td>Web</td><td>2023</td><td>No</td><td>Tests agents on sequential, compositional tasks that require state management across multiple steps, simulating real-world automation scenarios.</td><td>50 compositional tasks</td><td>Task Success Rate</td><td>Element Match</td><td>https: //github.com/ google-research google-research tree/master/ compositional rl/compwob</td></tr></table>

<!-- Media -->

Building upon the need for systematic evaluation, AN-DROIDLAB [363] establishes a comprehensive framework for Android-based auto mous agents. It introduces both an action space and operational modes that support consistent evaluations for text-only and multimodal models. By providing XML and SoM operation modes, ANDROIDLAB allows LLMs and LMMs to simulate real-world interactions in equivalent environments. The benchmark includes 138 tasks across nine apps, encompassing typical Android functionalities, and evaluates agents using metrics such as Success Rate and Reversed Redundancy.

To further challenge agents in handling both API and UI operations, Mobile-Bench [442] offers an in vative approach by combining these elements within a realistic Android environment. Its multi-app setup and three distinct task categories test agents' capabilities in handling simple and complex mobile interactions, pushing beyond traditional single-app scenarios. The evaluation leverages CheckPoint metrics, assessing agents at each key action step, providing insights into planning and decision-making skills.

Emphasizing safety in mobile device control, MobileSafe-tyBench 443 provides a structured evaluation framework that prioritizes both helpfulness and safety. It rigorously tests agents across common mobile tasks within an Android emulator, focusing on layered risk assessment, including legal compliance and privacy. A distinctive feature is its indirect prompt injection test to probe agent robustness. The evaluation ensures agents are scored on practical success while managing risks, advancing research in LLM reliability and secure auto mous device control.

Expanding the scope to multiple languages and application scenarios, SPA-BENCH 444 introduces an extensive benchmark for smartphone agents. It assesses both single-app and cross-app tasks in a plug-and-play framework that supports seamless agent integration. With a diverse task collection across Android apps, including system and third-party apps, SPA-BENCH offers a realistic testing environment measuring agent capabilities in understanding UIs and handling app navigation through metrics like success rate, efficiency, and resource usage.


<!-- Meanless: JOURNAL OF LATEX CLASS FILES, DECEMBER 2024<br>67 -->

<!-- Media -->

TABLE 34: Overview of web GUI agent benchmarks (Part III).

<table><tr><td>Benchmark</td><td>Platform</td><td>Year</td><td>Live</td><td>Highlight</td><td>Data Size</td><td>Metric</td><td>Measurement</td><td>Link</td></tr><tr><td>TURKING BENCH 418]</td><td>Web</td><td>2024</td><td>Yes</td><td>Uses natural HTML tasks from crowdsourcing to assess interaction skills with real-world web layouts and elements.</td><td>32.2K instances</td><td>ask Success Rate</td><td>Text Match, Element Match, Image Match</td><td>https: //turkingbench. github.io</td></tr><tr><td>VisualWebBench 213]</td><td>Web</td><td>2024</td><td>No</td><td>Provides a fine-grained assessment of multimodal large language models (MLLMs) on web-specific tasks</td><td>1,534 instances from 139 real websites across 87 subdomains</td><td>Task Success Rate, Turn Success Rate, Efficiency Metrics</td><td>Text Match, Image Match, Element Match, Action Match</td><td>https:// visualwebbench. github.io/</td></tr><tr><td>WONDERBREAD 423]</td><td>Web</td><td>2024</td><td>No</td><td>Focuses on business process management (BPM) tasks like documentation, knowledge transfer, and process improvement</td><td>2,928 human demonstrations across 598 distinct workflows</td><td>Task Success Rate, Step Success Rate. Efficiency Score, Completion under Policy</td><td>Text Match, Action Match, State Information</td><td>https: //github.com/ HazyResearch/ wonderbread</td></tr><tr><td>WebOlympus 424]</td><td>Web</td><td>2024</td><td>Yes</td><td>An open platform for web agents that simplifies running demos, evaluations, and data collection for web agents on live websites</td><td>50 tasks</td><td>Task Success e. Step Success Rate</td><td>Action Match</td><td>/</td></tr><tr><td>BrowserGym 426]</td><td>Web</td><td>2024</td><td>Yes</td><td>Provides a unified, extensible, and open-source environment for evaluating web agents with standardized APIs and observations.</td><td>Benchmarks include Mini-WoB(++) with 125 tasks, WebArena with 812 tasks, and WorkArena with up to 341 tasks per level.</td><td>Task Success Rate, Step Success Rate, Turn Success Rate, Efficiency Metrics.</td><td>Text-based matching and element match.</td><td>https: //github.com/ ServiceNow/ BrowserGym</td></tr><tr><td>WebWalkerQA 427]</td><td>Web</td><td>2025</td><td>Yes</td><td>Benchmarks the capacity of LLMs to handle deep. structured, and realistic web-based navigation and reasoning tasks.</td><td>680 high-quality QA pairs.</td><td>Task Success Rate, Efficiency Score.</td><td>Text Match, Action Match.</td><td>https: //github.com/ Alibaba-NLP/ WebWalker</td></tr><tr><td>WebGames 428</td><td>Web</td><td>2025</td><td>Yes</td><td>A comprehensive benchmark designed to evaluate the capabilities of general-purpose web-browsing AI agents through 50+ interactive challenges. It uniquely provides a hermetic testing environment with verifiable ground-truth solutions.</td><td>50+ challenges</td><td>Task \$k Success Rate</td><td>Action Match</td><td>https: //github.com/ convergence-ai/ webgames</td></tr></table>

<!-- Media -->

Focusing on efficient and user-friendly evaluation, Mo-bileAgentBench 446 presents a benchmark tailored for agents on Android devices. It offers a fully auto mous testing process, leveraging final UI state matching and real-time app event tracking. With 100 tasks across 10 open-source Android applications categorized by difficulty, it accommodates multiple paths to success, enhancing reliability and applicability. Comprehensive metrics, including task success rate, efficiency, latency, and token cost, provide insights into agent performance.

Complementing these efforts, LlamaTouch [445] introduces a benchmark and testbed for mobile UI task automation in real-world Android environments. Emphasizing essential state an tation, it enables precise evaluation of tasks regardless of execution path variability or dynamic UI elements. With 496 tasks spanning 57 unique applications, LlamaTouch demonstrates scalability and fidelity through advanced matching techniques, integrating pixel-level screenshots and textual screen hierarchies, reducing false negatives and supporting diverse task complexities.

Zhao et al., introduce GTArena [447], a formalized framework and benchmark designed to advance auto mous GUI testing agents. GTArena provides a standardized evaluation environment tailored for multimodal large language models. Central to its design is the vel Transition Tuple data structure, which systematically captures and analyzes GUI defects. The benchmark assesses three core tasks—test intention generation, task execution, and defect detection—using a diverse dataset comprising real-world, artificially injected, and synthetic defects, establishing GTArena as a pioneering benchmark for GUI testing agents.

Collectively, these benchmarks have significantly advanced the evaluation of mobile-based GUI agents, addressing challenges in task complexity, safety, efficiency, and scalability. Their contributions are instrumental in developing more capable and reliable agents for mobile platforms.


<!-- Meanless: JOURNAL OF LATEX CLASS FILES, DECEMBER 2024<br>68 -->

<!-- Media -->

TABLE 35: Overview of web GUI agent benchmarks (Part IV).

<table><tr><td>Benchmark</td><td>Platform</td><td>Year</td><td>Live</td><td>Highlight</td><td>Data Size</td><td>Metric</td><td>Measurement</td><td>Link</td></tr><tr><td>SafeArena 429]</td><td>Web</td><td>2025</td><td>Yes</td><td>The first benchmark specifically designed to evaluate the deliberate misuse of web agents by testing their ability to complete both safe and harmful tasks.</td><td>500 tasks</td><td>Task Success Rate, Completion under Policy, Risk Ratio</td><td>Text Match, State Information</td><td>https: <br> //safearena. <br> github.io</td></tr><tr><td>WABER [430]</td><td>Web</td><td>2025</td><td>Yes</td><td>Introduces two new evaluation metrics—Efficiency and Reliability—that go beyond standard success rate measurements</td><td>655 tasks</td><td>Task Success Rate, Efficiency Score</td><td>Action Match, State Information</td><td>https: //github.com/ SumanKNath/ WABER</td></tr><tr><td>Online-Mind2Web 431]</td><td>Web</td><td>2025</td><td>Yes</td><td>A real-world online evaluation benchmark designed to reflect actual user interactions with live web interfaces</td><td>300 tasks from 136 websites</td><td>Task Success Rate, Efficiency Score</td><td>Image Match, Action Match, State Information, LLM-as-a-Judge Evaluation</td><td>https: //github.com/ OSU-NLP-Group/ Online-Mind2Web</td></tr><tr><td>AgentDAM 432]</td><td>Web</td><td>2025</td><td>Yes</td><td>The first benchmark to evaluate privacy leakage risks in multimodal, realistic web environments using agentic models</td><td>246 human-annotated test cases</td><td>Task Success Rate, Risk Ratio</td><td>Action Match, Text Match</td><td>https: //github.com/ facebookresearch/ ai-agent-privacy</td></tr><tr><td>AgentRewardBenchWeb <br> 433</td><td></td><td>2025</td><td>No</td><td>The first benchmark to rigorously evaluate LLM-based judges against human expert annotations across multiple web agent tasks</td><td>1,302 trajectories, 351 tasks</td><td>Task Success Rate, Completion under Policy</td><td>Image Match, Element/State Match</td><td>https:// agent-reward-bench github.io</td></tr><tr><td>RealWebAssist 434</td><td>Web</td><td>2025</td><td>No</td><td>The first benchmark to evaluate long-horizon web assistance using real-world users' sequential instructions expressed in natural and often ambiguous language</td><td>1,885 instructions</td><td>Task Success Rate, Step Success Rate. Efficiency Score</td><td>Action Match</td><td>https: //scai.cs.jhu. edu/projects/ RealWebAssist/</td></tr><tr><td>435]</td><td>Web</td><td>2025</td><td>Yes</td><td>Fully deterministic, high-fidelity replicas of real-world websites (e.g., Airbnb, Amazon, Gmail), enabling safe, reproducible, and configurable testing for multi-turn GUI-based agents</td><td>112 tasks across 11 deterministic websites</td><td>Task Success Rate</td><td>Text Match, Action Match, State Information Match</td><td>https: //github.com/ agi-inc/agisdk</td></tr><tr><td>BEARCUBS 436</td><td>Web</td><td>2025</td><td>Yes</td><td>Emphasizes interaction with live web pages and includes multimodal tasks (e.g., video. audio, 3D) that cannot be solved by text-only methods, addressing limitations of prior benchmarks relying on static or simulated environments</td><td>111 questions</td><td>Task Success Rate, Efficiency Score</td><td>Text Match, Action Match</td><td>https: //bear-cubs. github.io</td></tr><tr><td>WASP 437]</td><td>Web</td><td>2025</td><td>Yes</td><td>The first end-to-end benchmark for evaluating web agents' security under realistic prompt injection attacks, simulating attacker capabilities in live sandboxed web environments</td><td>84 test cases</td><td>Task Success Rate. Completion under Policy, Risk Ratio</td><td>Action Match, State Information</td><td>/</td></tr></table>

<!-- Media -->

### 9.6 Computer Agent Benchmarks

Evaluating GUI agents on desktop computers involves diverse applications and complex workflows. Several benchmarks have been developed to assess agents' capabilities in these environments, each addressing specific challenges and advancing the field. We overview benchmarks for computer GUI agents in Table 38

An early benchmark in this domain is Act2Cap [327], which focuses on capturing and narrating GUI actions in video formats using a cursor as a pivotal visual guide. Act2Cap emphasizes the detailed nuances of GUI interactions, particularly cursor-based actions like clicks and drags, essential for advancing automation capabilities in GUI-intensive tasks. It includes a substantial dataset of 4,189 samples across various Windows GUI environments, employing metrics based on element-wise Intersection over Union to evaluate semantic accuracy and temporal and spatial precision.

To provide a scalable and genuine computer environment for multimodal agents, OSWorld [419] introduces a pioneering framework that supports task setup, execution-based evaluation, and interactive learning across multiple operating systems, including Ubuntu, Windows, and macOS. OSWorld serves as a unified environment that mirrors the complexity and diversity of real-world computer use, accommodating arbitrary applications and open-ended computer tasks. It includes a comprehensive suite of 369 tasks on Ubuntu and 43 tasks on Windows, utilizing execution-based evaluation metrics like success rate for rigorous assessment.

Building on OSWorld, WindowsArena [453] adapts the framework to create over 150 diverse tasks specifically for the Windows operating system. Focusing on multi-modal, multi-step tasks, it requires agents to demonstrate abilities in planning, screen understanding, and tool usage within a real Windows environment. Addressing the challenge of slow evaluation times, WindowsArena enables parallelized deployment in the Azure cloud, drastically reducing evaluation time and allowing for comprehensive testing across various applications and web domains.


<!-- Meanless: 69 -->

<!-- Media -->

JOURNAL OF LATEX CLASS FILES, DECEMBER 2024

TABLE 36: Overview of mobile GUI agent benchmarks (Part I).

<table><tr><td>Benchmark</td><td>Platform</td><td>Year</td><td>Live</td><td>Highlight</td><td>Data Size</td><td>Metric</td><td>Measurement</td><td>Link</td></tr><tr><td>AndroidEnv 263</td><td>Android</td><td>2021</td><td>Yes</td><td>Provides an open-source platform based on the Android ecosystem with over 100 tasks across approximately 30 apps, focusing on reinforcement learning for various Android interactions.</td><td>100+ tasks</td><td>NA</td><td>NA</td><td>https: //github.com/ google-deepmind/ android_env</td></tr><tr><td>PIXELHELP 144]</td><td>Android</td><td>2020</td><td>No</td><td>Includes a corpus of natural language instructions paired with UI actions across four task categories, aiding in grounding language to UI interactions.</td><td>187 multistep instructions</td><td>Step Success Rate</td><td>Element Match, Action Match</td><td>https: //github.com/ google-research google-research tree/master/ seq2act</td></tr><tr><td>Mobile-Env 438]</td><td>Android</td><td>2024</td><td>Yes</td><td>Comprehensive toolkit for Android GUI benchmarks to enable controlled evaluations of real-world app interactions.</td><td>224 tasks</td><td>Task Success Rate, Step Success Rate</td><td>Text Match, Element Match, Image Match, State Information</td><td>https://github.com/X-LANCE/ Mobile-Env</td></tr><tr><td>B-MOCA [439]</td><td>Android</td><td>2024</td><td>Yes</td><td>Benchmarks mobile device control agents on realistic tasks, incorporating UI layout and language randomization to evaluate generalization capabilities.</td><td>131 tasks</td><td>ask Success Rate</td><td>Element Match, State Information</td><td>https://b-moca.github.io/</td></tr><tr><td>AndroidWorld 440]</td><td>Android</td><td>2024</td><td>Yes</td><td>Offers a dynamic Android environment, allowing for diverse natural language instruction testing.</td><td>116 tasks</td><td>Task sk Success Rate</td><td>State Information</td><td>https: //github.com/ google-research android world</td></tr><tr><td>Mobile-Eval 158]</td><td>Android</td><td>2024</td><td>Yes</td><td>Benchmark based on mainstream Android apps, and designed to test common mobile interactions.</td><td>30 instructions</td><td>Task Success Rate, Step Success Rate, Efficiency Score</td><td>Text Match, Path Length</td><td>https://github.com/X-PLUG/ MobileAgent</td></tr><tr><td>DroidTask [156]</td><td>Android</td><td>2024</td><td>Yes</td><td>Android Task Automation benchmark supports exploration and task recording in real apps with corresponding GUI action traces.</td><td>158 tasks</td><td>Step Success Rate, Task Success Rate</td><td>Element Match, Action Match</td><td>https: //github.com/ MobileLLM/ AutoDroid</td></tr><tr><td>AITW 358</td><td>Android</td><td>2023</td><td>No</td><td>A large-scale dataset, which is partly inspired by PIXEL-HELP, covering diverse Android interactions.</td><td>715,142 episodes</td><td>Task Success Rate, Step Success Rate</td><td>Action Match</td><td>https: //github.com/ google-research google-research tree/master/ android in the wild</td></tr><tr><td>AndroidArena 441]</td><td>Android</td><td>2024</td><td>Yes</td><td>Focuses on daily cross-app and constrained tasks within the Android ecosystem, providing single-app and multi-app interaction scenarios.</td><td>221 tasks</td><td>Task Success Rate, Step Success Rate, Efficiency Score</td><td>Action Match, Path Length</td><td>https: //github.com/ AndroidArenaAgeni AndroidArena</td></tr><tr><td>ANDROIDLAB 363]</td><td>Android</td><td>2024</td><td>Yes</td><td>Provides a structured evaluation framework with 138 tasks across nine apps, supporting both text-only and multimodal agent evaluations on Android.</td><td>138 tasks</td><td>Task Success Rate, Step Success Rate Efficiency Score</td><td>Element Match, Image Match</td><td>https://github.com/THUDM/ Android-Lab</td></tr><tr><td>GTArena 447]</td><td>Mobile applications</td><td>2024</td><td>No</td><td>Introduces a Transition Tuple for GUI defects, enabling large-scale defect dataset creation and reproducible, end-to-end automated testing.</td><td>10,000+ GUI display and GUI interactions</td><td>Task Success Rate, Step Success Rate</td><td>Text Match, Element Match, Action Match</td><td>https: //github.com/ ZJU-ACES-ISE/ ChatUITest</td></tr></table>

<!-- Media -->

Focusing on office automation tasks, OF-FICEBENCH 455] introduces a groundbreaking framework for benchmarking LLM agents in realistic office workflows. Simulating intricate workflows across multiple office applications like Word, Excel, and Email within a Linux Docker environment, it evaluates agents' proficiency in cross-application automation. The benchmark challenges agents with complex tasks at varying difficulty levels, demanding adaptability to different complexities and use cases. Customized metrics assess operation accuracy and decision-making, providing critical insights into agents' capabilities in managing multi-application office scenarios.

Addressing the automation of data science and engineering workflows, Spider2-V [454] offers a distinctive benchmark. It features 494 real-world tasks across 20 enterprise-level applications, spanning the entire data science workflow from data warehousing to visualization. Assessing agents' abilities to handle both code generation and complex GUI interactions within authentic enterprise software environments on Ubuntu, it employs a multifaceted evaluation method that includes information-based validation, file-based comparison, and execution-based verification.


<!-- Meanless: JOURNAL OF LATEX CLASS FILES, DECEMBER 2024<br>70 -->

<!-- Media -->

TABLE 37: Overview of mobile GUI agent benchmarks (Part II).

<table><tr><td>Benchmark</td><td>Platform</td><td>Year</td><td>Live</td><td>Highlight</td><td>Data Size</td><td>Metric</td><td>Measurement</td><td>Link</td></tr><tr><td>A3 448</td><td>Mobile Android</td><td>2025</td><td>Yes</td><td>Introduces a novel business-level LLM-based evaluation process, significantly reducing human labor and coding expertise requirements.</td><td>201 tasks across 21 widely used apps.</td><td>Task Success Rate.</td><td>Element Match, Action Match.</td><td>https: //yuxiangchai. github.io/ Android-Agent-Arena</td></tr><tr><td>LlamaTouch 445</td><td>Mobile Android</td><td>2024</td><td>Yes</td><td>Enables faithful and scalable evaluations for mobile UI task automation by matching task execution traces against annotated essential states</td><td>496 tasks covering 57 unique Android applications</td><td>Task Success Rate, Step Success Rate, Efficiency Score</td><td>Text Action Match, State Information Match</td><td>https: <br> //github.com/ <br> LlamaTouch/ <br> LlamaTouch</td></tr><tr><td>MobileAgentBenc 446]</td><td>Mobile An droid</td><td>2024</td><td>Yes</td><td>Provides a fully autonomous evaluation process on real Android devices and flexibility in judging success conditions across multiple paths to completion</td><td>100 tasks across 10 open-source Android applications</td><td>Task Success Rate, Efficiency Score, Latency, Token Cost</td><td>State Information (UI State Matching)</td><td>https:// mobileagentbench. github.io/</td></tr><tr><td>Mobile-Bench 442]</td><td>Android</td><td>2024</td><td>Yes</td><td>Supports both UI and API-based actions in multi-app scenarios, testing agents on single and multi-task structures with a checkpoint-based evaluation approach.</td><td>832 entries (200+ tasks)</td><td>Task Success Rate, Step Success Rate, Efficiency Score</td><td>Action Match, Path Length</td><td>https://github com/XiaoMi/ MobileBench</td></tr><tr><td>Mobile Safety Bench 443]</td><td>Android</td><td>2024</td><td>Yes</td><td>Prioritizes safety evaluation in mobile control tasks, with distinct tasks focused on helpfulness, privacy, and legal compliance.</td><td>100 tasks</td><td>Task Success Rate, Risk Mitigation Success</td><td>Action Match with Safety Considered, Element Match, State Information</td><td>https:// mobilesafetybench. github.io/</td></tr><tr><td>SPA-BENCH 444]</td><td>Android</td><td>2024</td><td>Yes</td><td>Extensive evaluation framework supporting single-app and cross-app tasks in English and Chinese, providing a plug-and-play structure for diverse task scenarios.</td><td>340 tasks</td><td>Task Success Rate, Step Success Rate, Efficiency Score</td><td>Action Match, State Information, Time Spent, API Cost</td><td>https: <br> //spa-bench. <br> github.io</td></tr><tr><td>SPHINX 449]</td><td>Android</td><td>2025</td><td>Yes</td><td>Provides a fully automated benchmarking suite and introduces a multi-dimensional evaluation framework.</td><td>284 common tasks.</td><td>Task Success Rate, Efficiency Score Completion under Policy, Turn Success Rate.</td><td>Text Match, Image Match, Element Match, Action Match.</td><td>7</td></tr><tr><td>AEIA-MN 45 450]</td><td>Mobile Android</td><td>2025</td><td>Yes</td><td>Introduces the Active Environment Injection Attack (AEIA) framework that actively manipulates environmental elements (e.g., notifications) in mobile operating systems to mislead multimodal LLM-powered agents.</td><td>61 tasks (Android-World) 45 tasks (AppAgent)</td><td>Task Success Rate, Risk Ratio, Efficiency Score</td><td>Text Match, State Information, Action Match</td><td>/</td></tr><tr><td>AutoEval 451]</td><td>Mobile Ar droid</td><td>2025</td><td>Yes</td><td>Introduces a fully autonomous evaluation framework for mobile agents, eliminating the need for manual task reward signal definition and extensive evaluation code development.</td><td>93 tasks</td><td>Task Success Rate</td><td>Action Match, State Information</td><td>/</td></tr><tr><td>LearnGUI 321]</td><td>Mobile Android</td><td>2025</td><td>Yes</td><td>The first benchmark to systematically study few-shot demonstration-based learning in mobile GUI agents, featuring both offline and online task environments</td><td>Offline: 2,252 tasks with k-shot variants across 44 apps; Online: 101 interactive tasks across 20 apps</td><td>Task Success Rate</td><td>Action Match</td><td>https: //lgy0404.github. io/LearnAct</td></tr></table>

<!-- Media -->

In the realm of productivity software, AssistGUI [109] provides a pioneering framework for evaluating agents' capabilities. It introduces an Actor-Critic Embodied Agent framework capable of complex hierarchical task planning, GUI parsing, and action generation. The dataset includes diverse tasks across design, office work, and system settings, supported by project files for reproducibility. By emphasizing outcome-driven evaluation with pixel-level precision and procedural adherence, AssistGUI highlights the potential and limitations of current LLM-based agents in managing intricate desktop software workflows.

WorldGUI [456] is a benchmark designed to evaluate GUI agents under dynamic conditions on the Windows platform. Unlike previous static benchmarks, it introduces varied initial states to simulate real-world interactions across both desktop and web applications. Rather than always starting from a fixed default state, agents must adapt to changing UI layouts, user interactions, system settings, and pre-existing conditions, requiring robust adaptability to perform effectively. The benchmark comprises 315 tasks spanning 10 popular software applications and incorporates instructional videos, project files, and multiple pre-action scenarios, providing a comprehensive and realistic evaluation framework for assessing an agent's ability to handle complex task execution.


<!-- Meanless: JOURNAL OF LATEX CLASS FILES, DECEMBER 2024<br>71 -->

<!-- Media -->

TABLE 38: Overview of computer GUI agent benchmarks.

<table><tr><td>Benchmark</td><td>Platform</td><td>Year</td><td>Live</td><td>Highlight</td><td>Data Size</td><td>Metric</td><td>Measurement</td><td>Link</td></tr><tr><td>OSWorld 419]</td><td>Linux, Windows, ma-cOS</td><td>2024</td><td>Yes</td><td>Scalable, real computer environment for multimodal agents, supporting task setup, execution-based evaluation, and interactive learning across Ubuntu, Windows, and macOS.</td><td>369 Ubuntu tasks, 43 Windows tasks</td><td>Task Rate sk Success</td><td>Execution-based State Information (such 1 internal file interpretation, permission management)</td><td>https://os-world.github.io/</td></tr><tr><td>Windows Agent Arena [453]</td><td>Windows</td><td>2024</td><td>Yes</td><td>Adaptation of OSWorld focusing exclusively on the Windows OS with diverse multi-step tasks, enabling agents to use a wide range of applications and tools.</td><td>154 tasks</td><td>Task Success Rate</td><td>Same as OS-World, scalable with cloud parallelization</td><td>https: //microsoft. github.io/ WindowsAgentArena</td></tr><tr><td>OmniACT 459]</td><td>macOS, Linux, Windows</td><td>2024</td><td>No</td><td>Assesses agents' capability to generate executable programs for computer tasks across desktop and web applications in various OS environments, prioritizing multimodal challenges.</td><td>9,802 data points</td><td>Task Success Rate, Step Success Rate</td><td>Action Match</td><td>https: //huggingface. co/datasets/ Writer/omniact</td></tr><tr><td>VideoGUI 460]</td><td>Windows</td><td>2024</td><td>No</td><td>Focuses on visual-centric tasks from instructional videos, emphasizing planning and action precision in applications like Adobe Photoshop and Premiere Pro</td><td>178 tasks, 463 subtasks</td><td>Task sk Success Rate</td><td>State Information, Action Match</td><td>https://showlab.github.io/ videogui</td></tr><tr><td>Spider2-V 454</td><td>Linux</td><td>2024</td><td>Yes</td><td>Benchmarks agents across data science and engineering workflows in authentic enterprise software environments, covering tasks from data ingestion to visualization.</td><td>494 tasks</td><td>Task Rate Success</td><td>Action Match, State Information</td><td>https: //spider2-v. github.io</td></tr><tr><td>Act2Cap 327]</td><td>Windows</td><td>2024</td><td>Yes</td><td>Emphasizes GUI action narration using cursor-based prompts in video format, covering a variety of GUI interactions like clicks, typing, and dragging.</td><td>4,189 samples</td><td>Step Success Rate</td><td>Element Match</td><td>https://showlab.github.io/ GUI-Narrator</td></tr><tr><td>OFFICEBENCH 455]</td><td>Linux</td><td>2024</td><td>Yes</td><td>Tests cross-application automation in office work-flows with complex multistep tasks across applications like Word and Excel, assessing operational integration in realistic scenarios.</td><td>300 tasks</td><td>Task Success Rate</td><td>Action Match, Text Match, State Information</td><td>https://github.com/zlwang-cs/ OfficeBench</td></tr><tr><td>AssistGUI [109]</td><td>Windows</td><td>2024</td><td>Yes</td><td>The first benchmark focused on task-oriented desktop GUI automation</td><td>100 tasks from 9 popular applications</td><td>Task Success Rate, Efficiency Score</td><td>Element Match, Action Match</td><td>https://showlab github.io/ assistgui/</td></tr><tr><td>WorldGUI 456]</td><td>Windows</td><td>2025</td><td>Yes</td><td>First GUI benchmark designed to evaluate dynamic GUI interactions by incorporating various initial states.</td><td>315 total tasks from 10 Windows applications</td><td>Task Success Rate, Efficiency Score</td><td>Image Match, Element Match. Action Match</td><td>/</td></tr><tr><td></td><td>Desktop (Windows, Linux)</td><td>2025</td><td>No</td><td>The first large-scale benchmark specifically designed for desktop GUI agents</td><td>8,227 query-label pairs in total</td><td>Rate <icon/> SECESS</td><td>Action Match, Text Match</td><td>https://uivision.github.io</td></tr><tr><td>Computer Agent Arena [458]</td><td>Windows, Ubuntu, macOS</td><td>2025</td><td>Yes</td><td>The first large-scale, open-ended evaluation platform for multimodal LLM-based agents in real desktop computing environments</td><td>User-proposed tasks</td><td>Task Success Rate</td><td>Human evalua tors</td><td>https://arena.<br> xlang.ai/</td></tr></table>

<!-- Media -->

Computer Agent Arena [458] presents a new paradigm for benchmarking LLM-based GUI agents through live, user-configured desktop environments. Unlike traditional static datasets, it provides an interactive cloud-based infrastructure where agents are evaluated on tasks spanning web browsing, programming, and productivity using real applications like Google Docs, VSCode, and Slack. Its innovation lies in using head-to-head agent comparisons, human judgment, and Elo-based ranking to evaluate general-purpose digital agents in realistic settings. The benchmark supports Windows and Ubuntu, with MacOS support planned, and allows customized task scenarios with diverse software and website setups. By enabling crowdsourced evaluations and planning open-source releases, it fosters community-driven improvements and robust comparisons.

Collectively, these benchmarks provide comprehensive evaluation frameworks for GUI agents on desktop platforms, addressing challenges in task complexity, cross-application automation, scalability, and fidelity. Their contributions are instrumental in advancing the development of sophisticated agents capable of complex interactions in desktop environments.


<!-- Meanless: JOURNAL OF LATEX CLASS FILES, DECEMBER 2024<br>72 -->

### 9.7 Cross-Platform Agent Benchmarks

To develop GUI agents capable of operating across multiple platforms, cross-platform benchmarks are essential. These benchmarks challenge agents to adapt to different environments and interfaces, evaluating their versatility and robustness. We provide an overview of benchmarks for cross-platform GUI agents in Tables 39.

Addressing this need, VisualAgentBench (VAB) 374 represents a pioneering benchmark for evaluating GUI and multimodal agents across a broad spectrum of realistic, interactive tasks. Encompassing platforms such as Web (WebArena-Lite [412]), Android (VAB-Mobile [363]), and game environments, VAB focuses on vision-based interaction and high-level decision-making tasks. The benchmark employs a multi-level data collection strategy involving human demonstrations, program-based solvers, and model bootstrapping. Evaluation metrics concentrate on success rates, ensuring comprehensive performance assessments in tasks like navigation and content modification, thereby filling a significant gap in benchmarking standards for GUI-based LLM agents.

Complementing this, CRAB 461 introduces an in vative benchmark by evaluating multimodal language model agents in cross-environment interactions. It uniquely supports seamless multi-device task execution, evaluating agents in scenarios where tasks span both Ubuntu Linux and Android environments. By introducing a graph-based evaluation method that breaks down tasks into sub-goals and accommodates multiple correct paths to completion, CRAB provides a nuanced assessment of planning, decision-making, and adaptability. Metrics such as Completion Ratio, Execution Efficiency, Cost Efficiency, and Success Rate offer comprehensive insights into agent performance.

Focusing on GUI grounding for cross-platform visual agents, ScreenSpot [25] offers a comprehensive benchmark emphasizing tasks that rely on interpreting screenshots rather than structured data. ScreenSpot includes over 600 screen-shots and 1,200 diverse instructions spanning mobile (iOS, Android), desktop (macOS, Windows), and web platforms. It evaluates click accuracy and localization precision by measuring how effectively agents can identify and interact with GUI elements through visual cues alone. By challenging models with a wide variety of UI elements, ScreenSpot addresses real-world complexities, making it an essential resource for evaluating visual GUI agents across varied environments.

These cross-platform benchmarks collectively advance the development of GUI agents capable of operating seamlessly across multiple platforms. By providing comprehensive evaluation frameworks, they are instrumental in assessing and enhancing the versatility and adaptability of agents in diverse environments.

### 9.8 Takeaways

The evolution of GUI agent benchmarks reflects a broader shift towards more realistic, interactive, and comprehensive evaluation environments. This section highlights key trends and future directions in the benchmarking of LLM-brained GUI agents.

1) Towards More Interactive and Realistic Environments: Recent advancements in GUI agent benchmarking emphasize the transition from synthetic scenarios to more interactive and realistic environments. This shift is evident in the use of simulators, Docker containers, and real-world applications to create "live" environments that better mimic genuine user interactions. Such environments t only provide a more accurate assessment of agent capabilities but also pose new challenges in terms of performance and robustness.

2) Cross-Platform Benchmarks: The emergence of cross-platform benchmarks that encompass mobile, web, and desktop environments represents a significant step towards evaluating the generalizability of GUI agents. However, these benchmarks introduce fundamental challenges unique to each platform. A unified interface for accessing platform-specific information, such as HTML and DOM structures, could substantially streamline the benchmarking process and reduce implementation efforts. Future work should focus on standardizing these interfaces to facilitate seamless agent evaluation across diverse environments.

3) Increased Human Interaction and Realism: There is a growing trend towards incorporating more human-like interactions in benchmarks, as seen in multi-turn and conversational scenarios. These setups mirror real-world use cases more closely, thereby providing a rigorous test of an agent's ability to handle dynamic, iterative interactions. As GUI agents become more sophisticated, benchmarks must continue to evolve to include these nuanced interaction patterns, ensuring agents can operate effectively in complex, human-centric environments.

4) Scalability and Automation Challenges: Scalability remains a significant concern in benchmarking GUI agents. The creation of realistic tasks and the development of evaluation methods for individual cases often require substantial human effort. Automation of these processes could alleviate some of the scalability issues, enabling more extensive and efficient benchmarking. Future research should explore automated task generation and evaluation techniques to enhance scalability.

5) Emphasis on Safety, Privacy, and Compliance: There is a table trend towards evaluating GUI agents on safety, privacy, and compliance metrics. These considerations are increasingly important as agents are integrated into sensitive and regulated domains. Encouraging this trend will help ensure that agents t only perform tasks effectively but also adhere to necessary legal and ethical standards. Future benchmarks should continue to expand on these dimensions, incorporating evaluations that reflect real-world compliance and data security requirements.

The landscape of GUI agent benchmarking is rapidly evolving to meet the demands of increasingly complex and interactive environments. By embracing cross-platform evaluations, fostering human-like interactions, addressing scalability challenges, and prioritizing safety and compliance, the community can pave the way for the next generation of sophisticated GUI agents. Continued in vation and collaboration will be essential in refining benchmarks to ensure they accurately capture the multifaceted capabilities of modern agents, ultimately leading to more intuitive and effective human-computer interactions.


<!-- Meanless: JOURNAL OF LATEX CLASS FILES, DECEMBER 2024<br>73 -->

<!-- Media -->

TABLE 39: Overview of cross-platform GUI agent benchmarks.

<table><tr><td>Benchmark</td><td>Platform</td><td>Year</td><td>Live</td><td>Highlight</td><td>Data Size</td><td>Metric</td><td>Measurement</td><td>Link</td></tr><tr><td>VisualAgent Bench [374]</td><td>Web, Android, Game, Virtual Embodied</td><td>2024</td><td>Yes</td><td>First benchmark designed for visual foundation agents across GUI and multimodal tasks, focusing on vision-centric interactions in Android, web, and game environments.</td><td>4,482 trajectories</td><td>Task Rate Success</td><td>Text Match</td><td>https://github.com/THUDM/ VisualAgentBench/</td></tr><tr><td>SPR Benchmark 462]</td><td>Mobile, Web, Operating Systems</td><td>2024</td><td>Yes</td><td>Evaluates GUI screen readers' ability to describe both content and layout information</td><td>650 screen-shots annotated with 1,500 target points and regions</td><td>Task Success Rate, Efficiency Score</td><td>Text Match, Element Match</td><td>/</td></tr><tr><td>AgentStudio 237]</td><td>Windows, Linux, macOS</td><td>2024</td><td>Yes</td><td>Open toolkit for creating and benchmarking general-purpose virtual agents, supporting complex interactions across diverse software applications.</td><td>NA</td><td>Step Success Rate</td><td>Action Match, State Information, Image Match</td><td>https:// computer-agents. github.io/ agent-studio/</td></tr><tr><td>CRAB 461</td><td>Linux, Android</td><td>2024</td><td>Yes</td><td>Cross-environment benchmark evaluating agents across mobile and desktop devices, using a graph-based evaluation method to handle multiple correct paths and task flexibility.</td><td>120 tasks</td><td>Step Success Rate, Efficiency Score</td><td>Action Match</td><td>https: //github.com/ crab-benchmark</td></tr><tr><td>ScreenSpot 25]</td><td>iOS, Android, macOS, Windows, Web</td><td>2024</td><td>No</td><td>Vision-based GUI benchmark with pre-trained GUI grounding, assessing agents' ability to interact with GUI elements across mobile. desktop, and web platforms using only screenshots.</td><td>1,200 instructions</td><td>Step Success Rate</td><td>Action Match</td><td>https://github.com/njucckevin/ SeeClick</td></tr></table>

<!-- Media -->

## 10 APPLICATIONS OF LLM-BRAINED GUI AGENTS

As LLM-brained GUI agents continue to mature, a growing number of applications leverage this concept to create more intelligent, user-friendly, and natural language-driven interfaces. These advancements are reflected in research papers, open-source projects, and industry solutions. Typical applications encompass (i) GUI testing, which has transitioned from traditional script-based approaches to more intuitive, natural language-based interactions, and (ii) virtual assistants, which automate users' daily tasks in a more adaptive and responsive manner through natural language interfaces.

### 10.1 GUI Testing

GUI testing evaluates a software application's graphical user interface to ensure compliance with specified requirements, functionality, and user experience standards. It verifies interface elements like buttons, menus, and windows, as well as their responses to user interactions. Initially conducted manually, GUI testing evolved with the advent of automation tools such as Selenium and Appium, enabling testers to automate repetitive tasks, increase coverage, and reduce testing time [35], [543]. However, LLM-powered GUI agents have introduced a paradigm shift, allowing non-experts to test GUIs intuitively through natural language interfaces. These agents cover diverse scenarios, including general testing, input generation, and bug reproduction, without the need for traditional scripting [543].

Figure 29 and illustrates the use of an LLM-powered GUI agent to test font size adjustment on Windows OS. With only a natural language test case description, the agent autonomously performs the testing by executing UI operations, navigating through the settings menu, and leveraging its screen understanding capabilities to verify the final outcome of font size adjustment. This approach dramatically reduces the effort required for human or script-based testing. Next, we detail the GUI testing works powered by GUI agents, and first provide an overview Tables 40 41 and 42

#### 10.1.1 General Testing

Early explorations demonstrated how LLMs like GPT-3 could automate GUI testing by interpreting natural language test cases and programmatically executing them. For example, one approach integrates GUI states with GPT-3 prompts, leveraging tools like Selenium and OpenCV to reduce manual scripting and enable black-box testing [542]. Building on this, a subsequent study employed GPT-4 and Selenium WebDriver for web application testing, achieving superior branch coverage compared to traditional methods like monkey testing [463]. These advances highlight how LLMs simplify GUI testing workflows while significantly enhancing coverage and efficiency.

Further pushing boundaries, GPTDroid reframed GUI testing as an interactive Q&A task. By extracting structured semantic information from GUI pages and leveraging memory mechanisms for long-term exploration, it increased activity coverage by 32%, uncovering critical bugs with remarkable precision [125]. This approach underscores the potential of integrating conversational interfaces with memory for comprehensive app testing. For Android environments, DROIDAGENT introduced an intent-driven testing framework. It automates task generation and execution by perceiving GUI states in JSON format and using LLMs for realistic task planning. Its ability to set high-level goals and achieve superior feature coverage demonstrates how intent-based testing can transform functional verification in GUI applications [464].


<!-- Meanless: 74 -->

<!-- Media -->

JOURNAL OF LATEX CLASS FILES, DECEMBER 2024

TABLE 40: Overview of GUI-testing with LLM-powered GUI agents (Part I).

<table><tr><td>Project</td><td>Category</td><td>Platform</td><td>Model</td><td>Perception</td><td>Action</td><td>Scenario</td><td>Highlight</td><td>Link</td></tr><tr><td>Daniel and Anne 542]</td><td>General testing</td><td>General-purpose platforms</td><td>GPT-3</td><td>GUI structure and state</td><td>Standard U operations</td><td>Automates the software testing process using natural language test cases</td><td>Applies GPT-3's language understanding capabilities to GUI-based software testing. enabling natural interaction through text-based test case descriptions.</td><td>https: //github.com/ neuroevolution% 2Dai/ <br> SoftwareTestingLanguageModel:</td></tr><tr><td>Daniel and Anne 463]</td><td>General testing</td><td>Web platforms</td><td>GPT-4</td><td>HTML DOM structure</td><td>Standard I operations</td><td>Automated GUI testing to enhance branch coverage and efficiency</td><td>Performs end-to-end GUI testing using GPT-4's natural language understanding and reasoning capabilities.</td><td>https: <br> //github.com/ <br> Software TestingLLMs/ <br> WebtestingWithLLMs</td></tr><tr><td>GPTDroid [125]</td><td>General testing</td><td>Mobile Android</td><td>GPT-3.5</td><td>UI view hierarchy files</td><td>Standard UI operations and compound actions</td><td>Automates GUI testing to improve testing coverage and detect bugs efficiently</td><td>Formulates GUI testing as a Q& A task, utilizing LLM capabilities to provide human-like interaction.</td><td>https: //github.com/ franklinbill GPTDroid</td></tr><tr><td>DROID-AGENT 464]</td><td>General testing</td><td>Mobile Android</td><td>GPT-3.5, GPT-4</td><td>JSON representation of the GUI state</td><td>Standard UI operations, higher-level APIs, and custom actions</td><td>Semantic, intent-driven automation of GUI testing</td><td>Autonomously generates and executes high-level, realistic tasks for Android GUI testing based on app-specific functionalities.</td><td>https://github.com/coinse/ droidagent</td></tr><tr><td>AUITest-Agent 465]</td><td>General testing</td><td>Mobile Android</td><td>GPT-4</td><td>GUI screenshots, UI hierarchy files, CV-enhanced techniques like Vision-UI</td><td>Standard operations</td><td>Automated functional testing of GUIs</td><td>Features dynamic agent organization for step-oriented testing and a multi-source data extraction strategy for precise function verification.</td><td>https://github.com/bz-lab/ AUITestAgent</td></tr><tr><td>VisionDroid 466]</td><td>General testing</td><td>Mobile Android</td><td>GPT-4</td><td>GUI screenshots with annotated bounding boxes, View hierarchy files</td><td>Standard UI operations</td><td>Identifies n crash bugs</td><td>Integrates vision-driven prompts and GUI text alignment with vision-language models to enhance understanding of G contexts and app logic.</td><td>https: //github.com/ testtestA6/ VisionDroid</td></tr><tr><td>AXNav 467]</td><td>Accessibility testing</td><td>iOS mobile devices</td><td>GPT-4</td><td>GUI screenshots, UI element detection model, and OCR</td><td>Gestures, capturing screenshots, and highlighting potential accessibility issues</td><td>Automates accessibility testing workflows. including testing features like VoiceOver, Dynamic T Type, Bold Text, and Button Shapes</td><td>Adapts to natural language test instructions and generates annotated videos to visually and interactively review accessibility test results.</td><td>/</td></tr><tr><td>LLMigrate 473]</td><td>General testing</td><td>Mobile Android</td><td>GPT-4o</td><td>DOM and screenshots</td><td>Standard UI operations</td><td>Automates the transfer of usage-based UI tests between Android apps</td><td>Leverages multimodal LLMs to perform UI test transfers without requiring source code access</td><td>/</td></tr></table>

<!-- Media -->

ProphetAgent [482] introduces a novel approach to LLM-powered GUI testing by automatically synthesizing Android application test scripts from natural language descriptions. Departing from previous methods that directly apply LLMs to GUI screenshots or app behaviors, ProphetAgent builds a Clustered UI Transition Graph (CUTG) enriched with semantic annotations. This structured representation enables more accurate mapping between natural language test steps and GUI operations, leading to significant improvements in completion rate (78.1%) and action accuracy (83.3%). The system employs a dual-agent architecture: SemanticAgent handles semantic annotation, while GenerationAgent generates executable scripts. ProphetAgent demonstrates strong scalability and real-world applicability—reducing tester workload by over 70% at ByteDance. Its performance underscores the effectiveness of combining LLMs with explicit semantic knowledge graphs in GUI-based environments.

AUITestAgent extended the capabilities of LLM-powered GUI testing by bridging natural language-driven requirements and GUI functionality [465]. Employing multi-modal analysis and dynamic agent organization, it efficiently executes both simple and complex testing instructions. This framework highlights the value of combining multi-source data extraction with robust language models to automate functional testing in commercial apps. Incorporating vision-based methods, VisionDroid redefined GUI testing by aligning screenshots with textual contexts to detect non-crash bugs [466]. This innovation ensures application reliability by identifying logical inconsistencies and exploring app functionalities that conventional methods often overlook.

Accessibility testing has also benefited from LLM-powered agents. AXNav addresses challenges in iOS accessibility workflows, automating tests for features like VoiceOver and Dynamic Type using natural language instructions and pixel-based models. Its ability to generate annotated videos for interactive review positions AXNav as a scalable and user-friendly solution for accessibility testing [467].


<!-- Meanless: 75 -->

JOURNAL OF IATEX CLASS FILES, DECEMBER 2024

<!-- Media -->

TABLE 41: Overview of GUI-testing with LLM-powered GUI agents (Part II).

<table><tr><td>Project</td><td>Category</td><td>Platform</td><td>Model</td><td>Perception</td><td>Action</td><td>Scenario</td><td>Highlight</td><td>Link</td></tr><tr><td>Cui et al., 468]</td><td>Test input generation</td><td>Mobile Android</td><td>GPT-3.5, GPT-4</td><td>GUI structures and contextual information</td><td>Entering text inputs</td><td>Generating and validating text inputs for Android applications</td><td>Demonstrates the effectiveness of various LLMs in generating context-aware text inputs, improving UI test coverage, and identifying previously unreported bugs.</td><td>/</td></tr><tr><td>QTypist 469]</td><td>Test input generation</td><td>Mobile Android</td><td>GPT-3</td><td>UI hierarchy files</td><td>Generates semantic text inputs</td><td>Automates mobile GUI testing by generating appropriate text inputs</td><td>Formulates text input generation as a cloze-style fill-in-the-blank language task.</td><td>/</td></tr><tr><td>Crash-Translator 470]</td><td>Bug replay</td><td>Mobile Android</td><td>GPT-3</td><td>Crash-related stack trace information and GUI structure</td><td>Standard UI operations</td><td>Automates the reproduction of mobile application crashes</td><td>Leverages LLMs for iterative GUI navigation and crash reproduction from stack traces, integrating a reinforcement learning-based scoring system to optimize exploration steps.</td><td>https: //github.com/ wuchiuwong/ Crash Translator</td></tr><tr><td>AdbGPT 471]</td><td>Bug replay</td><td>Mobile Android</td><td>GPT-3.5</td><td>GUI structure and hierarchy</td><td>Standard U operations</td><td>Automates bug reproduction extracting S2R (Steps to Reproduce) entities</td><td>Combines prompt engineering with few-shot learning and chain-of-thought reasoning to leverage LLMs for GUI-based tasks.</td><td>https: //github.com/ sidongfeng/ AdbGPT</td></tr><tr><td>MagicWand 472]</td><td>Verrification</td><td>Mobile Android</td><td>GPT-4V</td><td>UI screen-shots and hierarchical UI control tree</td><td>Standard U operations</td><td>Automates the verification of "How-to" instructions from a search engine</td><td>Features a three-stage process: extracting instructions, executing them in a simulated environment, and reranking search results based on execution outcomes.</td><td>/</td></tr><tr><td>UXAgent 474</td><td>Usability testing for web design</td><td>Web</td><td>Self-designed</td><td>Simplified HTML representations</td><td>Standard L operations</td><td>Automated usability testing of web applications</td><td>ables LLM-powered automated usability testing by simulating thousands of user interactions. collecting both qualitative and quantitative data, and providing researchers with early feedback before real-user studies.</td><td>https: <br> //uxagent. <br> hailab.io</td></tr><tr><td>Guardian 475</td><td>GUI Testing</td><td>Mobile Android</td><td>GPT-3.5</td><td>GUI structure, Properties</td><td>Standard L operations</td><td>Autonomously explores mobile applications. interacting with the UI to validate functionalities.</td><td>Improves LLM-driven UI testing by offloading planning tasks to an external runtime system.</td><td>/</td></tr><tr><td>Test-Agent 476</td><td>GUI Testing</td><td>Android, iOS, Harmony OS</td><td>Not Mentioned</td><td>GUI screenshots, UI structure information</td><td>Standard 【 operations</td><td>Cross-platform mobile testing</td><td>Eliminates the need for pre-written test scripts by leveraging LLMs and multimodal perception to generate and execute test cases automatically.</td><td>/</td></tr><tr><td>VLM-Fuzz 477]</td><td>GUI Testing</td><td>Android (Mobile)</td><td>GPT-4o</td><td>GUI screenshots and structure information</td><td>Standard U operations. system-level actions</td><td>Automated Android testing detection of crashes and bugs</td><td>Integrates vision-language reasoning with heuristic-based depth-first search (DFS) to systematically explore complex Android UIs, achieving significantly higher code coverage</td><td>/</td></tr></table>

<!-- Media -->

#### 10.1.2 Text Input generation

In the realm of text input generation, Cui et al., demonstrated how GPT-3.5 and GPT-4 could enhance Android app testing by generating context-aware text inputs for UI fields [468]. By systematically evaluating these inputs across multiple apps, they revealed the potential of LLMs in improving test coverage and detecting unique bugs with minimal manual intervention. Similarly, QTypist formulated text input generation as a fill-in-the-blank task, leveraging LLMs to improve activity and page coverage by up to 52% [469].

#### 10.1.3 Bug Replay

For bug reproduction, CrashTranslator automated the reproduction of crashes from stack traces by integrating reinforcement learning with LLMs. Its iterative navigation and crash prediction steps significantly reduced debugging time and outperformed state-of-the-art methods [470]. Meanwhile, AdbGPT demonstrated how few-shot learning and chain-of-thought reasoning could transform textual bug reports into actionable GUI operations. By dynamically inferring GUI actions, AdbGPT provided an efficient and lightweight solution for bug replay [471].

BugCraft 478 leverages LLM-powered GUI agents to automate bug reproduction in games, specifically targeting the open-ended and complex environment of Minecraft. It employs GPT-40 as the inference engine, integrating textual bug reports, visual GUI understanding through OmniParser 184 , and external knowledge from the Minecraft Wiki to generate and execute structured reproduction steps. Actions are carried out via a custom Macro API, enabling robust interaction with both the game's GUI and environment. BugCraft's ability to translate unstructured bug descriptions into executable in-game behaviors highlights the strong potential of vision-enhanced LLM agents for advancing software testing and debugging.


<!-- Meanless: 76 -->

<!-- Media -->

JOURNAL OF LATEX CLASS FILES, DECEMBER 2024

TABLE 42: Overview of GUI-testing with LLM-powered GUI agents (Part III).

<table><tr><td>Project</td><td>Category</td><td>Platform</td><td>Model</td><td>Perception</td><td>Action</td><td>Scenario</td><td>Highlight</td><td>Link</td></tr><tr><td>BugCraft 478]</td><td>Bug Reproduction</td><td>Windows Computer</td><td>BugCraft based on GPT-40</td><td>GUI screenshots</td><td>Standard UI operations</td><td>Automatically reproduces crash bugs in Minecraft by reading user-submitted bug reports, generating structured steps, and executing them to cause a crash</td><td>First end-to-end framework that automates crash bug reproduction in a complex open-world game (Minecraft) using LLM-driven agents, vision-based UI parsing, and structured action execution</td><td>https:// bugcraft2025. github.io/</td></tr><tr><td>ReuseDroic 479]</td><td>GUI Testing</td><td>Mobile Android</td><td>ReuseDroid based on GPT-40</td><td>GUI screenshots and widget properties</td><td>Standard UI operations</td><td>Migrates GUI test cases between Android a that share similar functionality but differ in operational logic</td><td>Leverages visual contexts and dynamic feedback mechanisms to significantly boost migration success rates compared to prior mapping- and LLM-based methods</td><td>/</td></tr><tr><td>SeeAct-ATA and PinATA 480]</td><td>GUI Testing</td><td>Web</td><td>SeeAct [17]</td><td>GUI structure and DOM</td><td>Standard U operations</td><td>Automates manual to-end (E2E) web application testing</td><td>First open-source attempt to adapt LLM-powered Autonomous Web Agents into Autonomous Test Agents (ATA) for web testing</td><td>/</td></tr><tr><td>GERALLT 481]</td><td>GUI Testing</td><td>Desktop (Windows/Linux</td><td>GPT-40</td><td>GUI screenshots and structure information</td><td>Standard L operations</td><td>Finds unintuitive behavior, inconsistencies, and functional errors in GUIs without predefined test scripts</td><td>Pioneers LLM-driven testing on real-world desktop GUI applications (not web or mobile), combining structured GUI parsing with LLM-based control and evaluation</td><td>https://github.com/DLR-SC/ GERALLT</td></tr><tr><td>ProphetAge <br> 482]</td><td>gentGUI Testing</td><td>Android Mobile</td><td>and Gener-ationAgent using foundation models (GPT-40)</td><td>SemanticAgent XML UI trees</td><td>Executable UI test scripts</td><td>Automates GUI test case generation from natural language for regression and compatibility testing in mobile apps</td><td>Innovatively combines LLM reasoning with a semantically enriched GUI graph (CUTG), significantly improving GUI test synthesis performance and efficiency over state-of-the-art tools</td><td>https: //github.com/ prophetagent/ Home</td></tr><tr><td>Agent for User 483]</td><td>GUI Testing</td><td>Android Mobile</td><td>GPT-4</td><td>XML view hierarchy</td><td>Standard operations</td><td>Automated testing of multiuser interactive features</td><td>Introduces a multi-agent LLM framework where each agent simulates a user on a virtual device</td><td>/</td></tr></table>

<!-- Media -->

#### 10.1.4 Verification

Finally, as a novel application in testing, MagicWand showcased the potential of LLMs in automating "How-to" verifications. By extracting, executing, and refining instructions from search engines, it addressed critical challenges in user-centric task automation, improving the reliability of GUI-driven workflows 472

In summary, LLM-powered GUI agents have revolutionized GUI testing by introducing natural language-driven methods, vision-based alignment, and automated crash reproduction. These innovations have enhanced test coverage, efficiency, and accessibility, setting new benchmarks for intelligent GUI testing frameworks.

### 10.2 Virtual Assistants

Virtual assistants, such as Sir 32 are AI-driven applications that help users by performing tasks, answering questions, and executing commands across various platforms, including web browsers, mobile phones, and computers. Initially, these assistants were limited to handling simple commands via voice or text input, delivering rule-based responses or running fixed workflows similar to RPA. They focused on basic tasks, such as setting alarms or checking the weather.

With advancements in LLMs and agents, virtual assistants have evolved significantly. They now support more complex, context-aware interactions on device GUIs through textual or voice commands and provide personalized responses, catering to diverse applications and user needs on various platforms. This progression has transformed virtual assistants from basic utilities into intelligent, adaptive tools capable of managing intricate workflows and enhancing user productivity across platforms. Figure 30 presents a conceptual example of a GUI agent-powered virtual assistant on a smartphone ${}^{33}$ . In this scenario, the agent enables users to interact through chat, handling tasks such as setting up a screenshot shortcut on their behalf. This feature is particularly beneficial for users unfamiliar with the phone's functionalities, simplifying complex tasks into conversational commands.

---

<!-- Footnote -->

32. https://www.apple.com/siri/

33. The application and scenario depicted in the figure are conceptual and fabricated. They do not reflect the actual functionality of any specific smartphone. Readers should consult the phone manual or official guidance for accurate information on AI assistant capabilities.

<!-- Footnote -->

---


<!-- Meanless: JOURNAL OF LATEX CLASS FILES, DECEMBER 2024 -->

<!-- Media -->

<!-- figureText: Step 3: Move the "Text size" slider to adjust the font size to 175. Action: SetSlider(Slider("Text size"), 175)<br>Step 1: Click on the "Accessibility" section in the left-hand navigation menu. Action: Click(MenuItem("Accessibility"))<br>Step 2: Click on "Text size" from the list of options.<br>Action: Click(MenuItem("Text size"))<br>合<br>Test Case: Adjust the font size to 175% in the Windows Settings and verify that all displayed fonts across the interface scale up appropriately.<br>1 [umbers<br>0.5<br>Venel-statement<br><icon/><br>Step 4: Click the "Apply" button to Action: Click(Button("Apply"))<br>Step 5: Wait for the screen to complete and changes to apply. Action: Wait(15)<br>Step 6: Verify that the font size has been successfully adjusted and applied Action: VerifySetting("Text size", 175)<br>GUI Agent Tester<br>Accessibility $\rightarrow$ Text size<br>Accessibility - Text size<br>-<br>as System<br>If we<br><icon/><br>Please wait<br>#Time & languag<br>PARTING<br><icon/><br><icon/><br><icon/> -->

<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2025/11/2025_11_04__12_39_04_355eba.jpg"/>

<!-- Media -->

Fig. 29: An example of testing font size adjustment using an LLM-powered GUI agent.

To explore more real-world applications of virtual assistants powered by GUI agents, we provide an overview of advancements across research, open-source initiatives, and production-level applications, as summarized in Table 43 and 44

#### 10.2.1 Research

Recent research efforts have significantly advanced the capabilities of virtual assistants by integrating LLM-powered GUI agents, enabling more intelligent and adaptable interactions within various applications.

Firstly, the integration of LLMs into GUI-based automation has been explored to enhance business process automation. For instance, [485] introduces Agentic Process Automation through the development of ProAgent, which automates both the creation and execution of workflows in GUI environments. By utilizing agents like ControlAgent and DataAgent, it supports complex actions such as dynamic branching and report generation in applications like Slack and Google Sheets. This approach transcends traditional RPA by enabling flexible, intelligent workflows, significantly reducing the need for manual intervention and highlighting the transformative potential of LLM-powered agents in virtual assistants.

Building upon the idea of integrating LLMs with GUI environments, researchers have focused on mobile platforms to automate complex tasks. LLMPA [486] is a pioneering framework that leverages LLMs to automate multi-step tasks within mobile applications like Alipay. It interacts directly with app GUIs, mimicking human actions such as clicks and typing, and employs UI tree parsing and object detection for precise environment understanding. A unique controllable calibration module ensures logical action execution, demonstrating the potential of LLM-powered virtual assistants to handle intricate workflows and real-world impact in assisting users with diverse tasks.

Similarly, the automation of smartphone tasks through natural language prompts has been addressed by PromptRPA [490]. Utilizing a multi-agent framework, it automates tasks within smartphone GUI environments, tackling challenges like interface updates and user input variability. Advanced perception methods, including OCR and hierarchical GUI analysis, are employed to understand and interact with mobile interfaces. By supporting real-time feedback and iterative improvements, PromptRPA underscores the importance of user-centered design in LLM-driven virtual assistants.

In the realm of accessibility, LLM-powered GUI agents have been instrumental in enhancing user experience for individuals with disabilities. For example, VizAbility [484] enhances the accessibility of data visualizations for blind and low-vision users. By combining structured chart navigation with LLM-based conversational interactions, users can ask natural language queries and receive insights on chart content and trends. Leveraging frameworks like OII ${}^{34}$ and chart specifications such as Vega-Lite 35 VizAbility allows exploration of visual data without direct visual perception, addressing real-world accessibility challenges in GUIs.

Furthermore, addressing the needs of older adults, EasyAsk [491] serves as a context-aware in-app assistant that enhances usability for non-technical users. By integrating multi-modal inputs, combining natural voice queries and touch interactions with GUI elements, it generates accurate and contextual tutorial searches. EasyAsk demonstrates how GUI agents can enhance accessibility by integrating contextual information and interactive tutorials, empowering users to navigate smartphone functions effectively.

---

<!-- Footnote -->

34. https://mitvis.github.io/olli/

35. https://vega.github.io/

<!-- Footnote -->

---


<!-- Meanless: 78 -->

<!-- Media -->

JOURNAL OF LATEX CLASS FILES, DECEMBER 2024

TABLE 43: Overview of virtual assistants with LLM-powered GUI agents (Part I).

<table><tr><td>Project</td><td>Type</td><td>Platform</td><td>Model</td><td>Perception</td><td>Action</td><td>Scenario</td><td>Highlight</td><td>Link</td></tr><tr><td>ProAgent 485]</td><td>Research</td><td>Web and Desktop</td><td>GPT-4</td><td>Task descriptions and structured application data</td><td>Standard UI operations and dynamic branching</td><td>Automates business processes such as data analysis, report generation, and notifications via GUI-based tools</td><td>Introduces dynamic work-flows where agents interpret and execute tasks flexibly, surpassing traditional RPA systems</td><td>https: //github.com/ OpenBMB/ ProAgent</td></tr><tr><td>LLMPA 486]</td><td>Research</td><td>Mobile (Android)</td><td>AntLLM- 10b</td><td>UI tree structures, visual modeling, and text extraction modules</td><td>Standard UI operations</td><td>Automates user interactions within mobile apps, such as ticket booking</td><td>Integrates LLM reasoning capabilities with a modular design that supports task decomposition, object detection, and robust action prediction in GUI environments</td><td>/</td></tr><tr><td>VizAbility 484]</td><td>Research</td><td>Desktop</td><td>GPT-4V</td><td>Keyboard-navigable tree views</td><td>Navigates chart structures and generates answers</td><td>Assists blind and low-vision users in exploring and understanding data visualizations</td><td>Integrates structured chart navigation with LLM-powered conversational capabilities, enabling visually impaired users to query in natural language</td><td>https://dwr.bc.edu/vizability/</td></tr><tr><td>GPTVoice-Tasker 487]</td><td>Research</td><td>Mobile (Android)</td><td>GPT-4</td><td>Android Accessibility Tree</td><td>Standard UI operations</td><td>Automates user interactions on mobile devices through voice commands</td><td>Integrates LLMs for natural command interpretation and real-time GUI interactions, using a graph-based local database to record and replicate interactions</td><td>https: //github.com/ vuminhduc796 GPTVoiceTasker</td></tr><tr><td>AutoTask 488]</td><td>Research</td><td>Mobile (Android)</td><td>GPT-4</td><td>Android Accessibility Tree</td><td>Standard UI operations</td><td>Automates multistep tasks on mobile devices</td><td>Operates without predefined scripts or configurations, autonomously exploring GUI environments</td><td>https: //github.com/ BowenBryanWan AutoTask</td></tr><tr><td>AssistEditor 489]</td><td>Research</td><td>Windows</td><td>UniVTG 544]</td><td>GUI elements, user requirements, and video data</td><td>Standard UI operations</td><td>Automates video editing workflows</td><td>Employs a multi-agent collaboration framework where agents specialize in roles to integrate user requirements into video editing workflows</td><td>7</td></tr><tr><td>PromptRPA 490]</td><td>Research</td><td>Mobile (Android)</td><td>GPT- 4 and GPT-3.5 Turbo</td><td>Layout hierarchy and screenshots with OCR</td><td>Standard U operations and application-level functionalities</td><td>Automates smartphone tasks and creates interactive tutorials</td><td>Integrates user feedback loops for continuous improvement, addressing interface evolution and task variability</td><td>/</td></tr><tr><td>EasyAsk 491]</td><td>Research</td><td>Mobile (Android)</td><td>GPT-4</td><td>Android Accessibility Tree</td><td>Highlights specific UI elements for user interaction</td><td>Assists older adults in learning and navigating smartphone functions through in-app interactive tutorials</td><td>Combines voice and touch inputs, supplementing incomplete or ambiguous queries with in-app contextual information</td><td>/</td></tr><tr><td>WebNav 497]</td><td>Research</td><td>Web</td><td>Gemini 2.0 Flash Thinking</td><td>Standard UI operations</td><td>GUI screenshots and DOM</td><td>Assistive technology fo visually impaired users, enabling voice-based navigation complex websites</td><td>Combines a ReAct-style reasoning loop, real-time DOM labeling, and voice-driven interaction to support intelligent web navigation for visually impaired users</td><td>/</td></tr></table>

<!-- Media -->

Voice interaction has also been a focus area, with tools like GPTVoiceTasker [487] facilitating hands-free interaction with Android GUIs through natural language commands. It bridges the gap between voice commands and GUI-based actions using real-time semantic extraction and a hierarchical representation of UI elements. By automating multi-step tasks and learning from user behavior, it enhances task efficiency and reduces cognitive load, highlighting the transformative potential of LLMs in improving accessibility and user experience in mobile environments.

Expanding on voice-powered interactions, AutoTask 488 enables virtual assistants to execute multi-step tasks in GUI environments without predefined scripts. It autonomously explores and learns from mobile GUIs, effectively combining voice command interfaces with dynamic action engines to interact with GUI elements. Utilizing trial-and-error and experience-driven learning, AutoTask adapts to unknown tasks and environments, showcasing its potential in enhancing voice-driven virtual assistants for hands-free interactions.

Finally, in the domain of creative workflows, AssistEdi-tor 489 exemplifies a multi-agent framework for automating video editing tasks. By interacting with GUI environments, it autonomously performs complex workflows using dialogue systems and video understanding models to bridge user intent with professional editing tasks. The innovative use of specialized agents ensures efficient task distribution and execution, demonstrating the practical application of LLM-powered GUI agents in real-world scenarios and expanding automation into creative domains.

These research endeavors collectively showcase significant advancements in LLM-powered GUI agents, highlighting their potential to transform virtual assistants into intelligent, adaptable tools capable of handling complex tasks across various platforms and user needs.


<!-- Meanless: 79 -->

#### 10.2.2 Open-Source Projects

<!-- Media -->

JOURNAL OF LATEX CLASS FILES, DECEMBER 2024

TABLE 44: Overview of virtual assistants with LLM-powered GUI agents (Part II).

<table><tr><td>Project</td><td>Type</td><td>Platform</td><td>Model</td><td>Perception</td><td>Action</td><td>Scenario</td><td>Highlight</td><td>Link</td></tr><tr><td>OpenAdapt 492]</td><td>Open-source</td><td>Desktop</td><td>LLM, VLM (e.g., GPT-4, ACT-1)</td><td>Screenshots with CV tools for GUI parsing</td><td>Standard UI operations</td><td>Automates repetitive tasks across industries</td><td>Learns task automation by observing user interactions, eliminating manual scripting</td><td>https: //github.com/ OpenAdaptAI/ OpenAdapt</td></tr><tr><td>AgentSea 493]</td><td>Open-source</td><td>Desktop and Web</td><td>LLM, VLM</td><td>Screenshots with CV tools for GUI parsing</td><td>Standard UI operations</td><td>Automates tasks within GUI environments</td><td>Offers a modular toolkit adhering to the UNIX philosophy, allowing developers to create custom AI agents for diverse GUI environments</td><td>https://www.<br> agentsea.ai/</td></tr><tr><td>Open Interpreter 494]</td><td>Open-source</td><td>Desktop, Web, Mobile (Android)</td><td>LLM</td><td>System perception via command-line</td><td>Shell commands, code, and native APIs</td><td>Automates tasks, conducts data analysis, manages files, and controls web browsers for research</td><td>Executes code locally, providing full access to system resources and libraries, overcoming limitations of cloud-based services</td><td>https: //github.com/ OpenInterpreter/ open-interpreter</td></tr><tr><td>MultiOn 495]</td><td>Production</td><td>Web</td><td>LLM</td><td>│</td><td>Standard UI operations</td><td>Automates web-based tasks</td><td>Performs autonomous web actions via natural language commands</td><td>https://www.multion.ai/</td></tr><tr><td>YOYO Agent in MagicOS 496]</td><td>Production</td><td>Mobile (Magi-cOS 9.0)</td><td>MagicLM</td><td>GUI context</td><td>Executes in-app and cross-app operations</td><td>Automates daily tasks, enhancing productivity</td><td>Leverages MagicLM to understand and execute complex tasks across applications, learning user habits to provide personalized assistance</td><td>/</td></tr><tr><td>Power Automate 132]</td><td>Production</td><td>Windows</td><td>LLM, VLM</td><td>Records user interactions with the GUI</td><td>Standard UI operations</td><td>Automates repetitive tasks and streamlines work-flows</td><td>Translates natural language descriptions of desired automations into executable workflows</td><td>https://learn.microsoft.com/en-us/ power-automate/ desktop-flows/ create% 2Dflow-using% 2Dai-recorder</td></tr><tr><td>Eko 545]</td><td>Production</td><td>Web browsers and computer environments</td><td>ChatGPT and Claude 3.5</td><td>Visual-Interactive Element Perception (VIEP) technology for interacting with GUI elements.</td><td>Standard UI operations.</td><td>Automates tasks by handling diverse workflows.</td><td>Decomposes natural language task descriptions into executable workflows, enabling seamless integration of natural language and programming logic in agent design.</td><td>https: //eko.fellou.ai/</td></tr></table>

<!-- Media -->

In addition to research prototypes, open-source projects have contributed substantially to the development and accessibility of LLM-brained GUI agents, enabling wider adoption and customization.

One such project is OpenAdapt [492], an open-source framework that utilizes large multimodal models to automate tasks by observing and replicating user interactions within GUI environments. It captures screenshots and records user inputs, employing computer vision techniques to understand and execute standard UI operations. Designed to streamline workflows across various industries, OpenAdapt learns from user demonstrations, thereby reducing the need for manual scripting and showcasing adaptability in GUI-based task automation.

Similarly, AgentSea [493] offers a comprehensive and modular toolkit for creating intelligent agents that can navigate and interact with various GUI environments across multiple platforms. Its flexibility is particularly beneficial for developing virtual assistants capable of automating complex tasks within applications, enhancing user productivity. By adhering to the UNIX philosophy, AgentSea ensures that each tool is specialized, promoting ease of use and extensibility. Its open-source nature fosters community collaboration and innovation in AI-driven GUI automation.

Open Interpreter [494] further exemplifies the potential of open-source contributions by leveraging large language models to execute code locally. Users can interact with their computer's GUI through natural language commands, supporting multiple programming languages and operating across various platforms. By facilitating tasks such as data analysis, web automation, and system management, Open Interpreter provides unrestricted access to system resources and libraries, enhancing flexibility and control. Its customization capabilities make it a valuable asset for users aiming to streamline operations through AI-powered virtual assistance.

These open-source projects not only advance the state of LLM-powered GUI agents but also democratize access to intelligent virtual assistants, enabling developers and users to tailor solutions to specific needs and applications.

#### 10.2.3 Production

The integration of LLM-brained GUI agents into production environments demonstrates their practical viability and impact on enhancing user experiences in commercial applications.

Power Automate 132 exemplifies an AI-powered GUI agent that enhances user interaction with desktop applications. By allowing users to describe tasks in natural language while recording actions, it translates these descriptions into automated workflows, effectively bridging the gap between user intent and execution. Its ability to record and replicate user actions within the GUI streamlines the automation of repetitive tasks, making it a valuable tool for increasing efficiency and highlighting advancements in user-friendly automation solutions.


<!-- Meanless: JOURNAL OF LATEX CLASS FILES, DECEMBER 2024<br>80 -->

<!-- Media -->

<!-- figureText: Use Back Tap on your iPhone<br>User<br>Can you set a shortcut for taking screenshots on my phone?<br>Sure! You can set a shortcut to take screenshots by double-tapping or triple-tapping the back of your phone. Would you like to set it to double-tap?<br>Turn on Back Tap<br>Can it be set to double-tap?<br>Absolutely! Let me take care of that for you.<br>...After setting up the shortcut...<br>The shortcut has been successfully set! You can now take a screenshot by double-tapping the back of your phone.<br>GUI Agent Virtual Assistant<br>完成iPhone设置<br>Camera<br>Display & Test Size<br>Shake to Undo<br>OBD -->

<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2025/11/2025_11_04__12_39_04_948bfa.jpg"/>

Fig. 30: A conceptual example of a GUI agent-powered virtual assistant on a smartphone.

<!-- Media -->

In the realm of web interactions, MultiOn [495] serves as a personal AI agent that autonomously interacts with web-based GUIs to execute user-defined tasks. Leveraging large language models, it interprets natural language commands and translates them into precise web actions, effectively automating complex or repetitive tasks. MultiOn's approach to perceiving and manipulating web elements enables seamless functioning across various web platforms, enhancing user productivity and streamlining web interactions.

On mobile platforms, the YOYO Agent in MagicOS [496] exemplifies an LLM-powered GUI agent operating within the MagicOS 9.0 interface. Utilizing Honor's MagicLM, it comprehends and executes user commands across various applications, learning from user behavior to offer personalized assistance. This integration demonstrates how large language models can enhance virtual assistants, enabling them to perform complex tasks within GUI environments and improving user experience and productivity on mobile devices.

Eko 545] serves as a prime example of a versatile and efficient tool for developing intelligent agents capable of interacting with GUIs across various platforms. Its integration with multiple LLMs and the innovative Visual-Interactive Element Perception (VIEP) technology highlight its capability to perform complex tasks through natural language instructions. Eko's comprehensive tool support make it a valuable resource for developers aiming to create customizable and production-ready agent-based workflows. By facilitating seamless interaction within GUI environments, Eko exemplifies the advancements in virtual assistants powered by LLMs.

These production-level implementations highlight the practical applications and benefits of LLM-brained GUI agents in enhancing automation, productivity, and user engagement across different platforms and industries.

### 10.3 Takeaways

The application of LLM-brained GUI agents has ushered in new capabilities and interfaces for tasks such as GUI testing and virtual assistance, introducing natural language interactions, enhanced automation, and improved accessibility across platforms. These agents are transforming the way users interact with software applications by simplifying complex tasks and making technology more accessible. However, despite these advancements, LLM-brained GUI agents are still in their infancy, and several challenges need to be addressed for them to reach maturity. Key insights from recent developments include:

1) Natural Language-Driven Interactions: LLM-powered GUI agents have enabled users to interact with applications using natural language, significantly lowering the barrier to entry for non-expert users. In GUI testing, tools like GPTDroid [125] and AUITestAgent [465] allow testers to specify test cases and requirements in plain language, automating the execution and verification processes. Similarly, virtual assistants like LLMPA [486] and ProAgent [485] interpret user commands to perform complex tasks, showcasing the potential of natural language interfaces in simplifying user interactions across platforms.

2) Enhanced Automation of Complex Tasks: These agents have demonstrated the ability to automate multistep and intricate workflows without the need for manual scripting. Projects like AutoTask [488] and GPTVoic-eTasker [487] autonomously explore and interact with GUI environments, executing tasks based on high-level goals or voice commands. In GUI testing, agents have improved coverage and efficiency by automating the generation of test inputs and reproducing bugs from textual descriptions, as seen in CrashTranslator [470] and AdbGPT [471].


<!-- Meanless: JOURNAL OF LATEX CLASS FILES, DECEMBER 2024<br>81 -->

3) Multimodal Perception and Interaction: Integrating visual and textual inputs has enhanced the agents' understanding of GUI contexts, leading to better decision-making and interaction accuracy. Agents like VizAbil-ity [484] and OpenAdapt [492] utilize screenshots, UI trees, and OCR to perceive the environment more comprehensively. This multimodal approach is crucial for applications that require precise identification and manipulation of GUI elements, especially in dynamic or visually complex interfaces.

4) Improved Accessibility and User Experience: LLM-brained GUI agents have contributed to making technology more accessible to users with disabilities or limited technical proficiency. Tools like VizAbility [484] aid blind and low-vision users in understanding data visualizations, while EasyAsk [491] assists older adults in navigating smartphone functions. By tailoring interactions to the needs of diverse user groups, these agents enhance inclusivity and user experience.

LLM-brained GUI agents are transforming the landscape of GUI interaction and automation by introducing natural language understanding, enhanced automation capabilities, and improved accessibility. While they are still in the early stages of development, the ongoing advancements and emerging applications hold great promise for the future. Continued research and innovation are essential to overcome current challenges and fully realize the potential of these intelligent agents across diverse domains and platforms.

## 11 Limitations, Challenges and Future ROADMAP

Despite significant advancements in the development of LLM-brained GUI agents, it is important to acknowledge that this field is still in its infancy. Several technical challenges and limitations hinder their widespread adoption in real-world applications. Addressing these issues is crucial to enhance the agents' effectiveness, safety, and user acceptance. In this section, we outline key limitations and propose future research directions to overcome these challenges, providing concrete examples to illustrate each point.

### 11.1 Privacy Concerns

Privacy is a critical concern uniquely intensified in the context of LLM-powered GUI agents. These agents often require access to sensitive user data—such as screenshots, interaction histories, personal credentials, and confidential documents—to effectively perceive and interact with the GUI environment. In many cases, this data must be transmitted to remote servers for model inference, especially when relying on cloud-based LLMs [546]-[548]. Such deployments raise significant privacy risks, including data breaches, unauthorized access, and misuse of personal information. These concerns are further amplified when sensitive inputs are routed through third-party APIs or processed off-device, creating compliance and security vulnerabilities that can deter real-world adoption.

For instance, a GUI agent tasked with managing a user's email inbox may need to read, classify, and respond to messages containing highly personal or confidential content. Offloading this processing to the cloud introduces risks of exposure, prompting hesitation among users and organizations due to potential privacy violations [432], [549], [550]. Compared to traditional LLM applications, GUI agents operate at a finer granularity of user activity and often require broader system access, making privacy-preserving deployment strategies a critical and domain-specific challenge.

Potential Solutions: To mitigate privacy concerns, future research should focus on enabling on-device inference, where the language model operates directly on the user's device without uploading personal data [551], [552]. Achieving this requires advancements in model compression techniques 553, on-device optimization [554], and efficient inference algorithms 555 to accommodate the computational limitations of user devices. In addition, frameworks must incorporate data redaction, secure communication channels, and explicit scoping of data usage within the agent's context. Furthermore, integration with system-level privacy controls and user consent mechanisms (e.g., runtime permission dialogs or sandboxed execution) is essential for deployment in regulated domains.

From the technical perspective, implementing privacy-preserving techniques like federated learning [556], differential privacy [557], and homomorphic encryption [558] can enhance data security while allowing the model to learn from user data. Furthermore, developers of GUI agents should collaborate with privacy policymakers to ensure that user data and privacy are appropriately protected [559]. They should make the data handling processes transparent to users, clearly informing them about what data are being transmitted and how they are used, and obtain explicit user consent 560

### 11.2 Latency, Performance, and Resource Constraints

One challenge that is particularly salient for GUI agents-distinct from general LLM applications is the issue of latency in interactive, multi-step execution environments. Since GUI agents rely on large language models to plan and issue actions, their computational demands can lead to high latency and slow response times, which directly impact user experience [561]. This is especially critical in time-sensitive or interactive scenarios, where delays in action execution can cause user frustration or even trigger unintended system behavior. Unlike single-shot LLM tasks, GUI agents typically operate over extended sequences of steps, making latency cumulative and more disruptive over time. The problem is further amplified in on-device deployments, where computational resources are limited. For example, running an LLM-powered agent within a mobile app may result in sluggish performance or rapid battery depletion, significantly undermining usability on resource-constrained platforms 562-564]. These concerns are uniquely pronounced in GUI agents due to their need for real-time perception, decision-making, and UI control in dynamic environments [563].

Potential Solutions: Future work should aim to reduce inference latency by optimizing model architectures for speed and efficiency [565]. Techniques such as model distillation can create smaller, faster models without substantially compromising performance [566]. Leveraging hardware accelerators like GPUs, TPUs, or specialized AI chips, and exploring parallel processing methods can enhance computational efficiency [567]. Implementing incremental inference and caching mechanisms may also improve responsiveness by reusing computations where applicable [568]. Additionally, research into model optimization and compression techniques, such as pruning [569] and quantization [553] can produce lightweight models suitable for deployment on resource-constrained devices. Exploring edge computing [552] and distributed inference [570] can help distribute the computational load effectively.


<!-- Meanless: JOURNAL OF LATEX CLASS FILES, DECEMBER 2024<br>82 -->

Moreover, GUI agents should collaborate with application developers to encourage them to expose high-level native APIs for different functionalities [198], [199], which combine several UI operations into single API calls. By integrating these APIs into the GUI agent, tasks can be completed with fewer steps, making the process much faster and reducing cumulative latency.

### 11.3 Safety and Reliability

The real-world actuation capabilities of GUI agents introduce unique and significant safety and reliability risks beyond those faced by general-purpose LLMs. Because GUI agents can directly manipulate user interfaces-clicking buttons, deleting files, submitting forms, or initiating system-level operations-errors in action generation can have irreversible consequences [548], [571]. These may include data corruption, accidental message dispatches, application crashes, or unauthorized access to sensitive system components [572], [573]. Such risks are compounded by the inherent uncertainty and non-determinism in LLM outputs: agents may hallucinate actions, misinterpret UI contexts, or behave inconsistently across sessions [574-578]. For example, an agent automating financial transactions could mistakenly execute the wrong transfer, leading to material losses. Furthermore, GUI agents expose a broader attack surface than traditional LLM applications—they are susceptible to black-box adversarial attacks that could manipulate their inputs or exploit their decision policies [579].

Unlike passive language models, GUI agents operate within dynamic software ecosystems where incorrect actions can propagate across applications or escalate into systemwide disruptions. Integration challenges also arise, including compatibility with evolving UI frameworks, user permission boundaries, and software-specific safety constraints, and malicious attacks [580], [581]. These concerns, coupled with the lack of interpretability and formal guarantees, contribute to skepticism and reluctance from users and developers alike. Addressing safety and reliability in GUI agents thus requires not only robust model behavior but also runtime safeguards 582], rollback mechanisms, and interface-aware verification techniques tailored specifically to this interaction paradigm.

Potential Solutions: Ensuring safety and reliability necessitates robust error detection and handling mechanisms [583]. Future research should focus on integrating validation steps that verify the correctness of inferred actions before execution [584]. Developing formal verification methods [585], implementing exception handling routines [586], and establishing rollback procedures [587] are essential for preventing and mitigating the impact of errors. Additionally, incorporating permission management [588-591] to limit the agent's access rights can prevent unauthorized or harmful operations.

Furthermore, creating standardized interaction protocols can facilitate smoother and safer integration with various applications and systems [592]. Ensuring that agents comply with security best practices, such as secure authentication and authorization protocols [593], is essential.

### 11.4 Human-Agent Interaction

Human-agent interaction introduces distinct challenges in the context of GUI agents, where the agent and user operate within the same dynamic interface. Any user intervention-such as moving the mouse, altering window states, or modifying inputs—can inadvertently interfere with the agent's ongoing execution, potentially causing conflicts, unintended actions, or breakdowns in task flow [594], [595]. Designing robust collaboration protocols that govern when the agent should yield control, pause execution, or defer to the user is a non-trivial problem specific to GUI-based automation.

Further complicating this interaction is the ambiguity of user instructions. Natural language commands may be vague, under-specified, or context-dependent, leading to misinterpretations or incomplete task plans. GUI agents may also encounter runtime uncertainties—such as unexpected pop-ups, missing inputs, or conflicting UI states—that require them to seek user clarification or feedback [19], [596]. Determining when and how an agent should request user input-whether for disambiguation, permission, or verification-is critical for ensuring both reliability and user trust [67], [597], [598].

This challenge is exemplified in the fabricated scenario shown in Figure 31 where a GUI agent is instructed to send an email to "Tom." The agent must first prompt the user to log in securely, protecting credentials by avoiding automated input. It then encounters ambiguity when multiple contacts named "Tom" are found, and resolves it by prompting the user to select the intended recipient. Finally, before dispatching the email, the agent requests explicit confirmation, recognizing that email-sending is a non-reversible action with privacy implications [19]. Although the task appears simple, it reflects the complexity of real-world human-GUI agent collaboration, involving privacy preservation, ambiguity resolution, and intentionality confirmation [599]. These are not generic LLM issues, but domain-specific challenges rooted in shared interaction with software interfaces—underscoring the need for new design paradigms around shared control, interruption handling, and proactive clarification in GUI agent systems.

Potential Solutions: Emphasizing user-centered design 600 principles can address user needs and concerns, providing options for customization and control over the agent's behavior [596]. Equipping agents with the ability to engage in clarification dialogues when user instructions are unclear can enhance task accuracy [601]. Natural language understanding components can detect ambiguity and prompt users for additional information. For instance, the agent could ask, "There are two contacts named John. Do you mean John Smith or John Doe?" Incorporating human-in-the-loop systems allows for human intervention during task execution, enabling users to guide or correct the agent's decisions when necessary [602]. Developing adaptive interaction models that facilitate seamless collaboration between humans and agents is essential. Additionally, providing transparency and explainability in the agent's reasoning processes can build user trust and improve cooperation [67, 603], 604].


<!-- Meanless: JOURNAL OF LATEX CLASS FILES, DECEMBER 2024<br>83 -->

<!-- Media -->

<!-- figureText: (a) Ask for manual invervention<br>(b) Handle ambiguity<br>(c) Request for confirmation<br>Task: Draft an email to congratulate Tom on his paper about GUI agents being accepted by a top conference.<br>tom@abc.com<br>G<br>二日 (加味辛(去) (草) (草) (草) (草) (草) (<br>GUI Agent<br>Great, I found two potential matches for your recipient: "tom@abc.com" and "tom s@xyz.com" Could you confirm which one is correct, or let me know if it's neither?<br>I have completed email draft, would you like me to send this email?<br>Sure! To proceed, could you please log in to your Gmail account?<br>User<br>(...user logs into their Gmail account manually)<br>It is "tom@abc.com".<br>Looks good, please send it. -->

<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2025/11/2025_11_04__12_39_04_030a96.jpg"/>

Fig. 31: An illustrative example of human-agent interaction for completing an email sending request.

<!-- Media -->

Lastly, developing a virtual desktop environment for the agent to operate in-one that connects to the user's main desktop session without disrupting their workflow, can significantly enhance the user experience (UX) in human-agent interaction. The picture-in-picture mode implemented in UFO2 334 demonstrates this concept in practice, as illustrated in Figure 32 By allowing the agent to run within a resizable and movable virtualized desktop, users can easily minimize or reposition the agent window as needed. This flexibility improves both the usability and the overall UX of interacting with GUI-based agents.

### 11.5 Customization and Personalization

Effective GUI agents must go beyond generic task completion and provide experiences that are personalized to individual users, adapting to their unique workflows, preferences, and behavioral patterns [45], [605]. Unlike general LLM applications that operate in isolated prompts or conversations, GUI agents work across software environments where user interaction styles can vary significantly. A one-size-fits-all agent may fail to align with how a particular user edits documents, navigates interfaces, or organizes tasks—resulting in friction, inefficiency, or user frustration [606].

For instance, a GUI agent assisting with document editing must learn the user's preferred tone, formatting conventions, and vocabulary. Without this contextual understanding, the agent may offer irrelevant suggestions or enforce formatting inconsistent with the user's intent. Personalization in GUI agents thus requires longitudinal learning, where the agent continually adapts based on prior interactions, fine-tunes its behavior to match user expectations, and preserves consistency across sessions [607].

However, this introduces new challenges. The high variability in user preferences-especially in free-form GUI environments—makes it difficult to define universal personalization strategies. Moreover, collecting and leveraging user-specific data must be done responsibly, raising critical concerns around privacy, data retention, and on-device learning. Striking a balance between effective customization and user trust is particularly important for GUI agents, which often operate over sensitive documents, personal applications, or system-level interfaces.

Potential Solutions: Future research should focus on developing mechanisms for user modeling [608] and preference learning [609], enabling agents to tailor their actions to individual users. Techniques such as reinforcement learning from user feedback [610], collaborative filtering [611], and context-aware computing [612] can help agents learn user preferences over time. Ensuring that personalization is achieved without compromising privacy is essential [613], potentially through on-device learning and anonymized data processing. In a more futuristic, cyberpunk-inspired scenario, agents may inversely generate GUIs tailored to users' needs, enabling greater customization and personalization [614].

### 11.6 Ethical and Regulatory Challenges

LLM-powered GUI agents raise distinct ethical and regulatory concerns due to their ability to perform real-world actions across software interfaces. Unlike traditional LLMs, these agents can autonomously trigger operations, manipulate data, and interact with sensitive applications—amplifying risks around accountability, fairness, and user consent | 548 | 615 |- 618.

A key concern is bias inherited from training data, which can lead to unfair behavior in sensitive workflows. For example, a GUI agent assisting in hiring may unknowingly exhibit gender or racial bias 619 , 620 . These risks are harder to audit at the GUI level due to limited traceability across multi-application actions. Regulatory compliance adds further complexity. GUI agents often operate across domains with strict data protection laws, but lack standardized mechanisms for logging actions or securing user consent. This makes it challenging to meet legal and ethical standards, especially when agents act in opaque or background contexts. Addressing these issues requires tailored solutions for GUI agents, including permission controls, runtime confirmations, and transparent activity logs-ensuring safe, fair, and compliant deployment across diverse environments.


<!-- Meanless: JOURNAL OF LATEX CLASS FILES, DECEMBER 2024<br>84 -->

<!-- Media -->

<!-- figureText: ④.<br>都・一日月<br>4 AgentDesktopWPF - BentDesktopWPF - D<br>都是。步上唯有专门有价<br>Solution Explorer<br>- 0 x<br>MainVM.cs<br>UFODesktopServices<br>AgentDesktopWPF, cspto<br>MainWindow.xami.cs $\rightarrow \times$<br><icon/><br>AgentDesktopWPF<br><icon/><br>0<br>it pipeClient.ConnectAsync( );<br>All Solution 'AgentDesktop' (7 of 7 projects<br>ing (StreamReader reader = new StreamReader(pipeClient))<br>ing (StreamPriter writer = ncw StreamWriter(pipeClient) \{ AutoFlush = true \})<br>string response = analt reader.ReadLineAsync( );<br>Main Desktop<br>Dam Helpers<br>return response;<br>P 6 11 Images<br>338<br>D. 5 C# AgentState.<br>> <c = MainVM.cs<br>P.ac# PlantemVM.c<br>P & C# TaskVM.cs<br>if (!isExiting)<br>A B B Views<br>e.Cancel = true;<br>isSlidingOut = true;<br>Topmost = false;<br>D C# MainWindow.xamil.c<br>slideTimer.Start( );<br>else<br>UFO Desktop<br>除<br>log<br>Create one todo for learning<br>e.Cancel<br>LLM<br>Task Flow<br>Δ 1<br>Open the Microsoft To Do<br>Output<br>the Microsoft Information<br>Virtual Desktop<br>Since the Microsoft To Do<br>application is not visible in the<br>screenshot. I will use the function<br>11<br><icon/><br><icon/><br>Pause task<br>End task<br>日 Ready<br>今日热点<br>L<br>国<br>酸<br>面<br>oftapc-my.sha<br>唐向珺公布高考 -->

<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2025/11/2025_11_04__12_39_04_640fed.jpg"/>

Fig. 32: The Picture-in-Picture interface in ${\mathrm{{UFO}}}^{2}$ : a virtual desktop window enabling non-disruptive automation. Figure adapted from 334.

<!-- Media -->

Potential Solutions: Addressing these concerns requires establishing clear ethical guidelines and regulatory frameworks for the development and use of GUI agents 621. Future work should focus on creating mechanisms for auditing and monitoring agent behavior [622] to ensure compliance with ethical standards and legal requirements 623]. Incorporating bias detection and mitigation strategies in language models can help prevent discriminatory or unfair actions 624]. Providing users with control over data usage and clear information about the agent's capabilities can enhance transparency and trust.

### 11.7 Scalability and Generalization

GUI agents often struggle to scale beyond specific applications or environments, limiting their generalization. Each software interface features unique layouts, styles, and interaction patterns—even common UI elements like pop-up windows can vary widely 625]. These variations make it difficult to design agents that operate robustly across platforms without retraining or fine-tuning.

A further challenge is the dynamic nature of real-world GUIs. Frequent changes due to software updates, A/B testing, or interface redesigns—such as repositioned buttons or modified widget hierarchies—can easily break previously functional agents. For example, an agent trained on one version of a word processor may fail when the layout changes, or when deployed on a different program with similar functionality but a different interface structure. Even when GUIs share visual similarities, agents often fail to generalize without additional exploration or adaptation [626]. This lack of robustness restricts deployment in practical settings and increases the cost of maintenance, requiring frequent updates or retraining to stay aligned with evolving environments 627-629]. Overcoming this challenge remains critical for developing truly scalable and adaptable GUI agents.

Potential Solutions: To enhance scalability and generalization, one solution from the dataset perspective is to create comprehensive GUI agent datasets that cover a wide range of environments, user requests, GUI designs, platforms, and interaction patterns. By exposing the LLM to diverse data sources during training, the model can learn common patterns and develop a more generalized understanding, enabling it to adapt to infer the functionality of new interfaces based on learned similarities [630].

To further enhance adaptability, research can focus on techniques such as transfer learning [631] and meta-learning 632]. Transfer learning involves pre-training a model on a large, diverse dataset and then fine-tuning it on a smaller, task-specific dataset. In the context of GUI agents, this means training the LLM on a wide array of GUI interactions before customizing it for a particular application or domain. Meta-learning, enables the model to rapidly adapt to new tasks with minimal data by identifying underlying structures and patterns across different tasks. These approaches enable agents to generalize from limited data and adapt to new environments with minimal retraining.


<!-- Meanless: JOURNAL OF LATEX CLASS FILES, DECEMBER 2024<br>85 -->

However, even with these measures, the agent may still encounter difficulties in unfamiliar environments. To address this, we advocate for developers to provide helpful knowledge bases, such as guidance documents, application documentation, searchable FAQs, and even human demonstrations on how to use the application [633- 635]. Techniques like RAG [189] can be employed, where the agent retrieves relevant information from a knowledge base at runtime to inform its decisions [636]. For instance, if the agent encounters an unknown interface element, it can query the documentation to understand its purpose and how to interact with it. This approach enhances the agent's capabilities without requiring extensive retraining. Implementing these solutions requires collaborative efforts not only from agent developers but also from application or environment providers.

### 11.8 Summary

LLM-brained GUI agents hold significant promise for automating complex tasks and enhancing user productivity across various applications. However, realizing this potential requires addressing the outlined limitations through dedicated research and development efforts. By addressing these challenges, the community can develop more robust and widely adopted GUI agents.

Collaboration among researchers, industry practitioners, policymakers, and users is essential to navigate these challenges successfully. Establishing interdisciplinary teams can foster innovation and ensure that GUI agents are developed responsibly, with a clear understanding of technical, ethical, and societal implications. As the field progresses, continuous evaluation and adaptation will be crucial to align technological advancements with user needs and expectations, ultimately leading to more intelligent, safe, and user-friendly GUI agents.

## 12 CONCLUSION

The combination of LLMs and GUI automation marks a transformative moment in human-computer interaction. LLMs provide the "brain" for natural language processing, comprehension, and GUI understanding, while GUI automation tools serve as the "hands", translating the agent's cognitive abilities into actionable commands within software environments. Together, they form LLM-powered GUI agents that introduce a new paradigm in user interaction, allowing users to control applications through straightforward natural language commands instead of complex, platform-specific UI operations. This synergy has shown remarkable potential, with applications flourishing in both research and industry.

In this survey, we provide a comprehensive, systematic, and timely overview of the field of LLM-powered GUI agents. Our work introduces the core components and advanced techniques that underpin these agents, while also examining critical elements such as data collection, model development, frameworks, evaluation methodologies, and real-world applications. Additionally, we explore the current limitations and challenges faced by these agents and outline a roadmap for future research directions. We hope this survey serves as a valuable handbook for those learning about LLM-powered GUI agents and as a reference point for researchers aiming to stay at the forefront of developments in this field.

As we look to the future, the concept of LLM-brained GUI agents promises to become increasingly tangible, fundamentally enhancing productivity and accessibility in daily life. With ongoing research and development, this technology stands poised to reshape how we interact with digital systems, transforming complex workflows into seamless, natural interactions.

## REFERENCES

[1] B. J. Jansen, "The graphical user interface," ACM SIGCHI Bull., vol. 30, pp. 22-26, 1998. [Online]. Available: https: //api.semanticscholar.org/CorpusID:18416305

[2] H. Sampath, A. Merrick, and A. P. Macvean, "Accessibility of command line interfaces," Proceedings of the 2021 CHI Conference on Human Factors in Computing Systems, 2021. [Online]. Available: https://api.semanticscholar.org/CorpusID: 233987139

[3] R. Michalski, J. Grobelny, and W. Karwowski, "The effects of graphical interface design characteristics on human-computer interaction task efficiency," ArXiv, vol. abs/1211.6712, 2006. [Online]. Available: https://api.semanticscholar.org/CorpusID: 14695409

[4] T. D. Hellmann and F. Maurer, "Rule-based exploratory testing of graphical user interfaces," in 2011 Agile Conference. IEEE, 2011, pp. 107-116.

[5] J. Steven, P. Chandra, B. Fleck, and A. Podgurski, "jrapture: A capture/replay tool for observation-based testing," SIGSOFT Softw. Eng. Notes, vol. 25, no. 5, p. 158-167, Aug. 2000. [Online]. Available: https://doi.org/10.1145/347636.348993

[6] L. Ivančić, D. Suša Vugec, and V. Bosilj Vukšic, "Robotic process automation: systematic literature review," in Business Process Management: Blockchain and Central and Eastern Europe Forum: BPM 2019 Blockchain and CEE Forum, Vienna, Austria, September 1-6, 2019, Proceedings 17. Springer, 2019, pp. 280-295.

[7] W. contributors, "Large language model - wikipedia, the free encyclopedia," 2024, accessed: 2024-11-25. [Online]. Available: https://en.wikipedia.org/wiki/Large_language_model

[8] W. X. Zhao, K. Zhou, J. Li, T. Tang, X. Wang, Y. Hou, Y. Min, B. Zhang, J. Zhang, Z. Dong et al., "A survey of large language models," arXiv preprint arXiv:2303.18223, 2023.

[9] H. Naveed, A. U. Khan, S. Qiu, M. Sagib, S. Anwar, M. Usman, N. Akhtar, N. Barnes, and A. Mian, "A comprehensive overview of large language models," arXiv preprint arXiv:2307.06435, 2023.

[10] S. Yin, C. Fu, S. Zhao, K. Li, X. Sun, T. Xu, and E. Chen, "A survey on multimodal large language models," arXiv preprint arXiv:2306.13549, 2023.

[11] T. Wu, S. He, J. Liu, S. Sun, K. Liu, Q.-L. Han, and Y. Tang, "A brief overview of chatgpt: The history, status quo and potential future development," IEEE/CAA Journal of Automatica Sinica, vol. 10, no. 5, pp. 1122-1136, 2023.

[12] J. Liu, K. Wang, Y. Chen, X. Peng, Z. Chen, L. Zhang, and Y. Lou, "Large language model-based agents for software engineering: A survey," arXiv preprint arXiv:2409.02977, 2024.

[13] Z. Shen, "LIm with tools: A survey," arXiv preprint arXiv:2409.18807, 2024.

[14] T. Feng, C. Jin, J. Liu, K. Zhu, H. Tu, Z. Cheng, G. Lin, and J. You, "How far are we from agi: Are Ilms all we need?" Transactions on Machine Learning Research.

[15] W. Hong, W. Wang, Q. Lv, J. Xu, W. Yu, J. Ji, Y. Wang, Z. Wang, Y. Zhang, J. Li, B. Xu, Y. Dong, M. Ding, and J. Tang, "Cogagent: A visual language model for gui agents," 2023. [Online]. Available: https://arxiv.org/abs/2312.08914


<!-- Meanless: JOURNAL OF LATEX CLASS FILES, DECEMBER 2024<br>86 -->

[16] M. Xu, "Every software as an agent: Blueprint and case study," arXiv preprint arXiv:2502.04747, 2025.

[17] B. Zheng, B. Gou, J. Kil, H. Sun, and Y. Su, "Gpt-4v(ision) is a generalist web agent, if grounded," 2024. [Online]. Available: https://arxiv.org/abs/2401.01614

[18] C. Zhang, Z. Yang, J. Liu, Y. Han, X. Chen, Z. Huang, B. Fu, and G. Yu, "Appagent: Multimodal agents as smartphone users," 2023. [Online]. Available: https://arxiv.org/abs/2312.13771

[19] C. Zhang, L. Li, S. He, X. Zhang, B. Qiao, S. Qin, M. Ma, Y. Kang, Q. Lin, S. Rajmohan, D. Zhang, and Q. Zhang, "UFO: A UI-Focused Agent for Windows OS Interaction," arXiv preprint arXiv:2402.07939, 2024.

[20] Y. Guan, D. Wang, Z. Chu, S. Wang, F. Ni, R. Song, L. Li, J. Gu, and C. Zhuang, "Intelligent virtual assistants with Ilm-based process automation," ArXiv, vol. abs/2312.06677, 2023. [Online]. Available: https://api.semanticscholar.org/CorpusID:266174422

[21] Y. Zhang, X. Zhao, J. Yin, L. Zhang, and Z. Chen, "Operating system and artificial intelligence: A systematic review," arXiv preprint arXiv:2407.14567, 2024.

[22] K. Mei, Z. Li, S. Xu, R. Ye, Y. Ge, and Y. Zhang, "Aios: LIm agent operating system," arXiv e-prints, pp. arXiv-2403, 2024.

[23] W. Aljedaani, A. Habib, A. Aljohani, M. M. Eler, and Y. Feng, "Does chatgpt generate accessible code? investigating accessibility challenges in Ilm-generated source code," in International Cross-Disciplinary Conference on Web Accessibility, 2024. [Online]. Available: https://api.semanticscholar.org/CorpusID:273550267

[24] D. Chin, Y. Wang, and G. G. Xia, "Human-centered Ilm-agent user interface: A position paper," ArXiv, vol. abs/2405.13050, 2024. [Online]. Available: https://api.semanticscholar.org/CorpusID: 269982753

[25] K. Cheng, Q. Sun, Y. Chu, F. Xu, Y. Li, J. Zhang, and Z. Wu, "Seeclick: Harnessing gui grounding for advanced visual gui agents," 2024. [Online]. Available: https://arxiv.org/abs/2401.10935

[26] M. Zhuge, C. Zhao, D. R. Ashley, W. Wang, D. Khizbullin, Y. Xiong, Z. Liu, E. Chang, R. Krishnamoorthi, Y. Tian, Y. Shi, V. Chandra, and J. Schmidhuber, "Agent-as-a-judge: Evaluate agents with agents," 2024. [Online]. Available: https://api.semanticscholar.org/CorpusID:273350802

[27] K. Li and M. Wu, Effective GUI testing automation: Developing an automated GUI testing tool. John Wiley & Sons, 2006.

[28] O. Rodríguez-Valdés, T. E. Vos, P. Aho, and B. Marín, "30 years of automated gui testing: a bibliometric analysis," in Quality of Information and Communications Technology: 14th International Conference, QUATIC 2021, Algarve, Portugal, September 8-11, 2021, Proceedings 14. Springer, 2021, pp. 473-488.

[29] Y. L. Arnatovich and L. Wang, "A systematic literature review of automated techniques for functional gui testing of mobile applications," arXiv preprint arXiv:1812.11470, 2018.

[30] K. S. Said, L. Nie, A. A. Ajibode, and X. Zhou, "Gui testing for mobile applications: objectives, approaches and challenges," in Proceedings of the 12th Asia-Pacific Symposium on Internetware, 2020, pp. 51-60.

[31] X. Li, "Gui testing for android applications: a survey," in 2023 7th International Conference on Computer, Software and Modeling (ICCSM). IEEE, 2023, pp. 6-10.

[32] J.-J. Oksanen, "Test automation for windows gui application," 2023.

[33] P. S. Deshmukh, S. S. Date, P. N. Mahalle, and J. Barot, "Automated gui testing for enhancing user experience (ux): A survey of the state of the art," in International Conference on ICT for Sustainable Development. Springer, 2023, pp. 619-628.

[34] M. Bajammal, A. Stocco, D. Mazinanian, and A. Mesbah, "A survey on the use of computer vision to improve software engineering tasks," IEEE Transactions on Software Engineering, vol. 48, no. 5, pp. 1722-1742, 2020.

[35] S. Yu, C. Fang, Z. Tuo, Q. Zhang, C. Chen, Z. Chen, and Z. Su, "Vision-based mobile app gui testing: A survey," arXiv preprint arXiv:2310.13518, 2023.

[36] R. Syed, S. Suriadi, M. Adams, W. Bandara, S. J. Leemans, C. Ouyang, A. H. Ter Hofstede, I. Van De Weerd, M. T. Wynn, and H. A. Reijers, "Robotic process automation: contemporary themes and challenges," Computers in Industry, vol. 115, p. 103162, 2020.

[37] T. Chakraborti, V. Isahagian, R. Khalaf, Y. Khazaeni, V. Muthusamy, Y. Rizk, and M. Unuvar, "From robotic process automation to intelligent process automation: -emerging trends-," in Business Process Management: Blockchain and Robotic Process Automation Forum: BPM 2020 Blockchain and RPA Forum, Seville, Spain, September 13-18, 2020, Proceedings 18. Springer, 2020, pp. 215-228.

[38] J. G. Enríquez, A. Jiménez-Ramírez, F. J. Domínguez-Mayo, and J. A. García-García, "Robotic process automation: a scientific and industrial systematic mapping study," IEEE Access, vol. 8, pp. 39113-39129, 2020.

[39] J. Ribeiro, R. Lima, T. Eckhardt, and S. Paiva, "Robotic process automation and artificial intelligence in industry 4.0-a literature review," Procedia Computer Science, vol. 181, pp. 51-58, 2021.

[40] M. Nass, E. Alégroth, and R. Feldt, "Why many challenges with gui test automation (will) remain," Information and Software Technology, vol. 138, p. 106625, 2021.

[41] S. Agostinelli, A. Marrella, and M. Mecella, "Research challenges for intelligent robotic process automation," in Business Process Management Workshops: BPM 2019 International Workshops, Vienna, Austria, September 1-6, 2019, Revised Selected Papers 17. Springer, 2019, pp. 12-18.

[42] A. Wali, S. Mahamad, and S. Sulaiman, "Task automation intelligent agents: A review," Future Internet, vol. 15, no. 6, p. 196, 2023.

[43] P. Zhao, Z. Jin, and N. Cheng, "An in-depth survey of large language model-based artificial intelligence agents," arXiv preprint arXiv:2309.14365, 2023.

[44] Y. Cheng, C. Zhang, Z. Zhang, X. Meng, S. Hong, W. Li, Z. Wang, Z. Wang, F. Yin, J. Zhao et al., "Exploring large language model based intelligent agents: Definitions, methods, and prospects," arXiv preprint arXiv:2401.03428, 2024.

[45] Y. Li, H. Wen, W. Wang, X. Li, Y. Yuan, G. Liu, J. Liu, W. Xu, X. Wang, Y. Sun et al., "Personal Ilm agents: Insights and survey about the capability, efficiency and security," arXiv preprint arXiv:2401.05459, 2024.

[46] Z. Xi, W. Chen, X. Guo, W. He, Y. Ding, B. Hong, M. Zhang, J. Wang, S. Jin, E. Zhou et al., "The rise and potential of large language model based agents: A survey," arXiv preprint arXiv:2309.07864, 2023.

[47] L. Wang, C. Ma, X. Feng, Z. Zhang, H. Yang, J. Zhang, Z. Chen, J. Tang, X. Chen, Y. Lin et al., "A survey on large language model based autonomous agents," Frontiers of Computer Science, vol. 18, no. 6, p. 186345, 2024.

[48] T. Guo, X. Chen, Y. Wang, R. Chang, S. Pei, N. V. Chawla, O. Wiest, and X. Zhang, "Large language model based multi-agents: A survey of progress and challenges," arXiv preprint arXiv:2402.01680, 2024.

[49] S. Han, Q. Zhang, Y. Yao, W. Jin, Z. Xu, and C. He, "LIm multi-agent systems: Challenges and open problems," arXiv preprint arXiv:2402.03578, 2024.

[50] C. Sun, S. Huang, and D. Pompili, "Llm-based multi-agent reinforcement learning: Current and future directions," arXiv preprint arXiv:2405.11106, 2024.

[51] X. Huang, W. Liu, X. Chen, X. Wang, H. Wang, D. Lian, Y. Wang, R. Tang, and E. Chen, "Understanding the planning of Ilm agents: A survey," arXiv preprint arXiv:2402.02716, 2024.

[52] M. Aghzal, E. Plaku, G. J. Stein, and Z. Yao, "A survey on large language models for automated planning," arXiv preprint arXiv:2502.12435, 2025.

[53] J. Zheng, C. Shi, X. Cai, Q. Li, D. Zhang, C. Li, D. Yu, and Q. Ma, "Lifelong learning of large language model based agents: A roadmap," arXiv preprint arXiv:2501.07278, 2025.

[54] Z. Zhang, X. Bo, C. Ma, R. Li, X. Chen, Q. Dai, J. Zhu, Z. Dong, and J.-R. Wen, "A survey on the memory mechanism of large language model based agents," arXiv preprint arXiv:2404.13501, 2024.

[55] Y. Chang, X. Wang, J. Wang, Y. Wu, L. Yang, K. Zhu, H. Chen, X. Yi, C. Wang, Y. Wang et al., "A survey on evaluation of large language models," ACM Transactions on Intelligent Systems and Technology, vol. 15, no. 3, pp. 1-45, 2024.

[56] L. Li, G. Chen, H. Shi, J. Xiao, and L. Chen, "A survey on multimodal benchmarks: In the era of large ai models," arXiv preprint arXiv:2409.18142, 2024.

[57] Z. Li, X. Wu, H. Du, H. Nghiem, and G. Shi, "Benchmark evaluations, applications, and challenges of large vision language models: A survey," arXiv preprint arXiv:2501.02189, 2025.

[58] J. Huang and J. Zhang, "A survey on evaluation of multimodal large language models," arXiv preprint arXiv:2408.15769, 2024.

[59] J. Xie, Z. Chen, R. Zhang, X. Wan, and G. Li, "Large multimodal agents: A survey," arXiv preprint arXiv:2402.15116, 2024.

[60] Z. Durante, Q. Huang, N. Wake, R. Gong, J. S. Park, B. Sarkar, R. Taori, Y. Noda, D. Terzopoulos, Y. Choi et al., "Agent ai: Surveying the horizons of multimodal interaction," arXiv preprint arXiv:2401.03568, 2024.


<!-- Meanless: JOURNAL OF LATEX CLASS FILES, DECEMBER 2024<br>87 -->

[61] B. Wu, Y. Li, M. Fang, Z. Song, Z. Zhang, Y. Wei, and L. Chen, "Foundations and recent trends in multimodal mobile agents: A survey," arXiv preprint arXiv:2411.02006, 2024.

[62] S. Wang, W. Liu, J. Chen, W. Gan, X. Zeng, S. Yu, X. Hao, K. Shao, Y. Wang, and R. Tang, "Gui agents with foundation models: A comprehensive survey," 2024. [Online]. Available: https://arxiv.org/abs/2411.04890

[63] M. Gao, W. Bu, B. Miao, Y. Wu, Y. Li, J. Li, S. Tang, Q. Wu, Y. Zhuang, and M. Wang, "Generalist virtual agents: A survey on autonomous agents across digital platforms," arXiv preprint arXiv:2411.10943, 2024.

[64] D. Nguyen, J. Chen, Y. Wang, G. Wu, N. Park, Z. Hu, H. Lyu, J. Wu, R. Aponte, Y. Xia, X. Li, J. Shi, H. Chen, V. D. Lai, Z. Xie, S. Kim, R. Zhang, T. Yu, M. Tanjim, N. K. Ahmed, P. Mathur, S. Yoon, L. Yao, B. Kveton, T. H. Nguyen, T. Bui, T. Zhou, R. A. Rossi, and F. Dernoncourt, "Gui agents: A survey," 2024. [Online]. Available: https://arxiv.org/abs/2412.13501

[65] G. Liu, P. Zhao, L. Liu, Y. Guo, H. Xiao, W. Lin, Y. Chai, Y. Han, S. Ren, H. Wang et al., "Llm-powered gui agents in phone automation: Surveying progress and prospects," arXiv preprint arXiv:2504.19838, 2025.

[66] X. Hu, T. Xiong, B. Yi, Z. Wei, R. Xiao, Y. Chen, J. Ye, M. Tao, X. Zhou, Z. Zhao et al., "Os agents: A survey on mllm-based agents for general computing devices use," 2024.

[67] Y. Shi, W. Yu, W. Yao, W. Chen, and N. Liu, "Towards trustworthy gui agents: A survey," arXiv preprint arXiv:2503.23434, 2025.

[68] L. Ning, Z. Liang, Z. Jiang, H. Qu, Y. Ding, W. Fan, X.-y. Wei, S. Lin, H. Liu, P. S. Yu et al., "A survey of webagents: Towards next-generation ai agents for web automation with large foundation models," arXiv preprint arXiv:2503.23350, 2025.

[69] F. Tang, H. Xu, H. Zhang, S. Chen, X. Wu, Y. Shen, W. Zhang, G. Hou, Z. Tan, Y. Yan, K. Song, J. Shao, W. Lu, J. Xiao, and Y. Zhuang, "A survey on (m)Ilm-based gui agents," 2025. [Online]. Available: https://arxiv.org/abs/2504.13865

[70] J. Li and K. Huang, "A summary on gui agents with foundation models enhanced by reinforcement learning," 2025. [Online]. Available: https://arxiv.org/abs/2504.20464

[71] P. J. Sager, B. Meyer, P. Yan, R. von Wartburg-Kottler, L. Etaiwi, A. Enayati, G. Nobel, A. Abdulkadir, B. F. Grewe, and T. Stadel-mann, "Ai agents for computer use: A review of instruction-based computer control, gui automation, and operator assistants," arXiv preprint arXiv:2501.16150, 2025.

[72] T. S. d. Moura, E. L. Alves, H. F. d. Figueirèdo, and C. d. S. Baptista, "Cytestion: Automated gui testing for web applications," in Proceedings of the XXXVII Brazilian Symposium on Software Engineering, 2023, pp. 388-397.

[73] T. Yeh, T.-H. Chang, and R. C. Miller, "Sikuli: using gui screenshots for search and automation," in Proceedings of the 22nd annual ACM symposium on User interface software and technology, 2009, pp. 183-192.

[74] C. E. Shannon, "Prediction and entropy of printed english," Bell system technical journal, vol. 30, no. 1, pp. 50-64, 1951.

[75] W. B. Cavnar, J. M. Trenkle et al., "N-gram-based text categorization," in Proceedings of SDAIR-94, 3rd annual symposium on document analysis and information retrieval, vol. 161175. Ann Arbor, Michigan, 1994, p. 14.

[76] J. Chung, C. Gulcehre, K. Cho, and Y. Bengio, "Empirical evaluation of gated recurrent neural networks on sequence modeling," arXiv preprint arXiv:1412.3555, 2014.

[77] B. Mann, N. Ryder, M. Subbiah, J. Kaplan, P. Dhariwal, A. Neelakantan, P. Shyam, G. Sastry, A. Askell, S. Agarwal et al., "Language models are few-shot learners," arXiv preprint arXiv:2005.14165, vol. 1, 2020.

[78] J. Wei, M. Bosma, V. Y. Zhao, K. Guu, A. W. Yu, B. Lester, N. Du, A. M. Dai, and Q. V. Le, "Finetuned language models are zero-shot learners," arXiv preprint arXiv:2109.01652, 2021.

[79] L. R. Medsker, L. Jain et al., "Recurrent neural networks," Design and Applications, vol. 5, no. 64-67, p. 2, 2001.

[80] S. Hochreiter, "Long short-term memory," Neural Computation MIT-Press, 1997.

[81] A. Vaswani, "Attention is all you need," Advances in Neural Information Processing Systems, 2017.

[82] J. Devlin, "Bert: Pre-training of deep bidirectional transformers for language understanding," arXiv preprint arXiv:1810.04805, 2018.

[83] Y. Liu, "Roberta: A robustly optimized bert pretraining approach," arXiv preprint arXiv:1907.11692, vol. 364, 2019.

[84] Z. Lan, "Albert: A lite bert for self-supervised learning of language representations," arXiv preprint arXiv:1909.11942, 2019.

[85] A. Radford, "Improving language understanding by generative pre-training," 2018.

[86] A. Radford, J. Wu, R. Child, D. Luan, D. Amodei, I. Sutskever et al., "Language models are unsupervised multitask learners," OpenAI blog, vol. 1, no. 8, p. 9, 2019.

[87] C. Raffel, N. Shazeer, A. Roberts, K. Lee, S. Narang, M. Matena, Y. Zhou, W. Li, and P. J. Liu, "Exploring the limits of transfer learning with a unified text-to-text transformer," Journal of machine learning research, vol. 21, no. 140, pp. 1-67, 2020.

[88] M. Lewis, "Bart: Denoising sequence-to-sequence pre-training for natural language generation, translation, and comprehension," arXiv preprint arXiv:1910.13461, 2019.

[89] L. Ouyang, J. Wu, X. Jiang, D. Almeida, C. Wainwright, P. Mishkin, C. Zhang, S. Agarwal, K. Slama, A. Ray et al., "Training language models to follow instructions with human feedback," Advances in neural information processing systems, vol. 35, pp. 27730-27744, 2022.

[90] J. Achiam, S. Adler, S. Agarwal, L. Ahmad, I. Akkaya, F. L. Aleman, D. Almeida, J. Altenschmidt, S. Altman, S. Anadkat et al., "Gpt-4 technical report," arXiv preprint arXiv:2303.08774, 2023.

[91] A. Dubey, A. Jauhri, A. Pandey, A. Kadian, A. Al-Dahle, A. Letman, A. Mathur, A. Schelten, A. Yang, A. Fan et al., "The Ilama 3 herd of models," arXiv preprint arXiv:2407.21783, 2024.

[92] G. Team, R. Anil, S. Borgeaud, J.-B. Alayrac, J. Yu, R. Soricut, J. Schalkwyk, A. M. Dai, A. Hauth, K. Millican et al., "Gemini: a family of highly capable multimodal models," arXiv preprint arXiv:2312.11805, 2023.

[93] A. Hurst, A. Lerer, A. P. Goucher, A. Perelman, A. Ramesh, A. Clark, A. Ostrow, A. Welihinda, A. Hayes, A. Radford et al., "Gpt-40 system card," arXiv preprint arXiv:2410.21276, 2024.

[94] Y. Jiang, C. Zhang, S. He, Z. Yang, M. Ma, S. Qin, Y. Kang, Y. Dang, S. Rajmohan, Q. Lin et al., "Xpert: Empowering incident management with query recommendations via large language models," in Proceedings of the IEEE/ACM 46th International Conference on Software Engineering, 2024, pp. 1-13.

[95] C. Zhang, Z. Ma, Y. Wu, S. He, S. Qin, M. Ma, X. Qin, Y. Kang, Y. Liang, X. Gou et al., "Allhands: Ask me anything on large-scale verbatim feedback via large language models," arXiv preprint arXiv:2403.15157, 2024.

[96] J. Liu, C. Zhang, J. Qian, M. Ma, S. Qin, C. Bansal, Q. Lin, S. Rajmohan, and D. Zhang, "Large language models can deliver accurate and interpretable time series anomaly detection," arXiv preprint arXiv:2405.15370, 2024.

[97] Q. Dong, L. Li, D. Dai, C. Zheng, J. Ma, R. Li, H. Xia, J. Xu, Z. Wu, T. Liu et al., "A survey on in-context learning," arXiv preprint arXiv:2301.00234, 2022.

[98] S. Zhang, L. Dong, X. Li, S. Zhang, X. Sun, S. Wang, J. Li, R. Hu, T. Zhang, F. Wu et al., "Instruction tuning for large language models: A survey," arXiv preprint arXiv:2308.10792, 2023.

[99] J. Huang and K. C.-C. Chang, "Towards reasoning in large language models: A survey," arXiv preprint arXiv:2212.10403, 2022.

[100] J. Wei, X. Wang, D. Schuurmans, M. Bosma, F. Xia, E. Chi, Q. V. Le, D. Zhou et al., "Chain-of-thought prompting elicits reasoning in large language models," Advances in neural information processing systems, vol. 35, pp. 24824-24837, 2022.

[101] R. Ding, C. Zhang, L. Wang, Y. Xu, M. Ma, W. Zhang, S. Qin, S. Rajmohan, Q. Lin, and D. Zhang, "Everything of thoughts: Defying the law of penrose triangle for thought generation," arXiv preprint arXiv:2311.04254, 2023.

[102] M. Chen, J. Tworek, H. Jun, Q. Yuan, H. P. D. O. Pinto, J. Kaplan, H. Edwards, Y. Burda, N. Joseph, G. Brockman et al., "Evaluating large language models trained on code," arXiv preprint arXiv:2107.03374, 2021.

[103] T. D. White, G. Fraser, and G. J. Brown, "Improving random gui testing with image-based widget detection," in Proceedings of the 28th ACM SIGSOFT international symposium on software testing and analysis, 2019, pp. 307-317.

[104] G. Kim, P. Baldi, and S. McAleer, "Language models can solve computer tasks," 2023. [Online]. Available: https: //arxiv.org/abs/2303.17491


<!-- Meanless: JOURNAL OF LATEX CLASS FILES, DECEMBER 2024<br>88 -->

[105] B. Qiao, L. Li, X. Zhang, S. He, Y. Kang, C. Zhang, F. Yang, H. Dong, J. Zhang, L. Wang et al., "Taskweaver: A code-first agent framework," arXiv preprint arXiv:2311.17541, 2023.

[106] M. A. Boshart and M. J. Kosa, "Growing a gui from an xml tree," ACM SIGCSE Bulletin, vol. 35, no. 3, pp. 223-223, 2003.

[107] Y. Li and O. Hilliges, Artificial intelligence for human computer interaction: a modern approach. Springer, 2021.

[108] H. Y. Abuaddous, A. M. Saleh, O. Enaizan, F. Ghabban, and A. B. Al-Badareen, "Automated user experience (ux) testing for mobile application: Strengths and limitations." International Journal of Interactive Mobile Technologies, vol. 16, no. 4, 2022.

[109] D. Gao, L. Ji, Z. Bai, M. Ouyang, P. Li, D. Mao, Q. Wu, W. Zhang, P. Wang, X. Guo et al., "Assistgui: Task-oriented pc graphical user interface automation," in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 13289- 13298.

[110] J. Qian, Z. Shang, S. Yan, Y. Wang, and L. Chen, "Roscript: A visual script driven truly non-intrusive robotic testing system for touch screen applications," in 2020 IEEE/ACM 42nd International Conference on Software Engineering (ICSE), 2020, pp. 297-308.

[111] A. Bruns, A. Kornstadt, and D. Wichmann, "Web application tests with selenium," IEEE software, vol. 26, no. 5, pp. 88-91, 2009.

[112] N. Rupp, K. Peschke, M. Köppl, D. Drissner, and T. Zuchner, "Establishment of low-cost laboratory automation processes using autoit and 4-axis robots," SLAS technology, vol. 27, no. 5, pp. 312-318, 2022.

[113] M. F. Granda, O. Parra, and B. Alba-Sarango, "Towards a model-driven testing framework for gui test cases generation from user stories." in ENASE, 2021, pp. 453-460.

[114] J. Xu, W. Du, X. Liu, and X. Li, "Llm4workflow: An Ilm-based automated workflow model generation tool," Proceedings of the 39th IEEE/ACM International Conference on Automated Software Engineering, 2024. [Online]. Available: https://api.semanticscholar.org/CorpusID:273465368

[115] R. Gove and J. Faytong, "Machine learning and event-based software testing: classifiers for identifying infeasible gui event sequences," in Advances in computers. Elsevier, 2012, vol. 86, pp. 109-135.

[116] T. J.-J. Li, L. Popowski, T. Mitchell, and B. A. Myers, "Screen2vec: Semantic embedding of gui screens and gui components," in Proceedings of the 2021 CHI Conference on Human Factors in Computing Systems, 2021, pp. 1-15.

[117] T.-H. Chang, T. Yeh, and R. C. Miller, "Gui testing using computer vision," in Proceedings of the SIGCHI Conference on Human Factors in Computing Systems, 2010, pp. 1535-1544.

[118] Z. Zou, K. Chen, Z. Shi, Y. Guo, and J. Ye, "Object detection in 20 years: A survey," Proceedings of the IEEE, vol. 111, no. 3, pp. 257-276, 2023.

[119] J. Ye, K. Chen, X. Xie, L. Ma, R. Huang, Y. Chen, Y. Xue, and J. Zhao, "An empirical study of gui widget detection for industrial mobile games," in Proceedings of the 29th ACM Joint Meeting on European Software Engineering Conference and Symposium on the Foundations of Software Engineering, 2021, pp. 1427-1437.

[120] J. Chen, M. Xie, Z. Xing, C. Chen, X. Xu, L. Zhu, and G. Li, "Object detection for graphical user interface: Old fashioned or deep learning or a combination?" in proceedings of the 28th ACM joint meeting on European Software Engineering Conference and Symposium on the Foundations of Software Engineering, 2020, pp. 1202-1214.

[121] J. Qian, Y. Ma, C. Lin, and L. Chen, "Accelerating ocr-based widget localization for test automation of gui applications," in Proceedings of the 37th IEEE/ACM International Conference on Automated Software Engineering, 2022, pp. 1-13.

[122] O. Gambino, L. Rundo, V. Cannella, S. Vitabile, and R. Pirrone, "A framework for data-driven adaptive gui generation based on dicom," Journal of biomedical informatics, vol. 88, pp. 37-52, 2018.

[123] J. He, I.-L. Yen, T. Peng, J. Dong, and F. Bastani, "An adaptive user interface generation framework for web services," in 2008 IEEE Congress on Services Part II (services-2 2008). IEEE, 2008, pp. 175-182.

[124] Z. Stefanidi, G. Margetis, S. Ntoa, and G. Papagiannakis, "Real-time adaptation of context-aware intelligent user interfaces, for enhanced situational awareness," IEEE Access, vol. 10, pp. 23367- 23393, 2022.

[125] Z. Liu, C. Chen, J. Wang, M. Chen, B. Wu, X. Che, D. Wang, and Q. Wang, "Make Ilm a testing expert: Bringing human-like interaction to mobile gui testing via functionality-aware decisions," in Proceedings of the IEEE/ACM 46th International Conference on Software Engineering, 2024, pp. 1-13.

[126] P. Brie, N. Burny, A. Sluўters, and J. Vanderdonckt, "Evaluating a large language model on searching for gui layouts," Proceedings of the ACM on Human-Computer Interaction, vol. 7, no. EICS, pp. 1-37, 2023.

[127] T. Wetzlmaier, R. Ramler, and W. Putschögl, "A framework for monkey gui testing," in 2016 IEEE international conference on software testing, verification and validation (ICST). IEEE, 2016, pp. 416-423.

[128] X. Zeng, D. Li, W. Zheng, F. Xia, Y. Deng, W. Lam, W. Yang, and T. Xie, "Automated test input generation for android: are we really there yet in an industrial case?" in Proceedings of the 2016 24th ACM SIGSOFT International Symposium on Foundations of Software Engineering, ser. FSE 2016. New York, NY, USA: Association for Computing Machinery, 2016, p. 987-992. [Online]. Available: https://doi.org/10.1145/2950290.2983958

[129] A. M. Memon, M. E. Pollack, and M. L. Soffa, "Hierarchical gui test case generation using automated planning," IEEE transactions on software engineering, vol. 27, no. 2, pp. 144-155, 2001.

[130] S. Agostinelli, M. Lupia, A. Marrella, and M. Mecella, "Automated generation of executable rpa scripts from user interface logs," in Business Process Management: Blockchain and Robotic Process Automation Forum: BPM 2020 Blockchain and RPA Forum, Seville, Spain, September 13-18, 2020, Proceedings 18. Springer, 2020, pp. 116-131.

[131] A. Memon, I. Banerjee, N. Hashmi, and A. Nagarajan, "Dart: a framework for regression testing "nightly/daily builds" of gui applications," in International Conference on Software Maintenance, 2003. ICSM 2003. Proceedings., 2003, pp. 410-419.

[132] Microsoft, "Create desktop flows using record with copilot (preview)," 2024, accessed: 2024-11-16. [Online]. Available: https://learn.microsoft.com/en-us/power-automate/ desktop-flows/create-flow-using-ai-recorder

[133] selenium. (2024) Selenium: Browser automation. Accessed: 2024-11-05. [Online]. Available: https://www.selenium.dev/

[134] appium. (2024) Appium: Cross-platform automation framework for all kinds of apps. Accessed: 2024-11-05. [Online]. Available: https://appium.io/docs/en/latest/

[135] smartbear. (2024) Testcomplete: Automated ui testing tool. Accessed: 2024-11-05. [Online]. Available: https://smartbear.com/ product/testcomplete/

[136] katalon. (2024) Katalon studio: Easy test automation for web, api, mobile, and desktop. Accessed: 2024-11-05. [Online]. Available: https://katalon.com/katalon-studio

[137] ranorex. (2024) Ranorex studio: Test automation for gui testing. Accessed: 2024-11-05. [Online]. Available: https: //www.ranorex.com/

[138] G. Hu, L. Zhu, and J. Yang, "Appflow: using machine learning to synthesize robust, reusable ui tests," in Proceedings of the 2018 26th ACM Joint Meeting on European Software Engineering Conference and Symposium on the Foundations of Software Engineering, ser. ESEC/FSE 2018. New York, NY, USA: Association for Computing Machinery, 2018, p. 269-282. [Online]. Available: https://doi.org/10.1145/3236024.3236055

[139] Y. Li, Z. Yang, Y. Guo, and X. Chen, "Humanoid: A deep learning-based approach to automated black-box android app testing," in 2019 34th IEEE/ACM International Conference on Automated Software Engineering (ASE). IEEE, 2019, pp. 1070-1073.

[140] F. YazdaniBanafsheDaragh and S. Malek, "Deep gui: Black-box gui input generation with deep learning," in 2021 36th IEEE/ACM International Conference on Automated Software Engineering (ASE). IEEE, 2021, pp. 905-916.

[141] M. Xie, S. Feng, Z. Xing, J. Chen, and C. Chen, "Uied: a hybrid tool for gui element detection," in Proceedings of the 28th ACM Joint Meeting on European Software Engineering Conference and Symposium on the Foundations of Software Engineering, 2020, pp. 1655-1659.

[142] N. Xu, S. Masling, M. Du, G. Campagna, L. Heck, J. Landay, and M. S. Lam, "Grounding open-domain instructions to automate web support tasks," 2021. [Online]. Available: https://arxiv.org/abs/2103.16057

[143] S. Mazumder and O. Riva, "Flin: A flexible natural language interface for web navigation," arXiv preprint arXiv:2010.12844, 2020.

[144] Y. Li, J. He, X. Zhou, Y. Zhang, and J. Baldridge, "Mapping natural language instructions to mobile ui action sequences," in Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, 2020, pp. 8198-8210.


<!-- Meanless: JOURNAL OF LATEX CLASS FILES, DECEMBER 2024<br>89 -->

[145] T. Shi, A. Karpathy, L. Fan, J. Hernandez, and P. Liang, "World of bits: An open-domain platform for web-based agents," in Proceedings of the 34th International Conference on Machine Learning, ser. Proceedings of Machine Learning Research, D. Precup and Y. W. Teh, Eds., vol. 70. PMLR, 06-11 Aug 2017, pp. 3135-3144. [Online]. Available: https://proceedings.mlr.press/v70/shi17a.html

[146] E. Z. Liu, K. Guu, P. Pasupat, T. Shi, and P. Liang, "Reinforcement learning on web interfaces using workflow-guided exploration," 2018. [Online]. Available: https://arxiv.org/abs/1802.08802

[147] Y. Lan, Y. Lu, Z. Li, M. Pan, W. Yang, T. Zhang, and X. Li, "Deeply reinforcing android gui testing with deep reinforcement learning," in Proceedings of the 46th IEEE/ACM International Conference on Software Engineering, 2024, pp. 1-13.

[148] D. Toyama, P. Hamel, A. Gergely, G. Comanici, A. Glaese, Z. Ahmed, T. Jackson, S. Mourad, and D. Precup, "Androidenv: A reinforcement learning platform for android," arXiv preprint arXiv:2105.13231, 2021.

[149] S. Yao, H. Chen, J. Yang, and K. Narasimhan, "Webshop: Towards scalable real-world web interaction with grounded language agents," Advances in Neural Information Processing Systems, vol. 35, pp. 20744-20757, 2022.

[150] C. Zhang, P. Patras, and H. Haddadi, "Deep learning in mobile and wireless networking: A survey," IEEE Communications surveys & tutorials, vol. 21, no. 3, pp. 2224-2287, 2019.

[151] P. Martins, F. Sá, F. Morgado, and C. Cunha, "Using machine learning for cognitive robotic process automation (rpa)," in 2020 15th Iberian Conference on Information Systems and Technologies (CISTI). IEEE, 2020, pp. 1-6.

[152] I. Gur, H. Furuta, A. Huang, M. Safdari, Y. Matsuo, D. Eck, and A. Faust, "A real-world webagent with planning, long context understanding, and program synthesis," arXiv preprint arXiv:2307.12856, 2023.

[153] H. Furuta, K.-H. Lee, O. Nachum, Y. Matsuo, A. Faust, S. S. Gu, and I. Gur, "Multimodal web navigation with instruction-finetuned foundation models," arXiv preprint arXiv:2305.11854, 2023.

[154] K. Ma, H. Zhang, H. Wang, X. Pan, W. Yu, and D. Yu, "Laser: LIm agent with state-space exploration for web navigation," arXiv preprint arXiv:2309.08172, 2023.

[155] Y. Deng, X. Zhang, W. Zhang, Y. Yuan, S.-K. Ng, and T.-S. Chua, "On the multi-turn instruction following for conversational web agents," arXiv preprint arXiv:2402.15057, 2024.

[156] H. Wen, Y. Li, G. Liu, S. Zhao, T. Yu, T. J.-J. Li, S. Jiang, Y. Liu, Y. Zhang, and Y. Liu, "Autodroid: LIm-powered task automation in android," in Proceedings of the 30th Annual International Conference on Mobile Computing and Networking, 2024, pp. 543- 557.

[157] A. Yan, Z. Yang, W. Zhu, K. Lin, L. Li, J. Wang, J. Yang, Y. Zhong, J. McAuley, J. Gao et al., "Gpt-4v in wonderland: Large multimodal models for zero-shot smartphone gui navigation," arXiv preprint arXiv:2311.07562, 2023.

[158] J. Wang, H. Xu, J. Ye, M. Yan, W. Shen, J. Zhang, F. Huang, and J. Sang, "Mobile-agent: Autonomous multi-modal mobile device agent with visual perception," 2024. [Online]. Available: https://arxiv.org/abs/2401.16158

[159] S. Nong, J. Zhu, R. Wu, J. Jin, S. Shan, X. Huang, and W. Xu, "Mobileflow: A multimodal Ilm for mobile gui agent," 2024. [Online]. Available: https://arxiv.org/abs/2407.04346

[160] J. Zhang, J. Wu, Y. Teng, M. Liao, N. Xu, X. Xiao, Z. Wei, and D. Tang, "Android in the zoo: Chain-of-action-thought for gui agents," arXiv preprint arXiv:2403.02713, 2024.

[161] W. Tan, W. Zhang, X. Xu, H. Xia, Z. Ding, B. Li, B. Zhou, J. Yue, J. Jiang, Y. Li, R. An, M. Qin, C. Zong, L. Zheng, Y. Wu, X. Chai, Y. Bi, T. Xie, P. Gu, X. Li, C. Zhang, L. Tian, C. Wang, X. Wang, B. F. Karlsson, B. An, S. Yan, and Z. Lu, "Cradle: Empowering foundation agents towards general computer control," 2024. [Online]. Available: https://arxiv.org/abs/2403.03186

[162] Z. Wu, C. Han, Z. Ding, Z. Weng, Z. Liu, S. Yao, T. Yu, and L. Kong, "Os-copilot: Towards generalist computer agents with self-improvement," 2024. [Online]. Available: https://arxiv.org/abs/2402.07456

[163] Anthropic. (2024) Introducing computer use, a new claude 3.5 sonnet, and claude 3.5 haiku. Accessed: 2024- 10-26. [Online]. Available: https://www.anthropic.com/news/ 3-5-models-and-computer-use

[164] S. Hu, M. Ouyang, D. Gao, and M. Z. Shou, "The dawn of gui agent: A preliminary case study with claude 3.5 computer use," 2024. [Online]. Available: https://arxiv.org/abs/2411.10323

[165] OpenAI, "Computer-using agent: Introducing a universal interface for ai to interact with the digital world," 2025. [Online]. Available: https://openai.com/index/computer-using-agent

[166] C. Zhang, S. He, L. Li, S. Qin, Y. Kang, Q. Lin, and D. Zhang, "Api agents vs. gui agents: Divergence and convergence," arXiv preprint arXiv:2503.11069, 2025.

[167] A. M. Memon, I. Banerjee, and A. Nagarajan, "Gui ripping: reverse engineering of graphical user interfaces for testing." in WCRE, vol. 3, 2003, p. 260.

[168] J. Wang, Z. Liu, L. Zhao, Z. Wu, C. Ma, S. Yu, H. Dai, Q. Yang, Y. Liu, S. Zhang et al., "Review of large vision models and visual prompt engineering," Meta-Radiology, p. 100047, 2023.

[169] S. Mitra and T. Acharya, "Gesture recognition: A survey," IEEE Transactions on Systems, Man, and Cybernetics, Part C (Applications and Reviews), vol. 37, no. 3, pp. 311-324, 2007.

[170] R. Hardy and E. Rukzio, "Touch & interact: touch-based interaction of mobile phones with displays," in Proceedings of the 10th international conference on Human computer interaction with mobile devices and services, 2008, pp. 245-254.

[171] H. Lee, J. Park, and U. Lee, "A systematic survey on android api usage for data-driven analytics with smartphones," ACM Computing Surveys, vol. 55, no. 5, pp. 1-38, 2022.

[172] K. Jokinen, "User interaction in mobile navigation applications," in Map-based Mobile Services: Design, Interaction and Usability. Springer, 2008, pp. 168-197.

[173] W. Enck, D. Octeau, P. D. McDaniel, and S. Chaudhuri, "A study of android application security." in USENIX security symposium, vol. 2, no. 2, 2011.

[174] M. Egele, C. Kruegel, E. Kirda, and G. Vigna, "Pios: Detecting privacy leaks in ios applications." in NDSS, vol. 2011, 2011, p. 18th.

[175] B. Sierkowski, "Achieving web accessibility," in Proceedings of the 30th annual ACM SIGUCCS conference on User services, 2002, pp. 288-291.

[176] N. Fernandes, R. Lopes, and L. Carriço, "On web accessibility evaluation environments," in Proceedings of the International Cross-Disciplinary Conference on Web Accessibility, 2011, pp. 1-10.

[177] J. J. Garrett et al., "Ajax: A new approach to web applications," 2005.

[178] J. Yang, H. Zhang, F. Li, X. Zou, C. Li, and J. Gao, "Set-of-mark prompting unleashes extraordinary visual grounding in gpt-4v," arXiv preprint arXiv:2310.11441, 2023.

[179] X. Wu, J. Ye, K. Chen, X. Xie, Y. Hu, R. Huang, L. Ma, and J. Zhao, "Widget detection-based testing for industrial mobile games," in 2023 IEEE/ACM 45th International Conference on Software Engineering: Software Engineering in Practice (ICSE-SEIP). IEEE, 2023, pp. 173-184.

[180] E. Gamma, "Design patterns: elements of reusable object-oriented software," Person Education Inc, 1995.

[181] F. Wang, Z. Zhang, X. Zhang, Z. Wu, T. Mo, Q. Lu, W. Wang, R. Li, J. Xu, X. Tang, Q. He, Y. Ma, M. Huang, and S. Wang, "A comprehensive survey of small language models in the era of large language models: Techniques, enhancements, applications, collaboration with llms, and trustworthiness," 2024. [Online]. Available: https://arxiv.org/abs/2411.03350

[182] A. Kirillov, E. Mintun, N. Ravi, H. Mao, C. Rolland, L. Gustafson, T. Xiao, S. Whitehead, A. C. Berg, W.-Y. Lo et al., "Segment anything," in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 4015-4026.

[183] S. Liu, Z. Zeng, T. Ren, F. Li, H. Zhang, J. Yang, Q. Jiang, C. Li, J. Yang, H. Su et al., "Grounding dino: Marrying dino with grounded pre-training for open-set object detection," arXiv preprint arXiv:2303.05499, 2023.

[184] Y. Lu, J. Yang, Y. Shen, and A. Awadallah, "Omniparser for pure vision based gui agent," arXiv preprint arXiv:2408.00203, 2024.

[185] K. Moran, C. Watson, J. Hoskins, G. Purnell, and D. Poshyvanyk, "Detecting and summarizing gui changes in evolving mobile apps," in Proceedings of the 33rd ACM/IEEE international conference on automated software engineering, 2018, pp. 543-553.

[186] F. P. Ricós, R. Neeft, B. Marín, T. E. Vos, and P. Aho, "Using gui change detection for delta testing," in International Conference on Research Challenges in Information Science. Springer, 2023, pp. 509-517.


<!-- Meanless: JOURNAL OF LATEX CLASS FILES, DECEMBER 2024<br>90 -->

[187] Y. Du, F. Wei, and H. Zhang, "Anytool: Self-reflective, hierarchical agents for large-scale api calls," arXiv preprint arXiv:2402.04253, 2024.

[188] P. Lewis, E. Perez, A. Piktus, F. Petroni, V. Karpukhin, N. Goyal, H. Küttler, M. Lewis, W.-t. Yih, T. Rocktäschel et al., "Retrieval-augmented generation for knowledge-intensive nlp tasks," Advances in Neural Information Processing Systems, vol. 33, pp. 9459-9474, 2020.

[189] Y. Gao, Y. Xiong, X. Gao, K. Jia, J. Pan, Y. Bi, Y. Dai, J. Sun, M. Wang, and H. Wang, "Retrieval-augmented generation for large language models: A survey," arXiv preprint arXiv:2312.10997, 2023.

[190] S. Zhang, Z. Zhang, K. Chen, X. Ma, M. Yang, T. Zhao, and M. Zhang, "Dynamic planning for Ilm-based graphical user interface automation," arXiv preprint arXiv:2410.00467, 2024.

[191] J. Cho, J. Kim, D. Bae, J. Choo, Y. Gwon, and Y.-D. Kwon, "Caap: Context-aware action planning prompting to solve computer tasks with front-end ui only," arXiv preprint arXiv:2406.06947, 2024.

[192] G. Dagan, F. Keller, and A. Lascarides, "Dynamic planning with a Ilm," arXiv preprint arXiv:2308.06391, 2023.

[193] T. Khot, H. Trivedi, M. Finlayson, Y. Fu, K. Richardson, P. Clark, and A. Sabharwal, "Decomposed prompting: A modular approach for solving complex tasks," arXiv preprint arXiv:2210.02406, 2022.

[194] Y. Chen, A. Pesaranghader, T. Sadhu, and D. H. Yi, "Can we rely on Ilm agents to draft long-horizon plans? let's take travelplanner as an example," arXiv preprint arXiv:2408.06318, 2024.

[195] A. Ramesh, M. Pavlov, G. Goh, S. Gray, C. Voss, A. Radford, M. Chen, and I. Sutskever, "Zero-shot text-to-image generation," in International conference on machine learning. Pmlr, 2021, pp. 8821-8831.

[196] X. Gu, H. Zhang, D. Zhang, and S. Kim, "Deep api learning," in Proceedings of the 2016 24th ACM SIGSOFT international symposium on foundations of software engineering, 2016, pp. 631-642.

[197] T. Masterman, S. Besen, M. Sawtell, and A. Chao, "The landscape of emerging ai agent architectures for reasoning, planning, and tool calling: A survey," arXiv preprint arXiv:2404.11584, 2024.

[198] J. Lu, Z. Zhang, F. Yang, J. Zhang, L. Wang, C. Du, Q. Lin, S. Rajmohan, D. Zhang, and Q. Zhang, "Turn every application into an agent: Towards efficient human-agent-computer interaction with api-first Ilm-based agents," arXiv preprint arXiv:2409.17140, 2024.

[199] Y. Song, F. Xu, S. Zhou, and G. Neubig, "Beyond browsing: Api-based web agents," arXiv preprint arXiv:2410.16464, 2024.

[200] S. Lee, J. Choi, J. Lee, M. H. Wasi, H. Choi, S. Y. Ko, S. Oh, and I. Shin, "Explore, select, derive, and recall: Augmenting IIm with human-like memory for mobile task automation," arXiv preprint arXiv:2312.03003, 2023.

[201] J. Lu, S. An, M. Lin, G. Pergola, Y. He, D. Yin, X. Sun, and Y. Wu, "Memochat: Tuning Ilms to use memos for consistent long-range open-domain conversation," arXiv preprint arXiv:2308.08239, 2023.

[202] W. Wang, L. Dong, H. Cheng, X. Liu, X. Yan, J. Gao, and F. Wei, "Augmenting language models with long-term memory," Advances in Neural Information Processing Systems, vol. 36, 2024.

[203] J. Tack, J. Kim, E. Mitchell, J. Shin, Y. W. Teh, and J. R. Schwarz, "Online adaptation of language models with a memory of amortized contexts," arXiv preprint arXiv:2403.04317, 2024.

[204] X. Zhu, Y. Chen, H. Tian, C. Tao, W. Su, C. Yang, G. Huang, B. Li, L. Lu, X. Wang et al., "Ghost in the minecraft: Generally capable agents for open-world environments via large language models with text-based knowledge and memory," arXiv preprint arXiv:2305.17144, 2023.

[205] L. Zheng, R. Wang, X. Wang, and B. An, "Synapse: Trajectory-as-exemplar prompting with memory for computer control," 2024. [Online]. Available: https://arxiv.org/abs/2306.07863

[206] J. Li, D. Li, S. Savarese, and S. Hoi, "Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models," in International conference on machine learning. PMLR, 2023, pp. 19730-19742.

[207] D. Reis, J. Kupec, J. Hong, and A. Daoudi, "Real-time flying object detection with yolov8," arXiv preprint arXiv:2305.09972, 2023.

[208] A. Nguyen, "Improved gui grounding via iterative narrowing," 2024. [Online]. Available: https://arxiv.org/abs/2411.13591

[209] Z. Ge, J. Li, X. Pang, M. Gao, K. Pan, W. Lin, H. Fei, W. Zhang, S. Tang, and Y. Zhuang, "Iris: Breaking gui complexity with adaptive focus and self-refining," 2024. [Online]. Available: https://arxiv.org/abs/2412.10342

[210] J. Bai, S. Bai, S. Yang, S. Wang, S. Tan, P. Wang, J. Lin, C. Zhou, and J. Zhou, "Qwen-vl: A frontier large vision-language model with versatile abilities," arXiv preprint arXiv:2308.12966, 2023.

[211] H.-M. Xu, Q. Chen, L. Wang, and L. Liu, "Attention-driven gui grounding: Leveraging pretrained multimodal large language models without fine-tuning," 2024. [Online]. Available: https://arxiv.org/abs/2412.10840

[212] X. Deng, Y. Gu, B. Zheng, S. Chen, S. Stevens, B. Wang, H. Sun, and Y. Su, "Mind2web: Towards a generalist agent for the web," Advances in Neural Information Processing Systems, vol. 36, pp. 28091-28114, 2023.

[213] J. Liu, Y. Song, B. Y. Lin, W. Lam, G. Neubig, Y. Li, and X. Yue, "Visualwebbench: How far have multimodal Ilms evolved in web page understanding and grounding?" 2024. [Online]. Available: https://arxiv.org/abs/2404.05955

[214] Y. Yang, Y. Wang, D. Li, Z. Luo, B. Chen, C. Huang, and J. Li, "Aria-ui: Visual grounding for gui instructions," 2024. [Online]. Available: https://arxiv.org/abs/2412.16256

[215] D. Li, Y. Liu, H. Wu, Y. Wang, Z. Shen, B. Qu, X. Niu, G. Wang, B. Chen, and J. Li, "Aria: An open multimodal native mixture-of-experts model," arXiv preprint arXiv:2410.05993, 2024.

[216] B. Gou, R. Wang, B. Zheng, Y. Xie, C. Chang, Y. Shu, H. Sun, and Y. Su, "Navigating the digital world as humans do: Universal visual grounding for gui agents," 2024. [Online]. Available: https://arxiv.org/abs/2410.05243

[217] H. Liu, C. Li, Q. Wu, and Y. J. Lee, "Visual instruction tuning," Advances in neural information processing systems, vol. 36, 2024.

[218] Y. Fan, H. Zhao, R. Zhang, Y. Shen, X. E. Wang, and G. Wu, "Gui-bee: Align gui action grounding to novel environments via autonomous exploration," 2025. [Online]. Available: https://arxiv.org/abs/2501.13896

[219] W. Chen, J. Cui, J. Hu, Y. Qin, J. Fang, Y. Zhao, C. Wang, J. Liu, G. Chen, Y. Huo, Y. Yao, Y. Lin, Z. Liu, and M. Sun, "Guicourse: From general vision language models to versatile gui agents," 2024. [Online]. Available: https://arxiv.org/abs/2406.11317

[220] J. Liu, T. Ou, Y. Song, Y. Qu, W. Lam, C. Xiong, W. Chen, G. Neubig, and X. Yue, "Harnessing webpage uis for text-rich visual understanding," arXiv preprint arXiv:2410.13824, 2024.

[221] J. Yang and H. Hou, "Rwkv-ui: Ui understanding with enhanced perception and reasoning," arXiv preprint arXiv:2502.03971, 2025.

[222] X. Zhai, B. Mustafa, A. Kolesnikov, and L. Beyer, "Sigmoid loss for language image pre-training," in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 11 975- 11986.

[223] M. Oquab, T. Darcet, T. Moutakanni, H. Vo, M. Szafraniec, V. Khalidov, P. Fernandez, D. Haziza, F. Massa, A. El-Nouby et al., "Dinov2: Learning robust visual features without supervision," arXiv preprint arXiv:2304.07193, 2023.

[224] H. Laurengon, L. Tronchon, and V. Sanh, "Unlocking the conversion of web screenshots into html code with the websight dataset," arXiv preprint arXiv:2403.09029, 2024.

[225] J. Wu, S. Wang, S. Shen, Y.-H. Peng, J. Nichols, and J. P. Bigham, "Webui: A dataset for enhancing visual ui understanding with web semantics," in Proceedings of the 2023 CHI Conference on Human Factors in Computing Systems, 2023, pp. 1-14.

[226] S. Yun, H. Lin, R. Thushara, M. Q. Bhat, Y. Wang, Z. Jiang, M. Deng, J. Wang, T. Tao, J. Li et al., "Web2code: A large-scale webpage-to-code dataset and evaluation framework for multimodal Ilms," arXiv preprint arXiv:2406.20098, 2024.

[227] K. Singh, S. Singh, and M. Khanna, "Trishul: Towards region identification and screen hierarchy understanding for large vlm based gui agents," 2025. [Online]. Available: https://arxiv.org/abs/2502.08226

[228] H. Li, J. Chen, J. Su, Y. Chen, Q. Li, and Z. Zhang, "Autogui: Scaling gui grounding with automatic functionality annotations from Ilms," arXiv preprint arXiv:2502.01977, 2025.

[229] Y.-F. Zhang, Q. Wen, C. Fu, X. Wang, Z. Zhang, L. Wang, and R. Jin, "Beyond IIava-hd: Diving into high-resolution large multimodal models," arXiv preprint arXiv:2406.08487, 2024.

[230] Z. Wu, P. Cheng, Z. Wu, T. Ju, Z. Zhang, and G. Liu, "Smoothing grounding and reasoning for mllm-powered gui agents with query-oriented pivot tasks," arXiv preprint arXiv:2503.00401, 2025.

[231] P. Wang, S. Bai, S. Tan, S. Wang, Z. Fan, J. Bai, K. Chen, X. Liu, J. Wang, W. Ge, Y. Fan, K. Dang, M. Du, X. Ren, R. Men, D. Liu, C. Zhou, J. Zhou, and J. Lin, "Qwen2-vl: Enhancing vision-language model's perception of the world at any resolution," 2024. [Online]. Available: https://arxiv.org/abs/2409.12191


<!-- Meanless: JOURNAL OF LATEX CLASS FILES, DECEMBER 2024<br>91 -->

[232] Z. Wu, Z. Wu, F. Xu, Y. Wang, Q. Sun, C. Jia, K. Cheng, Z. Ding, L. Chen, P. P. Liang et al., "Os-atlas: A foundation action model for generalist gui agents," arXiv preprint arXiv:2410.23218, 2024.

[233] Z. Hui, Y. Li, D. zhao, T. Chen, C. Banbury, and K. Koishida, "Winclick: Gui grounding with multimodal large language models," 2025. [Online]. Available: https://arxiv.org/abs/2503.04730

[234] M. Abdin, J. Aneja, H. Awadalla, A. Awadallah, A. A. Awan, N. Bach, A. Bahree, A. Bakhtiari, J. Bao, H. Behl et al., "Phi-3 technical report: A highly capable language model locally on your phone," arXiv preprint arXiv:2404.14219, 2024.

[235] F. Tang, Y. Shen, H. Zhang, S. Chen, G. Hou, W. Zhang, W. Zhang, K. Song, W. Lu, and Y. Zhuang, "Think twice, click once: Enhancing gui grounding via fast and slow systems," arXiv preprint arXiv:2503.06470, 2025.

[236] Y. Xu, Z. Wang, J. Wang, D. Lu, T. Xie, A. Saha, D. Sahoo, T. Yu, and C. Xiong, "Aguvis: Unified pure vision agents for autonomous gui interaction," 2024. [Online]. Available: https://arxiv.org/abs/2412.04454

[237] L. Zheng, Z. Huang, Z. Xue, X. Wang, B. An, and S. Yan, "Agentstudio: A toolkit for building general virtual agents," 2024. [Online]. Available: https://arxiv.org/abs/2403.17918

[238] K. Q. Lin, L. Li, D. Gao, Z. Yang, S. Wu, Z. Bai, W. Lei, L. Wang, and M. Z. Shou, "Showui: One vision-language-action model for gui visual agent," 2024. [Online]. Available: https://arxiv.org/abs/2411.17465

[239] X. Liu, X. Zhang, Z. Zhang, and Y. Lu, "Ui-e2i-synth: Advancing gui grounding with large-scale instruction synthesis," arXiv preprint arXiv:2504.11257, 2025.

[240] T. Luo, L. Logeswaran, J. Johnson, and H. Lee, "Visual test-time scaling for gui agent grounding," 2025. [Online]. Available: https://arxiv.org/abs/2505.00684

[241] K. Li, Z. Meng, H. Lin, Z. Luo, Y. Tian, J. Ma, Z. Huang, and T.-S. Chua, "Screenspot-pro: Gui grounding for professional high-resolution computer use," 2025.

[242] Q. Yang, W. Bi, H. Shen, Y. Guo, and Y. Ma, "Pixelweb: The first web gui dataset with pixel-wise labels," 2025. [Online]. Available: https://arxiv.org/abs/2504.16419

[243] X. Zhan, T. Liu, L. Fan, L. Li, S. Chen, X. Luo, and Y. Liu, "Research on third-party libraries in android apps: A taxonomy and systematic literature review," IEEE Transactions on Software Engineering, vol. 48, no. 10, pp. 4181-4213, 2021.

[244] Y. Li, G. Li, L. He, J. Zheng, H. Li, and Z. Guan, "Widget captioning: Generating natural language description for mobile user interface elements," arXiv preprint arXiv:2010.04295, 2020.

[245] B. Wang, G. Li, X. Zhou, Z. Chen, T. Grossman, and Y. Li, "Screen2words: Automatic mobile ui summarization with multimodal learning," in The 34th Annual ACM Symposium on User Interface Software and Technology, 2021, pp. 498-510.

[246] C. Bai, X. Zang, Y. Xu, S. Sunkara, A. Rastogi, J. Chen et al., "Uibert: Learning generic multimodal representations for ui understanding," arXiv preprint arXiv:2107.13731, 2021.

[247] G. Li, H. A. A. K. Hammoud, H. Itani, D. Khizbullin, and B. Ghanem, "Camel: Communicative agents for "mind" exploration of large language model society," in Thirty-seventh Conference on Neural Information Processing Systems, 2023.

[248] W. Chen, Z. You, R. Li, Y. Guan, C. Qian, C. Zhao, C. Yang, R. Xie, Z. Liu, and M. Sun, "Internet of agents: Weaving a web of heterogeneous agents for collaborative intelligence," 2024. [Online]. Available: https://arxiv.org/abs/2407.07061

[249] Z. Song, Y. Li, M. Fang, Z. Chen, Z. Shi, Y. Huang, and L. Chen, "Mmac-copilot: Multi-modal agent collaboration operating system copilot," arXiv preprint arXiv:2404.18074, 2024.

[250] M. Renze and E. Guven, "Self-reflection in Ilm agents: Effects on problem-solving performance," arXiv preprint arXiv:2405.06682, 2024.

[251] J. Pan, Y. Zhang, N. Tomlin, Y. Zhou, S. Levine, and A. Suhr, "Autonomous evaluation and refinement of digital agents," in First Conference on Language Modeling, 2024.

[252] S. Yao, J. Zhao, D. Yu, N. Du, I. Shafran, K. Narasimhan, and Y. Cao, "React: Synergizing reasoning and acting in language models," arXiv preprint arXiv:2210.03629, 2022.

[253] N. Shinn, F. Cassano, A. Gopinath, K. Narasimhan, and S. Yao, "Reflexion: Language agents with verbal reinforcement learning," Advances in Neural Information Processing Systems, vol. 36, 2024.

[254] Z. Tao, T.-E. Lin, X. Chen, H. Li, Y. Wu, Y. Li, Z. Jin, F. Huang, D. Tao, and J. Zhou, "A survey on self-evolution of large language models," arXiv preprint arXiv:2404.14387, 2024.

[255] A. Zhao, D. Huang, Q. Xu, M. Lin, Y.-J. Liu, and G. Huang, "Expel: LIm agents are experiential learners," in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 38, no. 17, 2024, pp. 19632-19642.

[256] Z. Zhu, Y. Xue, X. Chen, D. Zhou, J. Tang, D. Schuurmans, and H. Dai, "Large language models can learn rules," arXiv preprint arXiv:2310.07064, 2023.

[257] Y. Zhang, P. Xiao, L. Wang, C. Zhang, M. Fang, Y. Du, Y. Puzyrev, R. Yao, S. Qin, Q. Lin, M. Pechenizkiy, D. Zhang, S. Rajmohan, and Q. Zhang, "Ruag: Learned-rule-augmented generation for large language models," 2024. [Online]. Available: https://arxiv.org/abs/2411.03349

[258] W. Jiang, Y. Zhuang, C. Song, X. Yang, and C. Zhang, "Appagentx: Evolving gui agents as proficient smartphone users," arXiv preprint arXiv:2503.02268, 2025.

[259] L. P. Kaelbling, M. L. Littman, and A. W. Moore, "Reinforcement learning: A survey," Journal of artificial intelligence research, vol. 4, pp. 237-285, 1996.

[260] Y. Wang, W. Zhong, L. Li, F. Mi, X. Zeng, W. Huang, L. Shang, X. Jiang, and Q. Liu, "Aligning large language models with human: A survey," arXiv preprint arXiv:2307.12966, 2023.

[261] Y. Zhai, H. Bai, Z. Lin, J. Pan, S. Tong, Y. Zhou, A. Suhr, S. Xie, Y. LeCun, Y. Ma et al., "Fine-tuning large vision-language models as decision-making agents via reinforcement learning," arXiv preprint arXiv:2405.10292, 2024.

[262] M. L. Puterman, "Markov decision processes," Handbooks in operations research and management science, vol. 2, pp. 331-434, 1990.

[263] D. Toyama, P. Hamel, A. Gergely, G. Comanici, A. Glaese, Z. Ahmed, T. Jackson, S. Mourad, and D. Precup, "Androidenv: A reinforcement learning platform for android," 2021. [Online]. Available: https://arxiv.org/abs/2105.13231

[264] H. Bai, Y. Zhou, M. Cemri, J. Pan, A. Suhr, S. Levine, and A. Kumar, "Digirl: Training in-the-wild device-control agents with autonomous reinforcement learning," 2024. [Online]. Available: https://arxiv.org/abs/2406.11896

[265] T. Wang, Z. Wu, J. Liu, J. Hao, J. Wang, and K. Shao, "Distrl: An asynchronous distributed reinforcement learning framework for on-device control agents," arXiv preprint arXiv:2410.14803, 2024.

[266] H. Chae, N. Kim, K. T. iunn Ong, M. Gwak, G. Song, J. Kim, S. Kim, D. Lee, and J. Yeo, "Web agents with world models: Learning and leveraging environment dynamics in web navigation," 2024. [Online]. Available: https://arxiv.org/abs/2410.13232

[267] I. Gur, H. Furuta, A. Huang, M. Safdari, Y. Matsuo, D. Eck, and A. Faust, "A real-world webagent with planning, long context understanding, and program synthesis," 2024. [Online]. Available: https://arxiv.org/abs/2307.12856

[268] K. Ma, H. Zhang, H. Wang, X. Pan, W. Yu, and D. Yu, "Laser: LIm agent with state-space exploration for web navigation," 2024. [Online]. Available: https://arxiv.org/abs/2309.08172

[269] H. He, W. Yao, K. Ma, W. Yu, Y. Dai, H. Zhang, Z. Lan, and D. Yu, "Webvoyager: Building an end-to-end web agent with large multimodal models," 2024. [Online]. Available: https://arxiv.org/abs/2401.13919

[270] H. Lai, X. Liu, I. L. long, S. Yao, Y. Chen, P. Shen, H. Yu, H. Zhang, X. Zhang, Y. Dong, and J. Tang, "Autowebglm: Bootstrap and reinforce a large language model-based web navigating agent," 2024. [Online]. Available: https://arxiv.org/abs/2404.03648

[271] T. Xie, F. Zhou, Z. Cheng, P. Shi, L. Weng, Y. Liu, T. J. Hua, J. Zhao, Q. Liu, C. Liu, L. Z. Liu, Y. Xu, H. Su, D. Shin, C. Xiong, and T. Yu, "Openagents: An open platform for language agents in the wild," 2023. [Online]. Available: https://arxiv.org/abs/2310.10634

[272] J. Kil, C. H. Song, B. Zheng, X. Deng, Y. Su, and W.-L. Chao, "Dual-view visual contextualization for web navigation," 2024. [Online]. Available: https://arxiv.org/abs/2402.04476

[273] T. Abuelsaad, D. Akkil, P. Dey, A. Jagmohan, A. Vempaty, and R. Kokku, "Agent-e: From autonomous web navigation to foundational design principles in agentic systems," 2024. [Online]. Available: https://arxiv.org/abs/2407.13032

[274] J. Y. Koh, S. McAleer, D. Fried, and R. Salakhutdinov, "Tree search for language model agents," arXiv preprint arXiv:2407.01476, 2024.

[275] Y. Zhang, Z. Ma, Y. Ma, Z. Han, Y. Wu, and V. Tresp, "Webpilot: A versatile and autonomous multi-agent system for web task execution with strategic exploration," arXiv preprint arXiv:2408.15978, 2024.


<!-- Meanless: JOURNAL OF IATEX CLASS FILES, DECEMBER 2024<br>92 -->

[276] K. Yang, Y. Liu, S. Chaudhary, R. Fakoor, P. Chaudhari, G. Karypis, and H. Rangwala, "Agentoccam: A simple yet strong baseline for Ilm-based web agents," 2024. [Online]. Available: https://arxiv.org/abs/2410.13825

[277] S. Murty, D. Bahdanau, and C. D. Manning, "Nnetscape navigator: Complex demonstrations for web agents without a demonstrator," arXiv preprint arXiv:2410.02907, 2024.

[278] M. Shahbandeh, P. Alian, N. Nashid, and A. Mesbah, "Navigate: Functionality-guided web application navigation," arXiv preprint arXiv:2409.10741, 2024.

[279] I. L. long, X. Liu, Y. Chen, H. Lai, S. Yao, P. Shen, H. Yu, Y. Dong, and J. Tang, "Openwebagent: An open toolkit to enable web agents on large language models," in Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 3: System Demonstrations), 2024, pp. 72-81.

[280] B. Tang and K. G. Shin, "Steward: Natural language web automation," arXiv preprint arXiv:2409.15441, 2024.

[281] P. Putta, E. Mills, N. Garg, S. Motwani, C. Finn, D. Garg, and R. Rafailov, "Agent q: Advanced reasoning and learning for autonomous ai agents," arXiv preprint arXiv:2408.07199, 2024.

[282] Y. Gu, B. Zheng, B. Gou, K. Zhang, C. Chang, S. Srivastava, Y. Xie, P. Qi, H. Sun, and Y. Su, "Is your Ilm secretly a world model of the internet? model-based planning for web agents," arXiv preprint arXiv:2411.06559, 2024.

[283] G. Verma, R. Kaur, N. Srishankar, Z. Zeng, T. Balch, and M. Veloso, "Adaptagent: Adapting multimodal web agents with few-shot learning from human demonstrations," arXiv preprint arXiv:2411.13451, 2024.

[284] J. Kim, D.-K. Kim, L. Logeswaran, S. Sohn, and H. Lee, "Auto-intent: Automated intent discovery and self-exploration for large language model web agents," arXiv preprint arXiv:2410.22552, 2024.

[285] J. Shen, A. Jain, Z. Xiao, I. Amlekar, M. Hadji, A. Podolny, and A. Talwalkar, "Scribeagent: Towards specialized web agents using production-scale workflow data," 2024. [Online]. Available: https://arxiv.org/abs/2411.15004

[286] Y. Zhou, Q. Yang, K. Lin, M. Bai, X. Zhou, Y.-X. Wang, S. Levine, and E. Li, "Proposer-agent-evaluator (pae): Autonomous skill discovery for foundation model internet agents," 2024. [Online]. Available: https://arxiv.org/abs/2412.13194

[287] J. Liu, J. Hao, C. Zhang, and Z. Hu, "Wepo: Web element preference optimization for Ilm-based web navigation," 2024. [Online]. Available: https://arxiv.org/abs/2412.10742

[288] T. Huang, K. Basu, I. Abdelaziz, P. Kapanipathi, J. May, and M. Chen, "R2d2: Remembering, reflecting and dynamic decision making for web agents," arXiv preprint arXiv:2501.12485, 2025.

[289] R. Zhang, M. Qiu, Z. Tan, M. Zhang, V. Lu, J. Peng, K. Xu, L. Z. Agudelo, P. Qian, and T. Chen, "Symbiotic cooperation for web agents: Harnessing complementary strengths of large and small Ilms," arXiv preprint arXiv:2502.07942, 2025.

[290] V. Pahuja, Y. Lu, C. Rosset, B. Gou, A. Mitra, S. Whitehead, Y. Su, and A. Awadallah, "Explorer: Scaling exploration-driven web trajectory synthesis for multimodal web agents," arXiv preprint arXiv:2502.11357, 2025.

[291] M. Wornow, A. Narayan, K. Opsahl-Ong, Q. McIntyre, N. Shah, and C. Re, "Automating the enterprise with foundation models," Proceedings of the VLDB Endowment, vol. 17, no. 11, pp. 2805- 2812, 2024.

[292] D. Zhang, B. Rama, J. Ni, S. He, F. Zhao, K. Chen, A. Chen, and J. Cao, "Litewebagent: The open-source suite for vlm-based web-agent applications," arXiv preprint arXiv:2503.02950, 2025.

[293] P. P. S. Dammu, "Towards ethical and personalized web navigation agents: A framework for user-aligned task execution," in Proceedings of the Eighteenth ACM International Conference on Web Search and Data Mining, 2025, pp. 1074-1076.

[294] L. E. Erdogan, N. Lee, S. Kim, S. Moon, H. Furuta, G. Anu-manchipalli, K. Keutzer, and A. Gholami, "Plan-and-act: Improving planning of agents for long-horizon tasks," arXiv preprint arXiv:2503.09572, 2025.

[295] B. Zheng, M. Y. Fatemi, X. Jin, Z. Z. Wang, A. Gandhi, Y. Song, Y. Gu, J. Srinivasa, G. Liu, G. Neubig et al., "Skillweaver: Web agents can self-improve by discovering and honing skills," arXiv preprint arXiv:2504.07079, 2025.

[296] Z. Z. Wang, A. Gandhi, G. Neubig, and D. Fried, "Inducing programmatic skills for agentic tasks," arXiv preprint arXiv:2504.06821, 2025.

[297] Z. Zhang, T. Fang, K. Ma, W. Yu, H. Zhang, H. Mi, and D. Yu, "Enhancing web agents with explicit rollback mechanisms," arXiv preprint arXiv:2504.11788, 2025.

[298] J. Zhang, J. Wu, Y. Teng, M. Liao, N. Xu, X. Xiao, Z. Wei, and D. Tang, "Android in the zoo: Chain-of-action-thought for gui agents," 2024. [Online]. Available: https://arxiv.org/abs/2403.02713

[299] Y. Song, Y. Bian, Y. Tang, G. Ma, and Z. Cai, "Visiontasker: Mobile task automation using vision based ui understanding and Ilm task planning," in Proceedings of the 37th Annual ACM Symposium on User Interface Software and Technology, ser. UIST '24. ACM, Oct. 2024, p. 1-17. [Online]. Available: http://dx.doi.org/10.1145/3654777.3676386

[300] H. Wen, H. Wang, J. Liu, and Y. Li, "Droidbot-gpt: Gpt-powered ui automation for android," 2024. [Online]. Available: https://arxiv.org/abs/2304.07061

[301] X. Ma, Z. Zhang, and H. Zhao, "Coco-agent: A comprehensive cognitive mllm agent for smartphone gui automation," 2024. [Online]. Available: https://arxiv.org/abs/2402.11941

[302] Z. Zhang and A. Zhang, "You only look at screens: Multimodal chain-of-action agents," 2024. [Online]. Available: https://arxiv.org/abs/2309.11436

[303] A. Yan, Z. Yang, W. Zhu, K. Lin, L. Li, J. Wang, J. Yang, Y. Zhong, J. McAuley, J. Gao, Z. Liu, and L. Wang, "Gpt-4v in wonderland: Large multimodal models for zero-shot smartphone gui navigation," 2023. [Online]. Available: https://arxiv.org/abs/2311.07562

[304] Y. Li, C. Zhang, W. Yang, B. Fu, P. Cheng, X. Chen, L. Chen, and Y. Wei, "Appagent v2: Advanced agent for flexible mobile interactions," 2024. [Online]. Available: https://arxiv.org/abs/2408.11824

[305] H. Wen, S. Tian, B. Pavlov, W. Du, Y. Li, G. Chang, S. Zhao, J. Liu, Y. Liu, Y.-Q. Zhang, and Y. Li, "Autodroid-v2: Boosting slm-based gui agents via code generation," 2024. [Online]. Available: https://arxiv.org/abs/2412.18116

[306] J. Wang, H. Xu, H. Jia, X. Zhang, M. Yan, W. Shen, J. Zhang, F. Huang, and J. Sang, "Mobile-agent-v2: Mobile device operation assistant with effective navigation via multi-agent collaboration," 2024. [Online]. Available: https://arxiv.org/abs/2406.01014

[307] J. Zhang, C. Zhao, Y. Zhao, Z. Yu, M. He, and J. Fan, "Mobileexperts: A dynamic tool-enabled agent team in mobile devices," 2024. [Online]. Available: https://arxiv.org/abs/2407.03913

[308] F. Christianos, G. Papoudakis, T. Coste, J. Hao, J. Wang, and K. Shao, "Lightweight neural app control," 2024. [Online]. Available: https://arxiv.org/abs/2410.17883

[309] Z. Zhu, H. Tang, Y. Li, K. Lan, Y. Jiang, H. Zhou, Y. Wang, S. Zhang, L. Sun, L. Chen et al., "Moba: A two-level agent system for efficient mobile task automation," arXiv preprint arXiv:2410.13757, 2024.

[310] S. Lee, J. Choi, J. Lee, M. H. Wasi, H. Choi, S. Ko, S. Oh, and I. Shin, "Mobilegpt: Augmenting llm with human-like app memory for mobile task automation," in Proceedings of the 30th Annual International Conference on Mobile Computing and Networking, 2024, pp. 1119-1133.

[311] Z. Wang, H. Xu, J. Wang, X. Zhang, M. Yan, J. Zhang, F. Huang, and H. Ji, "Mobile-agent-e: Self-evolving mobile assistant for complex tasks," 2025. [Online]. Available: https: //arxiv.org/abs/2501.11733

[312] J. Hoscilowicz, B. Maj, B. Kozakiewicz, O. Tymoshchuk, and A. Janicki, "Clickagent: Enhancing ui location capabilities of autonomous agents," arXiv preprint arXiv:2410.11872, 2024.

[313] Q. Wu, W. Liu, J. Luan, and B. Wang, "Reachagent: Enhancing mobile agent via page reaching and operation," arXiv preprint arXiv:2502.02955, 2025.

[314] W. Wang, Z. Yu, W. Liu, R. Ye, T. Jin, S. Chen, and Y. Wang, "Fedmobileagent: Training mobile agents using decentralized self-sourced data from diverse users," arXiv preprint arXiv:2502.02982, 2025.

[315] T. Huang, C. Yu, W. Shi, Z. Peng, D. Yang, W. Sun, and Y. Shi, "Prompt2task: Automating ui tasks on smartphones from textual prompts," ACM Transactions on Computer-Human Interaction.

[316] J. Wang, H. Xu, X. Zhang, M. Yan, J. Zhang, F. Huang, and J. Sang, "Mobile-agent-v: Learning mobile device operation through video-guided multi-agent collaboration," arXiv preprint arXiv:2502.17110, 2025.


<!-- Meanless: JOURNAL OF IATEX CLASS FILES, DECEMBER 2024<br>93 -->

[317] Y. Liu, H. Sun, W. Liu, J. Luan, B. Du, and R. Yan, "Mobilesteward: Integrating multiple app-oriented agents with self-evolution to automate cross-app instructions," arXiv preprint arXiv:2502.16796, 2025.

[318] Y. Zhou, S. Wang, S. Dai, Q. Jia, Z. Du, Z. Dong, and J. Xu, "Chop: Mobile operating assistant with constrained high-frequency optimized subtask planning," arXiv preprint arXiv:2503.03743, 2025.

[319] P. Cheng, Z. Wu, Z. Wu, A. Zhang, Z. Zhang, and G. Liu, "Os-kairos: Adaptive interaction for mllm-powered gui agents," arXiv preprint arXiv:2503.16465, 2025.

[320] G. Dai, S. Jiang, T. Cao, Y. Li, Y. Yang, R. Tan, M. Li, and L. Qiu, "Advancing mobile gui agents: A verifier-driven approach to practical deployment," arXiv preprint arXiv:2503.15937, 2025.

[321] G. Liu, P. Zhao, L. Liu, Z. Chen, Y. Chai, S. Ren, H. Wang, S. He, and W. Meng, "Learnact: Few-shot mobile gui agent with a unified demonstration benchmark," 2025. [Online]. Available: https://arxiv.org/abs/2504.13805

[322] H. Lai, J. Gao, X. Liu, Y. Xu, S. Zhang, Y. Dong, and J. Tang, "Androidgen: Building an android language agent under data scarcity," arXiv preprint arXiv:2504.19298, 2025.

[323] B. Wang, G. Li, and Y. Li, "Enabling conversational interaction with mobile ui using large language models," in Proceedings of the 2023 CHI Conference on Human Factors in Computing Systems, 2023, pp. 1-17.

[324] N. Kahlon, G. Rom, A. Efros, F. Galgani, O. Berkovitch, S. Caduri, W. E. Bishop, O. Riva, and I. Dagan, "Agent-initiated interaction in phone ui automation," arXiv preprint arXiv:2503.19537, 2025.

[325] W. E. Bishop, A. Li, C. Rawles, and O. Riva, "Latent state estimation helps ui agents to reason," arXiv preprint arXiv:2405.11120, 2024.

[326] S. Agashe, J. Han, S. Gan, J. Yang, A. Li, and X. E. Wang, "Agent s: An open agentic framework that uses computers like a human," 2024. [Online]. Available: https://arxiv.org/abs/2410.08164

[327] Q. Wu, D. Gao, K. Q. Lin, Z. Wu, X. Guo, P. Li, W. Zhang, H. Wang, and M. Z. Shou, "Gui action narrator: Where and when did that action take place?" 2024. [Online]. Available: https://arxiv.org/abs/2406.13719

[328] T. Li, G. Li, Z. Deng, B. Wang, and Y. Li, "A zero-shot language agent for computer control with structured reflection," arXiv preprint arXiv:2310.08740, 2023.

[329] Y. He, J. Jin, S. Xia, J. Su, R. Fan, H. Zou, X. Hu, and P. Liu, "Pc agent: While you sleep, ai works - a cognitive journey into digital world," 2024. [Online]. Available: https://arxiv.org/abs/2412.17589

[330] H. Liu, X. Zhang, H. Xu, Y. Wanyan, J. Wang, M. Yan, J. Zhang, C. Yuan, C. Xu, W. Hu et al., "Pc-agent: A hierarchical multi-agent collaboration framework for complex task automation on pc," arXiv preprint arXiv:2502.14282, 2025.

[331] P. Aggarwal and S. Welleck, "Programming with pixels: Computer-use meets software engineering," arXiv preprint arXiv:2502.18525, 2025.

[332] D. Zhao, L. Ma, S. Wang, M. Wang, and Z. Lv, "Cola: A scalable multi-agent framework for windows ui task automation," arXiv preprint arXiv:2503.09263, 2025.

[333] F. Lu, Z. Zhong, Z. Wei, S. Liu, C.-W. Fu, and J. Jia, "Steve: Astep verification pipeline for computer-use agent training," arXiv preprint arXiv:2503.12532, 2025.

[334] C. Zhang, H. Huang, C. Ni, J. Mu, S. Qin, S. He, L. Wang, F. Yang, P. Zhao, C. Du et al., "Ufo2: The desktop agentos," arXiv preprint arXiv:2504.14603, 2025.

[335] Y. Yin, Y. Mei, C. Yu, T. J.-J. Li, A. K. Jadoon, S. Cheng, W. Shi, M. Chen, and Y. Shi, "From operation to cognition: Automatic modeling cognitive dependencies from user demonstrations for gui task automation," in Proceedings of the 2025 CHI Conference on Human Factors in Computing Systems, 2025, pp. 1-24.

[336] X. Liu, B. Qin, D. Liang, G. Dong, H. Lai, H. Zhang, H. Zhao, I. L. long, J. Sun, J. Wang et al., "Autoglm: Autonomous foundation agents for guis," arXiv preprint arXiv:2411.00820, 2024.

[337] P. Pawlowski, K. Zawistowski, W. Lapacz, M. Skorupa, A. Wiacek, S. Postansque, and J. Hoscilowicz, "Tinyclick: Single-turn agent for empowering gui automation," arXiv preprint arXiv:2410.11871, 2024.

[338] H. Su, R. Sun, J. Yoon, P. Yin, T. Yu, and S. O. Arik, "Learn-by-interact: A data-centric framework for self-adaptive agents in realistic environments," 2025. [Online]. Available: https://arxiv.org/abs/2501.10893

[339] Z. He, Z. Liu, P. Li, M. Fung, M. Yan, J. Zhang, F. Huang, and Y. Liu, "Enhancing language multi-agent learning with multi-agent credit re-assignment for interactive environment generalization," arXiv preprint arXiv:2502.14496, 2025.

[340] X. Wang and B. Liu, "Oscar: Operating system control via state-aware reasoning and re-planning," arXiv preprint arXiv:2410.18963, 2024.

[341] C. Jia, M. Luo, Z. Dang, Q. Sun, F. Xu, J. Hu, T. Xie, and Z. Wu, "Agentstore: Scalable integration of heterogeneous agents as specialized generalist computer assistant," arXiv preprint arXiv:2410.18603, 2024.

[342] Y. Wang, H. Zhang, J. Tian, and Y. Tang, "Ponder & press: Advancing visual gui agent towards general computer control," 2024. [Online]. Available: https://arxiv.org/abs/2412.01268

[343] Y. Liu, P. Li, Z. Wei, C. Xie, X. Hu, X. Xu, S. Zhang, X. Han, H. Yang, and F. Wu, "Infiguiagent: A multimodal generalist gui agent with native reasoning and reflection," 2025. [Online]. Available: https://arxiv.org/abs/2501.04575

[344] S. Agashe, K. Wong, V. Tu, J. Yang, A. Li, and X. E. Wang, "Agent s2: A compositional generalist-specialist framework for computer use agents," arXiv preprint arXiv:2504.00906, 2025.

[345] Z. Hu, S. Xiong, Y. Zhang, S.-K. Ng, A. T. Luu, B. An, S. Yan, and B. Hooi, "Guiding vlm agents with process rewards at inference time for gui navigation," 2025. [Online]. Available: https://arxiv.org/abs/2504.16073

[346] J. Huang, Z. Zeng, W. Han, Y. Zhong, L. Zheng, S. Fu, J. Chen, and L. Ma, "Scaletrack: Scaling and back-tracking automated gui agents," 2025. [Online]. Available: https://arxiv.org/abs/2505.00416

[347] Q. Chen, D. Pitawela, C. Zhao, G. Zhou, H.-T. Chen, and Q. Wu, "Webvln: Vision-and-language navigation on websites," Proceedings of the AAAI Conference on Artificial Intelligence, vol. 38, no. 2, pp. 1165-1173, Mar. 2024. [Online]. Available: https://ojs.aaai.org/index.php/AAAI/article/view/27878

[348] X. H. Lu, Z. Kasner, and S. Reddy, "Weblinx: Real-world website navigation with multi-turn dialogue," in International Conference on Machine Learning. PMLR, 2024, pp. 33 007-33 056.

[349] Y. Pan, D. Kong, S. Zhou, C. Cui, Y. Leng, B. Jiang, H. Liu, Y. Shang, S. Zhou, T. Wu, and Z. Wu, "Webcanvas: Benchmarking web agents in online environments," 2024. [Online]. Available: https://arxiv.org/abs/2406.12373

[350] Y. Xu, D. Lu, Z. Shen, J. Wang, Z. Wang, Y. Mao, C. Xiong, and T. Yu, "Agenttrek: Agent trajectory synthesis via guiding replay with web tutorials," 2024. [Online]. Available: https://arxiv.org/abs/2412.09605

[351] B. Trabucco, G. Sigurdsson, R. Piramuthu, and R. Salakhutdi-nov, "Towards internet-scale training for agents," arXiv preprint arXiv:2502.06776, 2025.

[352] K. You, H. Zhang, E. Schoop, F. Weers, A. Swearngin, J. Nichols, Y. Yang, and Z. Gan, "Ferret-ui: Grounded mobile ui understanding with multimodal Ilms," in European Conference on Computer Vision. Springer, 2025, pp. 240-255.

[353] Q. Wu, W. Xu, W. Liu, T. Tan, J. Liu, A. Li, J. Luan, B. Wang, and S. Shang, "Mobilevlm: A vision-language model for better intra- and inter-ui understanding," 2024. [Online]. Available: https://arxiv.org/abs/2409.14818

[354] Z. Meng, Y. Dai, Z. Gong, S. Guo, M. Tang, and T. Wei, "Vga: Vision gui assistant - minimizing hallucinations through image-centric fine-tuning," 2024. [Online]. Available: https://arxiv.org/abs/2406.14056

[355] B. Deka, Z. Huang, C. Franzen, J. Hibschman, D. Afergan, Y. Li, J. Nichols, and R. Kumar, "Rico: A mobile app dataset for building data-driven design applications," in Proceedings of the 30th Annual ACM Symposium on User Interface Software and Technology, ser. UIST '17. New York, NY, USA: Association for Computing Machinery, 2017, p. 845-854. [Online]. Available: https://doi.org/10.1145/3126594.3126651

[356] A. Burns, D. Arsan, S. Agrawal, R. Kumar, K. Saenko, and B. A. Plummer, "A dataset for interactive vision-language navigation with unknown command feasibility," 2022. [Online]. Available: https://arxiv.org/abs/2202.02312

[357] L. Sun, X. Chen, L. Chen, T. Dai, Z. Zhu, and K. Yu, "Meta-gui: Towards multi-modal conversational agents on mobile gui," 2022. [Online]. Available: https://arxiv.org/abs/2205.11029

[358] C. Rawles, A. Li, D. Rodriguez, O. Riva, and T. Lillicrap, "An-droidinthewild: A large-scale dataset for android device control," Advances in Neural Information Processing Systems, vol. 36, pp. 59708-59728, 2023.

[359] Q. Lu, W. Shao, Z. Liu, F. Meng, B. Li, B. Chen, S. Huang, K. Zhang, Y. Qiao, and P. Luo, "Gui odyssey: A comprehensive dataset for cross-app gui navigation on mobile devices," 2024. [Online]. Available: https://arxiv.org/abs/2406.08451


<!-- Meanless: JOURNAL OF LATEX CLASS FILES, DECEMBER 2024<br>94 -->

[360] Y. Chai, S. Huang, Y. Niu, H. Xiao, L. Liu, D. Zhang, P. Gao, S. Ren, and H. Li, "Amex: Android multi-annotation expo dataset for mobile gui agents," 2024. [Online]. Available: https://arxiv.org/abs/2407.17490

[361] W. Chen, Z. Li, Z. Guo, and Y. Shen, "Octo-planner: On-device language model for planner-action agents," 2024. [Online]. Available: https://arxiv.org/abs/2406.18082

[362] K. Wang, T. Xia, Z. Gu, Y. Zhao, S. Shen, C. Meng, W. Wang, and K. Xu, "E-ant: A large-scale dataset for efficient automatic gui navigation," 2024. [Online]. Available: https://arxiv.org/abs/2406.14250

[363] Y. Xu, X. Liu, X. Sun, S. Cheng, H. Yu, H. Lai, S. Zhang, D. Zhang, J. Tang, and Y. Dong, "Androidlab: Training and systematic benchmarking of android autonomous agents," 2024. [Online]. Available: https://arxiv.org/abs/2410.24024

[364] L. Gao, L. Zhang, S. Wang, S. Wang, Y. Li, and M. Xu, "Mobileviews: A large-scale mobile gui dataset," arXiv preprint arXiv:2409.14337, 2024.

[365] Y. Sun, S. Zhao, T. Yu, H. Wen, S. Va, M. Xu, Y. Li, and C. Zhang, "Gui-xplore: Empowering generalizable gui agents with one exploration," arXiv preprint arXiv:2503.17709, 2025.

[366] R. Niu, J. Li, S. Wang, Y. Fu, X. Hu, X. Leng, H. Kong, Y. Chang, and Q. Wang, "Screenagent: A vision language model-driven computer control agent," 2024. [Online]. Available: https://arxiv.org/abs/2402.07945

[367] L. Wang, F. Yang, C. Zhang, J. Lu, J. Qian, S. He, P. Zhao, B. Qiao, R. Huang, S. Qin, Q. Su, J. Ye, Y. Zhang, J.-G. Lou, Q. Lin, S. Rajmohan, D. Zhang, and Q. Zhang, "Large action models: From inception to implementation," 2024. [Online]. Available: https://arxiv.org/abs/2412.10047

[368] Y. Xu, L. Yang, H. Chen, H. Wang, Z. Chen, and Y. Tang, "Deskvision: Large scale desktop region captioning for advanced gui agents," arXiv preprint arXiv:2503.11170, 2025.

[369] Q. Sun, K. Cheng, Z. Ding, C. Jin, Y. Wang, F. Xu, Z. Wu, C. Jia, L. Chen, Z. Liu et al., "Os-genesis: Automating gui agent trajectory construction via reverse task synthesis," arXiv preprint arXiv:2412.19723, 2024.

[370] R. Chawla, A. Jha, M. Kumar, M. NS, and I. Bhola, "Guide: Graphical user interface data for execution," arXiv preprint arXiv:2404.16048, 2024.

[371] D. Chen, Y. Huang, S. Wu, J. Tang, L. Chen, Y. Bai, Z. He, C. Wang, H. Zhou, Y. Li, T. Zhou, Y. Yu, C. Gao, Q. Zhang, Y. Gui, Z. Li, Y. Wan, P. Zhou, J. Gao, and L. Sun, "Gui-world: A dataset for gui-oriented multimodal Ilm-based agents," 2024. [Online]. Available: https://arxiv.org/abs/2406.10819

[372] J. Zhang, T. Lan, M. Zhu, Z. Liu, T. Hoang, S. Kokane, W. Yao, J. Tan, A. Prabhakar, H. Chen et al., "xlam: A family of large action models to empower ai agent systems," arXiv preprint arXiv:2409.03215, 2024.

[373] H. Shen, C. Liu, G. Li, X. Wang, Y. Zhou, C. Ma, and X. Ji, "Falcon-ui: Understanding gui before following user instructions," arXiv preprint arXiv:2412.09362, 2024.

[374] X. Liu, T. Zhang, Y. Gu, I. L. long, Y. Xu, X. Song, S. Zhang, H. Lai, X. Liu, H. Zhao, J. Sun, X. Yang, Y. Yang, Z. Qi, S. Yao, X. Sun, S. Cheng, Q. Zheng, H. Yu, H. Zhang, W. Hong, M. Ding, L. Pan, X. Gu, A. Zeng, Z. Du, C. H. Song, Y. Su, Y. Dong, and J. Tang, "Visualagentbench: Towards large multimodal models as visual foundation agents," 2024. [Online]. Available: https://arxiv.org/abs/2408.06327

[375] G. Baechler, S. Sunkara, M. Wang, F. Zubach, H. Mansoor, V. Etter, V. Cărbune, J. Lin, J. Chen, and A. Sharma, "Screenai: A vision-language model for ui and infographics understanding," 2024. [Online]. Available: https://arxiv.org/abs/2402.04615

[376] I. Chaimalas, A. VyLAniauskas, and G. Brostow, "Explorer: Robust collection of interactable gui elements," arXiv preprint arXiv:2504.09352, 2025.

[377] Z. Cheng, Z. Huang, J. Pan, Z. Hou, and M. Zhan, "Navi-plus: Managing ambiguous gui navigation tasks with follow-up," arXiv preprint arXiv:2503.24180, 2025.

[378] OpenAI, "Gpt-4v(ision) system card," OpenAI, Tech. Rep., September 2023. [Online]. Available: https://cdn.openai.com/ papers/GPTV_System_Card.pdf

[379] Z. Chen, J. Wu, W. Wang, W. Su, G. Chen, S. Xing, M. Zhong, Q. Zhang, X. Zhu, L. Lu et al., "Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks," in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 24 185-24 198.

[380] Z. Chen, W. Wang, H. Tian, S. Ye, Z. Gao, E. Cui, W. Tong, K. Hu, J. Luo, Z. Ma et al., "How far are we to gpt-4v? closing the gap to commercial multimodal models with open-source suites," arXiv preprint arXiv:2404.16821, 2024.

[381] W. Wang, Q. Lv, W. Yu, W. Hong, J. Qi, Y. Wang, J. Ji, Z. Yang, L. Zhao, X. Song, J. Xu, B. Xu, J. Li, Y. Dong, M. Ding, and J. Tang, "Cogvlm: Visual expert for pretrained language models," 2024. [Online]. Available: https://arxiv.org/abs/2311.03079

[382] H. You, H. Zhang, Z. Gan, X. Du, B. Zhang, Z. Wang, L. Cao, S.-F. Chang, and Y. Yang, "Ferret: Refer and ground anything anywhere at any granularity," arXiv preprint arXiv:2310.07704, 2023.

[383] H. Liu, C. Li, Y. Li, and Y. J. Lee, "Improved baselines with visual instruction tuning," in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 26 296- 26306.

[384] Z. Huang, Z. Cheng, J. Pan, Z. Hou, and M. Zhan, "Spirit-sight agent: Advanced gui agent with one look," arXiv preprint arXiv:2503.03196, 2025.

[385] M. Fereidouni and A. B. Siddique, "Search beyond queries: Training smaller language models for web interactions via reinforcement learning," 2024. [Online]. Available: https: //arxiv.org/abs/2404.10887

[386] L.-A. Thil, M. Popa, and G. Spanakis, "Navigating webai: Training agents to complete web tasks with large language models and reinforcement learning," in Proceedings of the 39th ACM/SIGAPP Symposium on Applied Computing, ser. SAC '24, vol. 30. ACM, Apr. 2024, p. 866-874. [Online]. Available: http://dx.doi.org/10.1145/3605098.3635903

[387] H. He, W. Yao, K. Ma, W. Yu, H. Zhang, T. Fang, Z. Lan, and D. Yu, "Openwebvoyager: Building multimodal web agents via iterative real-world exploration, feedback and optimization," 2024. [Online]. Available: https://arxiv.org/abs/2410.19609

[388] Z. Qi, X. Liu, I. L. long, H. Lai, X. Sun, X. Yang, J. Sun, Y. Yang, S. Yao, T. Zhang, W. Xu, J. Tang, and Y. Dong, "Webrl: Training Ilm web agents via self-evolving online curriculum reinforcement learning," 2024. [Online]. Available: https://arxiv.org/abs/2411.02337

[389] J. Zhang, Z. Ding, C. Ma, Z. Chen, Q. Sun, Z. Lan, and J. He, "Breaking the data barrier-building gui agents through task generalization," arXiv preprint arXiv:2504.10127, 2025.

[390] Y. Qian, Y. Lu, A. Hauptmann, and O. Riva, "Visual grounding for user interfaces," in Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 6: Industry Track), Y. Yang, A. Davani, A. Sil, and A. Kumar, Eds. Mexico City, Mexico: Association for Computational Linguistics, Jun. 2024, pp. 97-107. [Online]. Available: https://aclanthology.org/2024.naacl-industry.9

[391] W. Chen, Z. Li, and M. Ma, "Octopus: On-device language model for function calling of software apis," 2024. [Online]. Available: https://arxiv.org/abs/2404.01549

[392] W. Chen and Z. Li, "Octopus v2: On-device language model for super agent," 2024. [Online]. Available: https: //arxiv.org/abs/2404.01744

[393] —, "Octopus v3: Technical report for on-device sub-billion multimodal ai agent," 2024. [Online]. Available: https: //arxiv.org/abs/2404.11459

[394] —, "Octopus v4: Graph of language models," 2024. [Online]. Available: https://arxiv.org/abs/2404.19296

[395] W. Li, F.-L. Hsu, W. Bishop, F. Campbell-Ajala, M. Lin, and O. Riva, "Uinav: A practical approach to train on-device automation agents," in Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 6: Industry Track), 2024, pp. 36- 51.

[396] Q. Wu, J. Liu, J. Hao, J. Wang, and K. Shao, "Vsc-rl: Advancing autonomous vision-language agents with variational subgoal-conditioned reinforcement learning," arXiv preprint arXiv:2502.07949, 2025.

[397] G. Papoudakis, T. Coste, Z. Wu, J. Hao, J. Wang, and K. Shao, "Appvlm: A lightweight vision language model for online app control," arXiv preprint arXiv:2502.06395, 2025.

[398] H. Bai, Y. Zhou, L. E. Li, S. Levine, and A. Kumar, "Digi-q: Learning q-value functions for training device-control agents," arXiv preprint arXiv:2502.15760, 2025.


<!-- Meanless: JOURNAL OF LATEX CLASS FILES, DECEMBER 2024<br>95 -->

[399] J. Zheng, L. Wang, F. Yang, C. Zhang, L. Mei, W. Yin, Q. Lin, D. Zhang, S. Rajmohan, and Q. Zhang, "Vem: Environment-free exploration for training gui agent with value environment model," arXiv preprint arXiv:2502.18906, 2025.

[400] Z. Wang, W. Chen, L. Yang, S. Zhou, S. Zhao, H. Zhan, J. Jin, L. Li, Z. Shao, and J. Bu, "Mp-gui: Modality perception with mllms for gui understanding," arXiv preprint arXiv:2503.14021, 2025.

[401] Z. Lu, Y. Chai, Y. Guo, X. Yin, L. Liu, H. Wang, G. Xiong, and H. Li, "Ui-r1: Enhancing action prediction of gui agents by reinforcement learning," arXiv preprint arXiv:2503.21620, 2025.

[402] D. Luo, B. Tang, K. Li, G. Papoudakis, J. Song, S. Gong, J. Hao, J. Wang, and K. Shao, "Vimo: A generative visual gui world model for app agent," 2025. [Online]. Available: https://arxiv.org/abs/2504.13936

[403] J. Yang, Y. Dong, S. Liu, B. Li, Z. Wang, H. Tan, C. Jiang, J. Kang, Y. Zhang, K. Zhou et al., "Octopus: Embodied vision-language programmer from environmental feedback," in European Conference on Computer Vision. Springer, 2025, pp. 20-38.

[404] Y. Jin, S. Petrangeli, Y. Shen, and G. Wu, "Screenllm: Stateful screen schema for efficient action understanding and prediction," 2025.

[405] Z. Zhang, W. Xie, X. Zhang, and Y. Lu, "Reinforced ui instruction grounding: Towards a generic ui task automation api," 2023. [Online]. Available: https://arxiv.org/abs/2310.04716

[406] Z. Li, K. You, H. Zhang, D. Feng, H. Agrawal, X. Li, M. P. S. Moorthy, J. Nichols, Y. Yang, and Z. Gan, "Ferret-ui 2: Mastering universal user interface understanding across platforms," arXiv preprint arXiv:2410.18967, 2024.

[407] Y. Qin, Y. Ye, J. Fang, H. Wang, S. Liang, S. Tian, J. Zhang, J. Li, Y. Li, S. Huang, W. Zhong, K. Li, J. Yang, Y. Miao, W. Lin, L. Liu, X. Jiang, Q. Ma, J. Li, X. Xiao, K. Cai, C. Li, Y. Zheng, C. Jin, C. Li, X. Zhou, M. Wang, H. Chen, Z. Li, H. Yang, H. Liu, F. Lin, T. Peng, X. Liu, and G. Shi, "Ui-tars: Pioneering automated gui interaction with native agents," 2025. [Online]. Available: https://arxiv.org/abs/2501.12326

[408] A. Rahman, R. Chawla, M. Kumar, A. Datta, A. Jha, M. NS, and I. Bhola, "V-zen: Efficient gui understanding and precise grounding with a novel multimodal Ilm," arXiv preprint arXiv:2405.15341, 2024.

[409] J. Yang, R. Tan, Q. Wu, R. Zheng, B. Peng, Y. Liang, Y. Gu, M. Cai, S. Ye, J. Jang et al., "Magma: A foundation model for multimodal ai agents," arXiv preprint arXiv:2502.13130, 2025.

[410] X. Xia and R. Luo, "Gui-r1: A generalist r1-style vision-language action model for gui agents," arXiv preprint arXiv:2504.10458, 2025.

[411] Y. Liu, P. Li, C. Xie, X. Hu, X. Han, S. Zhang, H. Yang, and F. Wu, "Infigui-r1: Advancing multimodal gui agents from reactive actors to deliberative reasoners," 2025. [Online]. Available: https://arxiv.org/abs/2504.14239

[412] S. Zhou, F. F. Xu, H. Zhu, X. Zhou, R. Lo, A. Sridhar, X. Cheng, T. Ou, Y. Bisk, D. Fried et al., "Webarena: A realistic web environment for building autonomous agents," in The Twelfth International Conference on Learning Representations.

[413] J. Y. Koh, R. Lo, L. Jang, V. Duvvur, M. C. Lim, P.- Y. Huang, G. Neubig, S. Zhou, R. Salakhutdinov, and D. Fried, "Visualwebarena: Evaluating multimodal agents on realistic visual web tasks," 2024. [Online]. Available: https://arxiv.org/abs/2401.13649

[414] Y. Deng, X. Zhang, W. Zhang, Y. Yuan, S.-K. Ng, and T.-S. Chua, "On the multi-turn instruction following for conversational web agents," 2024. [Online]. Available: https://arxiv.org/abs/2402.15057

[415] Z. Zhang, S. Tian, L. Chen, and Z. Liu, "Mmina: Benchmarking multihop multimodal internet agents," 2024. [Online]. Available: https://arxiv.org/abs/2404.09992

[416] I. Levy, B. Wiesel, S. Marreed, A. Oved, A. Yaeli, and S. Shlo-mov, "St-webagentbench: A benchmark for evaluating safety and trustworthiness in web agents," arXiv preprint arXiv:2410.06703, 2024.

[417] H. Furuta, Y. Matsuo, A. Faust, and I. Gur, "Exposing limitations of language model agents in sequential-task compositions on the web," in ICLR 2024 Workshop on Large Language Model (LLM) Agents, 2024.

[418] K. Xu, Y. Kordi, T. Nayak, A. Asija, Y. Wang, K. Sanders, A. Byerly, J. Zhang, B. Van Durme, and D. Khashabi, "Tur [k] ingbench: A challenge benchmark for web agents," arXiv preprint arXiv:2403.11905, 2024.

[419] T. Xie, D. Zhang, J. Chen, X. Li, S. Zhao, R. Cao, T. J. Hua, Z. Cheng, D. Shin, F. Lei, Y. Liu, Y. Xu, S. Zhou, S. Savarese, C. Xiong, V. Zhong, and T. Yu, "Osworld: Benchmarking multimodal agents for open-ended tasks in real computer environments," 2024. [Online]. Available: https://arxiv.org/abs/2404.07972

[420] A. Drouin, M. Gasse, M. Caccia, I. H. Laradji, M. Del Verme, T. Marty, L. Boisvert, M. Thakkar, Q. Cappart, D. Vazquez et al., "Workarena: How capable are web agents at solving common knowledge work tasks?" arXiv preprint arXiv:2403.07718, 2024.

[421] L. Jang, Y. Li, C. Ding, J. Lin, P. P. Liang, D. Zhao, R. Bonatti, and K. Koishida, "Videowebarena: Evaluating long context multimodal agents with video understanding web tasks," arXiv preprint arXiv:2410.19100, 2024.

[422] X. Ma, Y. Wang, Y. Yao, T. Yuan, A. Zhang, Z. Zhang, and H. Zhao, "Caution for the environment: Multimodal agents are susceptible to environmental distractions," 2024. [Online]. Available: https://arxiv.org/abs/2408.02544

[423] M. Wornow, A. Narayan, B. Viggiano, I. S. Khare, T. Verma, T. Thompson, M. A. F. Hernandez, S. Sundar, C. Trujillo, K. Chawla et al., "Wonderbread: A benchmark for evaluating multimodal foundation models on business process management tasks," in The Thirty-eight Conference on Neural Information Processing Systems Datasets and Benchmarks Track.

[424] B. Zheng, B. Gou, S. Salisbury, Z. Du, H. Sun, and Y. Su, "Webolympus: An open platform for web agents on live websites," in Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, 2024, pp. 187-197.

[425] S. Yao, H. Chen, J. Yang, and K. Narasimhan, "Webshop: Towards scalable real-world web interaction with grounded language agents," 2023. [Online]. Available: https://arxiv.org/abs/2207.01206

[426] D. Chezelles, T. Le Sellier, M. Gasse, A. Lacoste, A. Drouin, M. Caccia, L. Boisvert, M. Thakkar, T. Marty, R. Assouel et al., "The browsergym ecosystem for web agent research," arXiv preprint arXiv:2412.05467, 2024.

[427] J. Wu, W. Yin, Y. Jiang, Z. Wang, Z. Xi, R. Fang, D. Zhou, P. Xie, and F. Huang, "Webwalker: Benchmarking Ilms in web traversal," 2025. [Online]. Available: https://arxiv.org/abs/2501.07572

[428] G. Thomas, A. J. Chan, J. Kang, W. Wu, F. Christianos, F. Greenlee, A. Toulis, and M. Purtorab, "Webgames: Challenging general-purpose web-browsing ai agents," arXiv preprint arXiv:2502.18356, 2025.

[429] A. D. Tur, N. Meade, X. H. Lù, A. Zambrano, A. Patel, E. Durmus, S. Gella, K. Stańczak, and S. Reddy, "Safearena: Evaluating the safety of autonomous web agents," 2025. [Online]. Available: https://arxiv.org/abs/2503.04957

[430] S. Kara, F. Faisal, and S. Nath, "Waber: Web agent benchmarking for efficiency and reliability," in ICLR 2025 Workshop on Foundation Models in the Wild.

[431] T. Xue, W. Qi, T. Shi, C. H. Song, B. Gou, D. Song, H. Sun, and Y. Su, "An illusion of progress? assessing the current state of web agents," arXiv preprint arXiv:2504.01382, 2025.

[432] A. Zharmagambetov, C. Guo, I. Evtimov, M. Pavlova, R. Salakhutdi-nov, and K. Chaudhuri, "Agentdam: Privacy leakage evaluation for autonomous web agents," arXiv preprint arXiv:2503.09780, 2025.

[433] X. H. Lù, A. Kazemnejad, N. Meade, A. Patel, D. Shin, A. Zambrano, K. Stańczak, P. Shaw, C. J. Pal, and S. Reddy, "Agentrewardbench: Evaluating automatic evaluations of web agent trajectories," arXiv preprint arXiv:2504.08942, 2025.

[434] S. Ye, H. Shi, D. Shih, H. Yun, T. Roosta, and T. Shu, "Real-webassist: A benchmark for long-horizon web assistance with real-world users," arXiv preprint arXiv:2504.10445, 2025.

[435] D. Garg, S. VanWeelden, D. Caples, A. Draguns, N. Ravi, P. Putta, N. Garg, T. Abraham, M. Lara, F. Lopez et al., "Real: Benchmarking autonomous agents on deterministic simulations of real websites," arXiv preprint arXiv:2504.11543, 2025.

[436] Y. Song, K. Thai, C. M. Pham, Y. Chang, M. Nadaf, and M. Iyyer, "Bearcubs: A benchmark for computer-using web agents," arXiv preprint arXiv:2503.07919, 2025.

[437] I. Evtimov, A. Zharmagambetov, A. Grattafiori, C. Guo, and K. Chaudhuri, "Wasp: Benchmarking web agent security against prompt injection attacks," arXiv preprint arXiv:2504.18575, 2025.

[438] D. Zhang, Z. Shen, R. Xie, S. Zhang, T. Xie, Z. Zhao, S. Chen, L. Chen, H. Xu, R. Cao, and K. Yu, "Mobile-env: Building qualified evaluation benchmarks for Ilm-gui interaction," 2024. [Online]. Available: https://arxiv.org/abs/2305.08144


<!-- Meanless: JOURNAL OF LATEX CLASS FILES, DECEMBER 2024<br>96 -->

[439] J. Lee, T. Min, M. An, C. Kim, and K. Lee, "Benchmarking mobile device control agents across diverse configurations," 2024. [Online]. Available: https://arxiv.org/abs/2404.16660

[440] C. Rawles, S. Clinckemaillie, Y. Chang, J. Waltz, G. Lau, M. Fair, A. Li, W. Bishop, W. Li, F. Campbell-Ajala, D. Toyama, R. Berry, D. Tyamagundlu, T. Lillicrap, and O. Riva, "Androidworld: A dynamic benchmarking environment for autonomous agents," 2024. [Online]. Available: https://arxiv.org/abs/2405.14573

[441] M. Xing, R. Zhang, H. Xue, Q. Chen, F. Yang, and Z. Xiao, "Understanding the weakness of large language model agents within a complex android environment," in Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, 2024, pp. 6061-6072.

[442] S. Deng, W. Xu, H. Sun, W. Liu, T. Tan, J. Liu, A. Li, J. Luan, B. Wang, R. Yan et al., "Mobile-bench: An evaluation benchmark for Ilm-based mobile agents," arXiv preprint arXiv:2407.00993, 2024.

[443] J. Lee, D. Hahm, J. S. Choi, W. B. Knox, and K. Lee, "Mobilesafe-tybench: Evaluating safety of autonomous agents in mobile device control," arXiv preprint arXiv:2410.17520, 2024.

[444] J. Chen, D. Yuen, B. Xie, Y. Yang, G. Chen, Z. Wu, L. Yixing, X. Zhou, W. Liu, S. Wang et al., "Spa-bench: A comprehensive benchmark for smartphone agent evaluation," in NeurlPS 2024 Workshop on Open-World Agents, 2024.

[445] L. Zhang, S. Wang, X. Jia, Z. Zheng, Y. Yan, L. Gao, Y. Li, and M. Xu, "Llamatouch: A faithful and scalable testbed for mobile ui task automation," 2024. [Online]. Available: https://arxiv.org/abs/2404.16054

[446] L. Wang, Y. Deng, Y. Zha, G. Mao, Q. Wang, T. Min, W. Chen, and S. Chen, "Mobileagentbench: An efficient and user-friendly benchmark for mobile Ilm agents," 2024. [Online]. Available: https://arxiv.org/abs/2406.08184

[447] K. Zhao, J. Song, L. Sha, H. Shen, Z. Chen, T. Zhao, X. Liang, and J. Yin, "Gui testing arena: A unified benchmark for advancing autonomous gui testing agent," arXiv preprint arXiv:2412.18426, 2024.

[448] Y. Chai, H. Li, J. Zhang, L. Liu, G. Wang, S. Ren, S. Huang, and H. Li, "A3: Android agent arena for mobile gui agents," 2025. [Online]. Available: https://arxiv.org/abs/2501.01149

[449] D. Ran, M. Wu, H. Yu, Y. Li, J. Ren, Y. Cao, X. Zeng, H. Lu, Z. Xu, M. Xu et al., "Beyond pass or fail: A multi-dimensional benchmark for mobile ui navigation," arXiv preprint arXiv:2501.02863, 2025.

[450] Y. Chen, X. Hu, K. Yin, J. Li, and S. Zhang, "Aeia-mn: Evaluating the robustness of multimodal Ilm-powered mobile agents against active environmental injection attacks," arXiv preprint arXiv:2502.13053, 2025.

[451] J. Sun, Z. Hua, and Y. Xia, "Autoeval: A practical framework for autonomous evaluation of mobile agents," arXiv preprint arXiv:2503.02403, 2025.

[452] W. Wang, Z. Yu, R. Ye, J. Zhang, S. Chen, and Y. Wang, "Fedmabench: Benchmarking mobile agents on decentralized heterogeneous user data," arXiv preprint arXiv:2503.05143, 2025.

[453] R. Bonatti, D. Zhao, F. Bonacci, D. Dupont, S. Abdali, Y. Li, Y. Lu, J. Wagle, K. Koishida, A. Bucker, L. Jang, and Z. Hui, "Windows agent arena: Evaluating multi-modal os agents at scale," 2024. [Online]. Available: https://arxiv.org/abs/2409.08264

[454] R. Cao, F. Lei, H. Wu, J. Chen, Y. Fu, H. Gao, X. Xiong, H. Zhang, Y. Mao, W. Hu, T. Xie, H. Xu, D. Zhang, S. Wang, R. Sun, P. Yin, C. Xiong, A. Ni, Q. Liu, V. Zhong, L. Chen, K. Yu, and T. Yu, "Spider2-v: How far are multimodal agents from automating data science and engineering workflows?" 2024. [Online]. Available: https://arxiv.org/abs/2407.10956

[455] Z. Wang, Y. Cui, L. Zhong, Z. Zhang, D. Yin, B. Y. Lin, and J. Shang, "Officebench: Benchmarking language agents across multiple applications for office automation," 2024. [Online]. Available: https://arxiv.org/abs/2407.19056

[456] H. H. Zhao, D. Gao, and M. Z. Shou, "Worldgui: Dynamic testing for comprehensive desktop gui automation," 2025. [Online]. Available: https://arxiv.org/abs/2502.08047

[457] S. Nayak, X. Jian, K. Q. Lin, J. A. Rodriguez, M. Kalsi, R. Awal, N. Chapados, M. T. Özsu, A. Agrawal, D. Vazquez et al., "Ui-vision: A desktop-centric gui benchmark for visual perception and interaction," arXiv preprint arXiv:2503.15661, 2025.

[458] B. Wang, X. Wang, J. Deng, T. Xie, R. Li, Y. Zhang, G. Li, T. J. Hua, I. Stoica, W.-L. Chiang, D. Yang, Y. Su, Y. Zhang, Z. Wang, V. Zhong, and T. Yu, "Computer agent arena: Compare & test computer use agents on crowdsourced real-world tasks," 2025.

[459] R. Kapoor, Y. P. Butala, M. Russak, J. Y. Koh, K. Kamble, W. Alshikh, and R. Salakhutdinov, "Omniact: A dataset and benchmark for enabling multimodal generalist autonomous agents for desktop and web," 2024. [Online]. Available: https://arxiv.org/abs/2402.17553

[460] K. Q. Lin, L. Li, D. Gao, Q. WU, M. Yan, Z. Yang, L. Wang, and M. Z. Shou, "Videogui: A benchmark for gui automation from instructional videos," 2024. [Online]. Available: https://arxiv.org/abs/2406.10227

[461] T. Xu, L. Chen, D.-J. Wu, Y. Chen, Z. Zhang, X. Yao, Z. Xie, Y. Chen, S. Liu, B. Qian, P. Torr, B. Ghanem, and G. Li, "Crab: Cross-environment agent benchmark for multimodal language model agents," 2024. [Online]. Available: https://arxiv.org/abs/2407.01511

[462] Y. Fan, L. Ding, C.-C. Kuo, S. Jiang, Y. Zhao, X. Guan, J. Yang, Y. Zhang, and X. E. Wang, "Read anywhere pointed: Layout-aware gui screen reading with tree-of-lens grounding," 2024. [Online]. Available: https://arxiv.org/abs/2406.19263

[463] D. Zimmermann and A. Koziolek, "Gui-based software testing: An automated approach using gpt-4 and selenium webdriver," in 2023 38th IEEE/ACM International Conference on Automated Software Engineering Workshops (ASEW). IEEE, 2023, pp. 171-174.

[464] J. Yoon, R. Feldt, and S. Yoo, "Intent-driven mobile gui testing with autonomoufs large language model agents," in 2024 IEEE Conference on Software Testing, Verification and Validation (ICST). IEEE, 2024, pp. 129-139.

[465] Y. Hu, X. Wang, Y. Wang, Y. Zhang, S. Guo, C. Chen, X. Wang, and Y. Zhou, "Auitestagent: Automatic requirements oriented gui function testing," 2024. [Online]. Available: https://arxiv.org/abs/2407.09018

[466] Z. Liu, C. Li, C. Chen, J. Wang, B. Wu, Y. Wang, J. Hu, and Q. Wang, "Vision-driven automated mobile gui testing via multimodal large language model," 2024. [Online]. Available: https://arxiv.org/abs/2407.03037

[467] M. Taeb, A. Swearngin, E. Schoop, R. Cheng, Y. Jiang, and J. Nichols, "Axnaux: Replaying accessibility tests from natural language," in Proceedings of the CHI Conference on Human Factors in Computing Systems, 2024, pp. 1-16.

[468] C. Cui, T. Li, J. Wang, C. Chen, D. Towey, and R. Huang, "Large language models for mobile gui text input generation: An empirical study," arXiv preprint arXiv:2404.08948, 2024.

[469] Z. Liu, C. Chen, J. Wang, X. Che, Y. Huang, J. Hu, and Q. Wang, "Fill in the blank: Context-aware automated text input generation for mobile gui testing," in 2023 IEEE/ACM 45th International Conference on Software Engineering (ICSE). IEEE, 2023, pp. 1355-1367.

[470] Y. Huang, J. Wang, Z. Liu, Y. Wang, S. Wang, C. Chen, Y. Hu, and Q. Wang, "Crashtranslator: Automatically reproducing mobile application crashes directly from stack trace," in Proceedings of the 46th IEEE/ACM International Conference on Software Engineering, 2024, pp. 1-13.

[471] S. Feng and C. Chen, "Prompting is all you need: Automated android bug replay with large language models," in Proceedings of the 46th IEEE/ACM International Conference on Software Engineering, 2024, pp. 1-13.

[472] L. Ding, J. Bheemanpally, and Y. Zhang, "Improving technical" how-to" query accuracy with automated search results verification and reranking," arXiv preprint arXiv:2404.08860, 2024.

[473] B. Beyzaei, S. Talebipour, G. Rafiei, N. Medvidovic, and S. Malek, "Automated test transfer across android apps using large language models," arXiv preprint arXiv:2411.17933, 2024.

[474] Y. Lu, B. Yao, H. Gu, J. Huang, J. Wang, L. Li, J. Gesi, Q. He, T. J.-J. Li, and D. Wang, "Uxagent: An Ilm agent-based usability testing framework for web design," arXiv preprint arXiv:2502.12561, 2025.

[475] D. Ran, H. Wang, Z. Song, M. Wu, Y. Cao, Y. Zhang, W. Yang, and T. Xie, "Guardian: A runtime framework for Ilm-based ui exploration," in Proceedings of the 33rd ACM SIGSOFT International Symposium on Software Testing and Analysis, 2024, pp. 958-970.

[476] Y. Li, Y. Li, and Y. Yang, "Test-agent: A multimodal app automation testing framework based on the large language model," in 2024 IEEE 4th International Conference on Digital Twins and Parallel Intelligence (DTPI). IEEE, 2024, pp. 609-614.

[477] B. F. Demissie, Y. N. Tun, L. K. Shar, and M. Ceccato, "Vlm-fuzz: Vision language model assisted recursive depth-first search exploration for effective ui testing of android apps," arXiv preprint arXiv:2504.11675, 2025.


<!-- Meanless: JOURNAL OF LATEX CLASS FILES, DECEMBER 2024<br>97 -->

[478] E. Yapağci, Y. A. S. Öztürk, and E. Tüzün, "Bugcraft: End-to-end crash bug reproduction using Ilm agents in minecraft," arXiv preprint arXiv:2503.20036, 2025.

[479] X. Li, J. Cao, Y. Liu, S.-C. Cheung, and H. Wang, "Reusedroid: A vlm-empowered android ui test migrator boosted by active feedback," arXiv preprint arXiv:2504.02357, 2025.

[480] A. Chevrot, A. Vernotte, J.-R. Falleri, X. Blanc, and B. Leg-eard, "Are autonomous web agents good testers?" arXiv preprint arXiv:2504.01495, 2025.

[481] T. Rosenbach, D. Heidrich, and A. Weinert, "Automated testing of the gui of a real-life engineering software using large language models," in 2025 IEEE International Conference on Software Testing, Verification and Validation Workshops (ICSTW). IEEE, 2025, pp. 103-110.

[482] Q. Kong, Z. Lv, Y. Xiong, J. Sun, T. Su, D. Wang, L. Li, X. Yang, and G. Huo, "Prophetagent: Automatically synthesizing gui tests from test cases in natural language for mobile apps."

[483] S. Feng, C. Du, H. Liu, Q. Wang, Z. Lv, G. Huo, X. Yang, and C. Chen, "Agent for user: Testing multi-user interactive features in tiktok," arXiv preprint arXiv:2504.15474, 2025.

[484] J. Gorniak, Y. Kim, D. Wei, and N. W. Kim, "Vizability: Enhancing chart accessibility with Ilm-based conversational interaction," in Proceedings of the 37th Annual ACM Symposium on User Interface Software and Technology, 2024, pp. 1-19.

[485] Y. Ye, X. Cong, S. Tian, J. Cao, H. Wang, Y. Qin, Y. Lu, H. Yu, H. Wang, Y. Lin et al., "Proagent: From robotic process automation to agentic process automation," arXiv preprint arXiv:2311.10751, 2023.

[486] Y. Guan, D. Wang, Z. Chu, S. Wang, F. Ni, R. Song, and C. Zhuang, "Intelligent agents with Ilm-based process automation," in Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, 2024, pp. 5018-5027.

[487] M. D. Vu, H. Wang, Z. Li, J. Chen, S. Zhao, Z. Xing, and C. Chen, "Gptvoicetasker: LIm-powered virtual assistant for smartphone," arXiv preprint arXiv:2401.14268, 2024.

[488] L. Pan, B. Wang, C. Yu, Y. Chen, X. Zhang, and Y. Shi, "Autotask: Executing arbitrary voice commands by exploring and learning from mobile gui," arXiv preprint arXiv:2312.16062, 2023.

[489] D. Gao, S. Hu, Z. Bai, Q. Lin, and M. Z. Shou, "Assisteditor: Multi-agent collaboration for gui workflow automation in video creation," in Proceedings of the 32nd ACM International Conference on Multimedia, 2024, pp. 11 255-11 257.

[490] T. Huang, C. Yu, W. Shi, Z. Peng, D. Yang, W. Sun, and Y. Shi, "Promptrpa: Generating robotic process automation on smart-phones from textual prompts," arXiv preprint arXiv:2404.02475, 2024.

[491] W. Gao, K. Du, Y. Luo, W. Shi, C. Yu, and Y. Shi, "Easyask: An in-app contextual tutorial search assistant for older adults with voice and touch inputs," Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies, vol. 8, no. 3, pp. 1-27, 2024.

[492] OpenAdapt AI, "OpenAdapt: Open Source Generative Process Automation," 2024, accessed: 2024-10-26. [Online]. Available: https://github.com/OpenAdaptAI/OpenAdapt

[493] AgentSeaf AI. (2024) Introduction to agentsea platform. Accessed: 2024-10-26. [Online]. Available: https://www.agentsea.ai/

[494] O. Interpreter, "Open interpreter: A natural language interface for computers," GitHub repository, 2024, accessed: 2024- 10-27. [Online]. Available: https://github.com/OpenInterpreter/ open-interpreter

[495] MultiOn AI. (2024) Multion ai: Ai agents that act on your behalf. Accessed: 2024-10-26. [Online]. Available: https://www.multion.ai/

[496] HONOR, "Honor introduces magicos 9.0," 2024, accessed: 2024-11-16. [Online]. Available: https://www.fonearena.com/blog/ 438680/honor-magicos-9-0-features.html

[497] T. Srinivasan and S. Patapati, "Webnav: An intelligent agent for voice-controlled web navigation," arXiv preprint arXiv:2503.13843, 2025.

[498] H. W. Chung, L. Hou, S. Longpre, B. Zoph, Y. Tay, W. Fedus, Y. Li, X. Wang, M. Dehghani, S. Brahma et al., "Scaling instruction-finetuned language models," Journal of Machine Learning Research, vol. 25, no. 70, pp. 1-53, 2024.

[499] T. GLM, A. Zeng, B. Xu, B. Wang, C. Zhang, D. Yin, D. Rojas, G. Feng, H. Zhao, H. Lai et al., "Chatglm: A family of large language models from glm-130b to glm-4 all tools," arXiv preprint arXiv:2406.12793, 2024.

[500] C. B. Browne, E. Powley, D. Whitehouse, S. M. Lucas, P. I. Cowling, P. Rohlfshagen, S. Tavener, D. Perez, S. Samothrakis, and S. Colton, "A survey of monte carlo tree search methods," IEEE Transactions on Computational Intelligence and AI in games, vol. 4, no. 1, pp. 1-43, 2012.

[501] A. Q. Jiang, A. Sablayrolles, A. Mensch, C. Bamford, D. S. Chaplot, D. d. I. Casas, F. Bressand, G. Lengyel, G. Lample, L. Saulnier et al., "Mistral 7b," arXiv preprint arXiv:2310.06825, 2023.

[502] G. Team, T. Mesnard, C. Hardin, R. Dadashi, S. Bhupatiraju, S. Pathak, L. Sifre, M. Rivière, M. S. Kale, J. Love et al., "Gemma: Open models based on gemini research and technology," arXiv preprint arXiv:2403.08295, 2024.

[503] R. Rafailov, A. Sharma, E. Mitchell, C. D. Manning, S. Ermon, and C. Finn, "Direct preference optimization: Your language model is secretly a reward model," Advances in Neural Information Processing Systems, vol. 36, 2024.

[504] D. Guo, D. Yang, H. Zhang, J. Song, R. Zhang, R. Xu, Q. Zhu, S. Ma, P. Wang, X. Bi et al., "Deepseek-r1: Incentivizing reasoning capability in Ilms via reinforcement learning," arXiv preprint arXiv:2501.12948, 2025.

[505] CogAgent Team, "Cogagent: Cognitive ai agent platform," https: //cogagent.aminer.cn/home| 2024, accessed: 2024-12-17.

[506] S. Yao, D. Yu, J. Zhao, I. Shafran, T. Griffiths, Y. Cao, and K. Narasimhan, "Tree of thoughts: Deliberate problem solving with large language models," Advances in Neural Information Processing Systems, vol. 36, 2024.

[507] R. Anil, A. M. Dai, O. Firat, M. Johnson, D. Lepikhin, A. Passos, S. Shakeri, E. Taropa, P. Bailey, Z. Chen, E. Chu, J. H. Clark, L. E. Shafey, Y. Huang, K. Meier-Hellstern, G. Mishra, E. Moreira, M. Omernick, K. Robinson, S. Ruder, Y. Tay, K. Xiao, Y. Xu, Y. Zhang, G. H. Abrego, J. Ahn, J. Austin, P. Barham, J. Botha, J. Bradbury, S. Brahma, K. Brooks, M. Catasta, Y. Cheng, C. Cherry, C. A. Choquette-Choo, A. Chowdhery, C. Crepy, S. Dave, M. Dehghani, S. Dev, J. Devlin, M. Díaz, N. Du, E. Dyer, V. Feinberg, F. Feng, V. Fienber, M. Freitag, X. Garcia, S. Gehrmann, L. Gonzalez, G. Gur-Ari, S. Hand, H. Hashemi, L. Hou, J. Howland, A. Hu, J. Hui, J. Hurwitz, M. Isard, A. Ittycheriah, M. Jagielski, W. Jia, K. Kenealy, M. Krikun, S. Kudugunta, C. Lan, K. Lee, B. Lee, E. Li, M. Li, W. Li, Y. Li, J. Li, H. Lim, H. Lin, Z. Liu, F. Liu, M. Maggioni, A. Mahendru, J. Maynez, V. Misra, M. Moussalem, Z. Nado, J. Nham, E. Ni, A. Nystrom, A. Parrish, M. Pellat, M. Polacek, A. Polozov, R. Pope, S. Qiao, E. Reif, B. Richter, P. Riley, A. C. Ros, A. Roy, B. Saeta, R. Samuel, R. Shelby, A. Slone, D. Smilkov, D. R. So, D. Sohn, S. Tokumine, D. Valter, V. Vasudevan, K. Vodrahalli, X. Wang, P. Wang, Z. Wang, T. Wang, J. Wieting, Y. Wu, K. Xu, Y. Xu, L. Xue, P. Yin, J. Yu, Q. Zhang, S. Zheng, C. Zheng, W. Zhou, D. Zhou, S. Petrov, and Y. Wu, "Palm 2 technical report," 2023. [Online]. Available: https://arxiv.org/abs/2305.10403

[508] Baidu Research, "ERNIE Bot: Baidu's Knowledge-Enhanced Large Language Model Built on Full AI Stack Technology," 2024, [Online; accessed 9-November-2024]. [Online]. Available: https://research.baidu.com/Blog/index-view?id=183

[509] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark et al., "Learning transferable visual models from natural language supervision," in International conference on machine learning. PMLR, 2021, pp. 8748-8763.

[510] W.-L. Chiang, Z. Li, Z. Lin, Y. Sheng, Z. Wu, H. Zhang, L. Zheng, S. Zhuang, Y. Zhuang, J. E. Gonzalez, I. Stoica, and E. P. Xing, "Vicuna: An open-source chatbot impressing gpt-4 with 90%* chatgpt quality," March 2023. [Online]. Available: https://Imsys.org/blog/2023-03-30-vicuna/

[511] J. Bai, S. Bai, Y. Chu, Z. Cui, K. Dang, X. Deng, Y. Fan, W. Ge, Y. Han, F. Huang, B. Hui, L. Ji, M. Li, J. Lin, R. Lin, D. Liu, G. Liu, C. Lu, K. Lu, J. Ma, R. Men, X. Ren, X. Ren, C. Tan, S. Tan, J. Tu, P. Wang, S. Wang, W. Wang, S. Wu, B. Xu, J. Xu, A. Yang, H. Yang, J. Yang, S. Yang, Y. Yao, B. Yu, H. Yuan, Z. Yuan, J. Zhang, X. Zhang, Y. Zhang, Z. Zhang, C. Zhou, J. Zhou, X. Zhou, and T. Zhu, "Qwen technical report," 2023. [Online]. Available: https://arxiv.org/abs/2309.16609

[512] M. Deitke, C. Clark, S. Lee, R. Tripathi, Y. Yang, J. S. Park, M. Salehi, N. Muennighoff, K. Lo, L. Soldaini, J. Lu, T. Anderson, E. Bransom, K. Ehsani, H. Ngo, Y. Chen, A. Patel, M. Yatskar, C. Callison-Burch, A. Head, R. Hendrix, F. Bastani, E. VanderBilt, N. Lambert, Y. Chou, A. Chheda, J. Sparks, S. Skjonsberg, M. Schmitz, A. Sarnat, B. Bischoff, P. Walsh, C. Newell, P. Wolters, T. Gupta, K.-H. Zeng, J. Borchardt, D. Groeneveld, C. Nam, S. Lebrecht, C. Wittlif, C. Schoenick, O. Michel, R. Krishna, L. Weihs, N. A. Smith, H. Hajishirzi, R. Girshick, A. Farhadi, and A. Kembhavi, "Molmo and pixmo: Open weights and open data for state-of-the-art vision-language models," 2024. [Online]. Available: https://arxiv.org/abs/2409.17146


<!-- Meanless: JOURNAL OF IATEX CLASS FILES, DECEMBER 2024<br>98 -->

[513] OpenAI, "Operator system card," Jan. 2025, released on January 23, 2025.

[514] B. Xiao, H. Wu, W. Xu, X. Dai, H. Hu, Y. Lu, M. Zeng, C. Liu, and L. Yuan, "Florence-2: Advancing a unified representation for a variety of vision tasks," 2023. [Online]. Available: https://arxiv.org/abs/2311.06242

[515] W. Li, W. Bishop, A. Li, C. Rawles, F. Campbell-Ajala, D. Tyama-gundlu, and O. Riva, "On the effects of data scale on computer control agents," arXiv preprint arXiv:2406.03679, 2024.

[516] D. Chen, Y. Huang, Z. Ma, H. Chen, X. Pan, C. Ge, D. Gao, Y. Xie, Z. Liu, J. Gao et al., "Data-juicer: A one-stop data processing system for large language models," in Companion of the 2024 International Conference on Management of Data, 2024, pp. 120- 134.

[517] B. Ding, C. Qin, R. Zhao, T. Luo, X. Li, G. Chen, W. Xia, J. Hu, L. A. Tuan, and S. Joty, "Data augmentation using Ilms: Data perspectives, learning paradigms and challenges," in Findings of the Association for Computational Linguistics ACL 2024, 2024, pp. 1679-1705.

[518] Z. Tan, D. Li, S. Wang, A. Beigi, B. Jiang, A. Bhattacharjee, M. Karami, J. Li, L. Cheng, and H. Liu, "Large language models for data annotation: A survey," arXiv preprint arXiv:2402.13446, 2024.

[519] J. Andreas, J. Bufe, D. Burkett, C. Chen, J. Clausman, J. Crawford, K. Crim, J. DeLoach, L. Dorner, J. Eisner et al., "Task-oriented dialogue as dataflow synthesis," Transactions of the Association for Computational Linguistics, vol. 8, pp. 556-571, 2020.

[520] Z. Guo, S. Cheng, H. Wang, S. Liang, Y. Qin, P. Li, Z. Liu, M. Sun, and Y. Liu, "Stabletoolbench: Towards stable large-scale benchmarking on tool learning of large language models," 2024.

[521] C. Ma, J. Zhang, Z. Zhu, C. Yang, Y. Yang, Y. Jin, Z. Lan, L. Kong, and J. He, "Agentboard: An analytical evaluation board of multi-turn Ilm agents," arXiv preprint arXiv:2401.13178, 2024.

[522] A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly, J. Uszkoreit, and N. Houlsby, "An image is worth 16x16 words: Transformers for image recognition at scale," 2021. [Online]. Available: https://arxiv.org/abs/2010.11929

[523] H. Laurengon, L. Tronchon, M. Cord, and V. Sanh, "What matters when building vision-language models?" arXiv preprint arXiv:2405.02246, 2024.

[524] Z. Du, Y. Qian, X. Liu, M. Ding, J. Qiu, Z. Yang, and J. Tang, "Glm: General language model pretraining with autoregressive blank infilling," arXiv preprint arXiv:2103.10360, 2021.

[525] Z. Liu, Y. Lin, Y. Cao, H. Hu, Y. Wei, Z. Zhang, S. Lin, and B. Guo, "Swin transformer: Hierarchical vision transformer using shifted windows," in Proceedings of the IEEE/CVF international conference on computer vision, 2021, pp. 10012-10022.

[526] B. Rozière, J. Gehring, F. Gloeckle, S. Sootla, I. Gat, X. E. Tan, Y. Adi, J. Liu, R. Sauvestre, T. Remez, J. Rapin, A. Kozhevnikov, I. Evtimov, J. Bitton, M. Bhatt, C. C. Ferrer, A. Grattafiori, W. Xiong, A. Défossez, J. Copet, F. Azhar, H. Touvron, L. Martin, N. Usunier, T. Scialom, and G. Synnaeve, "Code Ilama: Open foundation models for code," 2024. [Online]. Available: https://arxiv.org/abs/2308.12950

[527] W. Cai, J. Jiang, F. Wang, J. Tang, S. Kim, and J. Huang, "A survey on mixture of experts," arXiv preprint arXiv:2407.06204, 2024.

[528] I. Turc, M.-W. Chang, K. Lee, and K. Toutanova, "Well-read students learn better: On the importance of pre-training compact models," arXiv preprint arXiv:1908.08962, 2019.

[529] L. Beyer, A. Steiner, A. S. Pinto, A. Kolesnikov, X. Wang, D. Salz, M. Neumann, I. Alabdulmohsin, M. Tschannen, E. Bugliarello et al., "Paligemma: A versatile 3b vlm for transfer," arXiv preprint arXiv:2407.07726, 2024.

[530] Z. Cai, M. Cao, H. Chen, K. Chen, K. Chen, X. Chen, X. Chen, Z. Chen, Z. Chen, P. Chu et al., "Internlm2 technical report," arXiv preprint arXiv:2403.17297, 2024.

[531] R. Rombach, A. Blattmann, D. Lorenz, P. Esser, and B. Ommer, "High-resolution image synthesis with latent diffusion models," in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2022, pp. 10684-10695.

[532] Y.-C. Hsiao, F. Zubach, G. Baechler, V. Carbune, J. Lin, M. Wang, S. Sunkara, Y. Zhu, and J. Chen, "Screenga: Large-scale question-answer pairs over mobile app screenshots," 2024. [Online]. Available: https://arxiv.org/abs/2209.08199

[533] A. Jaech, A. Kalai, A. Lerer, A. Richardson, A. El-Kishky, A. Low, A. Helyar, A. Madry, A. Beutel, A. Carney et al., "Openai o1 system card," arXiv preprint arXiv:2412.16720, 2024.

[534] Z. Shao, P. Wang, Q. Zhu, R. Xu, J. Song, X. Bi, H. Zhang, M. Zhang, Y. Li, Y. Wu et al., "Deepseekmath: Pushing the limits of mathematical reasoning in open language models," arXiv preprint arXiv:2402.03300, 2024.

[535] MosaicML, "Mosaicml: Mpt-7b," 2023, accessed: 2024-11-19. [Online]. Available: https://www.mosaicml.com/blog/mpt-7b

[536] X. Chen, X. Wang, L. Beyer, A. Kolesnikov, J. Wu, P. Voigtlaender, B. Mustafa, S. Goodman, I. Alabdulmohsin, P. Padlewski et al., "Pali-3 vision language models: Smaller, faster, stronger," arXiv preprint arXiv:2310.09199, 2023.

[537] Q. Sun, Y. Fang, L. Wu, X. Wang, and Y. Cao, "Eva-clip: Improved training techniques for clip at scale," arXiv preprint arXiv:2303.15389, 2023.

[538] D. Guo, Q. Zhu, D. Yang, Z. Xie, K. Dong, W. Zhang, G. Chen, X. Bi, Y. Wu, Y. Li et al., "Deepseek-coder: When the large language model meets programming-the rise of code intelligence," arXiv preprint arXiv:2401.14196, 2024.

[539] Z. Liu, H. Mao, C.-Y. Wu, C. Feichtenhofer, T. Darrell, and S. Xie, "A convnet for the 2020s," in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2022, pp. 11976- 11986.

[540] S. Bai, K. Chen, X. Liu, J. Wang, W. Ge, S. Song, K. Dang, P. Wang, S. Wang, J. Tang et al., "Qwen2. 5-vl technical report," arXiv preprint arXiv:2502.13923, 2025.

[541] X. Liu, H. Yu, H. Zhang, Y. Xu, X. Lei, H. Lai, Y. Gu, H. Ding, K. Men, K. Yang et al., "Agentbench: Evaluating Ilms as agents," arXiv preprint arXiv:2308.03688, 2023.

[542] D. Zimmermann and A. Koziolek, "Automating gui-based software testing with gpt-3," in 2023 IEEE International Conference on Software Testing, Verification and Validation Workshops (ICSTW), 2023, pp. 62-65.

[543] J. Wang, Y. Huang, C. Chen, Z. Liu, S. Wang, and Q. Wang, "Software testing with large language models: Survey, landscape, and vision," IEEE Transactions on Software Engineering, 2024.

[544] K. Q. Lin, P. Zhang, J. Chen, S. Pramanick, D. Gao, A. J. Wang, R. Yan, and M. Z. Shou, "Univtg: Towards unified video-language temporal grounding," in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 2794-2804.

[545] F. Al, "Eko - build production-ready agentic workflow with natural language," https://eko.fellou.ai/ 2025, accessed: 2025-01-15.

[546] Z. Liao, L. Mo, C. Xu, M. Kang, J. Zhang, C. Xiao, Y. Tian, B. Li, and H. Sun, "Eia: Environmental injection attack on generalist web agents for privacy leakage," arXiv preprint arXiv:2409.11295, 2024.

[547] F. He, T. Zhu, D. Ye, B. Liu, W. Zhou, and P. S. Yu, "The emerged security and privacy of Ilm agent: A survey with case studies," arXiv preprint arXiv:2407.19354, 2024.

[548] Y. Gan, Y. Yang, Z. Ma, P. He, R. Zeng, Y. Wang, Q. Li, C. Zhou, S. Li, T. Wang, Y. Gao, Y. Wu, and S. Ji, "Navigating the risks: A survey of security, privacy, and ethics threats in Ilm-based agents," 2024. [Online]. Available: https://arxiv.org/abs/2411.09523

[549] Y. Yang, X. Yang, S. Li, C. Lin, Z. Zhao, C. Shen, and T. Zhang, "Security matrix for multimodal agents on mobile devices: A systematic and proof of concept study," arXiv preprint arXiv:2407.09295, 2024.

[550] X. Zhang, H. Xu, Z. Ba, Z. Wang, Y. Hong, J. Liu, Z. Qin, and K. Ren, "Privacyasst: Safeguarding user privacy in tool-using large language model agents," IEEE Transactions on Dependable and Secure Computing, 2024.

[551] J. Xu, Z. Li, W. Chen, Q. Wang, X. Gao, Q. Cai, and Z. Ling, "On-device language models: A comprehensive review," arXiv preprint arXiv:2409.00088, 2024.

[552] G. Qu, Q. Chen, W. Wei, Z. Lin, X. Chen, and K. Huang, "Mobile edge intelligence for large language models: A contemporary survey," arXiv preprint arXiv:2407.18921, 2024.

[553] J. Lin, J. Tang, H. Tang, S. Yang, W.-M. Chen, W.-C. Wang, G. Xiao, X. Dang, C. Gan, and S. Han, "Awq: Activation-aware weight quantization for on-device Ilm compression and acceleration," Proceedings of Machine Learning and Systems, vol. 6, pp. 87-100, 2024.


<!-- Meanless: JOURNAL OF LATEX CLASS FILES, DECEMBER 2024<br>99 -->

[554] Z. Liu, C. Zhao, F. Iandola, C. Lai, Y. Tian, I. Fedorov, Y. Xiong, E. Chang, Y. Shi, R. Krishnamoorthi et al., "Mobilellm: Optimizing sub-billion parameter language models for on-device use cases," arXiv preprint arXiv:2402.14905, 2024.

[555] Z. Zhou, X. Ning, K. Hong, T. Fu, J. Xu, S. Li, Y. Lou, L. Wang, Z. Yuan, X. Li et al., "A survey on efficient inference for large language models," arXiv preprint arXiv:2404.14294, 2024.

[556] W. Kuang, B. Qian, Z. Li, D. Chen, D. Gao, X. Pan, Y. Xie, Y. Li, B. Ding, and J. Zhou, "Federatedscope-Ilm: A comprehensive package for fine-tuning large language models in federated learning," in Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, 2024, pp. 5260-5271.

[557] P. Mai, R. Yan, Z. Huang, Y. Yang, and Y. Pang, "Split-and-denoise: Protect large language model inference with local differential privacy," arXiv preprint arXiv:2310.09130, 2023.

[558] L. de Castro, A. Polychroniadou, and D. Escudero, "Privacy-preserving large language model inference via gpu-accelerated fully homomorphic encryption," in Neurips Safe Generative AI Workshop 2024.

[559] J. Wolff, W. Lehr, and C. S. Yoo, "Lessons from gdpr for ai policymaking," Virginia Journal of Law & Technology, vol. 27, no. 4, p. 2, 2024.

[560] Z. Zhang, M. Jia, H.-P. Lee, B. Yao, S. Das, A. Lerner, D. Wang, and T. Li, ""it's a fair game", or is it? examining how users navigate disclosure risks and benefits when using Ilm-based conversational agents," in Proceedings of the CHI Conference on Human Factors in Computing Systems, 2024, pp. 1-26.

[561] B. Li, Y. Jiang, V. Gadepally, and D. Tiwari, "Llm inference serving: Survey of recent advances and opportunities," arXiv preprint arXiv:2407.12391, 2024.

[562] M. Xu, W. Yin, D. Cai, R. Yi, D. Xu, Q. Wang, B. Wu, Y. Zhao, C. Yang, S. Wang et al., "A survey of resource-efficient llm and multimodal foundation models," arXiv preprint arXiv:2401.08092, 2024.

[563] D. Chen, Y. Liu, M. Zhou, Y. Zhao, H. Wang, S. Wang, X. Chen, T. F. Bissyandé, J. Klein, and L. Li, "LIm for mobile: An initial roadmap," arXiv preprint arXiv:2407.06573, 2024.

[564] L. Krupp, D. Geißler, P. Lukowicz, and J. Karolus, "Towards sustainable web agents: A plea for transparency and dedicated metrics for energy consumption," arXiv preprint arXiv:2502.17903, 2025.

[565] Z. Wan, X. Wang, C. Liu, S. Alam, Y. Zheng, J. Liu, Z. Qu, S. Yan, Y. Zhu, Q. Zhang et al., "Efficient large language models: A survey," arXiv preprint arXiv:2312.03863, 2023.

[566] X. Xu, M. Li, C. Tao, T. Shen, R. Cheng, J. Li, C. Xu, D. Tao, and T. Zhou, "A survey on knowledge distillation of large language models," arXiv preprint arXiv:2402.13116, 2024.

[567] C. Kachris, "A survey on hardware accelerators for large language models," arXiv preprint arXiv:2401.09890, 2024.

[568] W. Lee, J. Lee, J. Seo, and J. Sim, "\{InfiniGen\}: Efficient generative inference of large language models with dynamic \{KV\} cache management," in 18th USENIX Symposium on Operating Systems Design and Implementation (OSDI 24), 2024, pp. 155-172.

[569] Z. Wang, J. Wohlwend, and T. Lei, "Structured pruning of large language models," arXiv preprint arXiv:1910.04732, 2019.

[570] B. Wu, Y. Zhong, Z. Zhang, G. Huang, X. Liu, and X. Jin, "Fast distributed inference serving for large language models," arXiv preprint arXiv:2305.05920, 2023.

[571] U. Anwar, A. Saparov, J. Rando, D. Paleka, M. Turpin, P. Hase, E. S. Lubana, E. Jenner, S. Casper, O. Sourbut et al., "Foundational challenges in assuring alignment and safety of large language models," arXiv preprint arXiv:2404.09932, 2024.

[572] L. Zhong and Z. Wang, "A study on robustness and reliability of large language model code generation," arXiv preprint arXiv:2308.10335, 2023.

[573] T. Yuan, Z. He, L. Dong, Y. Wang, R. Zhao, T. Xia, L. Xu, B. Zhou, F. Li, Z. Zhang et al., "R-judge: Benchmarking safety risk awareness for Ilm agents," arXiv preprint arXiv:2401.10019, 2024.

[574] L. Zhang, Q. Jin, H. Huang, D. Zhang, and F. Wei, "Respond in my language: Mitigating language inconsistency in response generation based on large language models," in Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 2024, pp. 4177-4192.

[575] H. Zhao, T. Chen, and Z. Wang, "On the robustness of gui grounding models against image attacks," arXiv preprint arXiv:2504.04716, 2025.

[576] Y. Zhang, Y. Li, L. Cui, D. Cai, L. Liu, T. Fu, X. Huang, E. Zhao, Y. Zhang, Y. Chen, L. Wang, A. T. Luu, W. Bi, F. Shi, and S. Shi, "Siren's song in the ai ocean: A survey on hallucination in large language models," 2023. [Online]. Available: https://arxiv.org/abs/2309.01219

[577] J. Y. F. Chiang, S. Lee, J.-B. Huang, F. Huang, and Y. Chen, "Why are web ai agents more vulnerable than standalone Ilms? a security analysis," arXiv preprint arXiv:2502.20383, 2025.

[578] C. Chen, Z. Zhang, B. Guo, S. Ma, I. Khalilov, S. A. Gebreegziabher, Y. Ye, Z. Xiao, Y. Yao, T. Li et al., "The obvious invisible threat: LIm-powered gui agents' vulnerability to fine-print injections," arXiv preprint arXiv:2504.11281, 2025.

[579] C. Xu, M. Kang, J. Zhang, Z. Liao, L. Mo, M. Yuan, H. Sun, and B. Li, "Advweb: Controllable black-box attacks on vlm-powered web agents," arXiv preprint arXiv:2410.17401, 2024.

[580] Y. Yang, X. Yang, S. Li, C. Lin, Z. Zhao, C. Shen, and T. Zhang, "Systematic categorization, construction and evaluation of new attacks against multi-modal mobile gui agents," 2025. [Online]. Available: https://arxiv.org/abs/2407.09295

[581] L. Aichberger, A. Paren, Y. Gal, P. Torr, and A. Bibi, "Attacking multimodal os agents with malicious image patches," in ICLR 2025 Workshop on Foundation Models in the Wild.

[582] J. Lee, D. Lee, C. Choi, Y. Im, J. Wi, K. Heo, S. Oh, S. Lee, and I. Shin, "Safeguarding mobile gui agent via logic-based action verification," arXiv preprint arXiv:2503.18492, 2025.

[583] L. Pan, M. S. Saxon, W. Xu, D. Nathani, X. Wang, and W. Y. Wang, "Automatically correcting large language models: Surveying the landscape of diverse self-correction strategies," ArXiv, vol. abs/2308.03188, 2023. [Online]. Available: https://api.semanticscholar.org/CorpusID:260682695

[584] X. Huang, W. Ruan, W. Huang, G. Jin, Y. Dong, C. Wu, S. Bensalem, R. Mu, Y. Qi, X. Zhao, K. Cai, Y. Zhang, S. Wu, P. Xu, D. Wu, A. Freitas, and M. A. Mustafa, "A survey of safety and trustworthiness of large language models through the lens of verification and validation," Artif. Intell. Rev., vol. 57, p. 175, 2023. [Online]. Available: https://api.semanticscholar.org/CorpusID:258823083

[585] S. Jha, S. K. Jha, P. Lincoln, N. D. Bastian, A. Velasquez, and S. Neema, "Dehallucinating large language models using formal methods guided iterative prompting," 2023 IEEE International Conference on Assured Autonomy (ICAA), pp. 149-152, 2023. [Online]. Available: https://api.semanticscholar.org/CorpusID: 260810131

[586] Q. Zhang, T. Zhang, J. Zhai, C. Fang, B.-C. Yu, W. Sun, and Z. Chen, "A critical review of large language model on software engineering: An example from chatgpt and automated program repair," ArXiv, vol. abs/2310.08879, 2023. [Online]. Available: https://api.semanticscholar.org/CorpusID:264127977

[587] R. Koo and S. Toueg, "Checkpointing and rollback-recovery for distributed systems," IEEE Transactions on Software Engineering, vol. SE-13, pp. 23-31, 1986. [Online]. Available: https://api.semanticscholar.org/CorpusID:206777989

[588] Y. Luo, Q. Zhang, Q. Shen, H. Liu, and Z. Wu, "Android multi-level system permission management approach," ArXiv, vol. abs/1712.02217, 2017. [Online]. Available: https://api.semanticscholar.org/CorpusID:20909985

[589] H. Hao, V. Singh, and W. Du, "On the effectiveness of api-level access control using bytecode rewriting in android," in Proceedings of the 8th ACM SIGSAC symposium on Information, computer and communications security, 2013, pp. 25-36.

[590] A. P. Felt, E. Chin, S. Hanna, D. Song, and D. Wagner, "Android permissions demystified," in Proceedings of the 18th ACM conference on Computer and communications security, 2011, pp. 627-638.

[591] M. Lutaaya, "Rethinking app permissions on ios," in Extended Abstracts of the 2018 CHI Conference on Human Factors in Computing Systems, 2018, pp. 1-6.

[592] Z. Xiang, L. Zheng, Y. Li, J. Hong, Q. Li, H. Xie, J. Zhang, Z. Xiong, C. Xie, C. Yang et al., "Guardagent: Safeguard Ilm agents by a guard agent via knowledge-enabled reasoning," arXiv preprint arXiv:2406.09187, 2024.

[593] S. Berkovits, J. D. Guttman, and V. Swarup, "Authentication for mobile agents," in Mobile Agents and Security, 1998. [Online]. Available: https://api.semanticscholar.org/CorpusID:13987376

[594] J. Gao, S. A. Gebreegziabher, K. T. W. Choo, T. J.-J. Li, S. T. Perrault, and T. W. Malone, "A taxonomy for human-Ilm interaction modes: An initial exploration," in Extended Abstracts of the CHI Conference on Human Factors in Computing Systems, 2024, pp. 1-11.


<!-- Meanless: JOURNAL OF LATEX CLASS FILES, DECEMBER 2024<br>100 -->

[595] J. M. Bradshaw, P. J. Feltovich, and M. Johnson, "Human-agent interaction," in The handbook of human-machine interaction. CRC Press, 2017, pp. 283-300.

[596] X. Feng, Z.-Y. Chen, Y. Qin, Y. Lin, X. Chen, Z. Liu, and J.-R. Wen, "Large language model-based human-agent collaboration for complex task solving," arXiv preprint arXiv:2402.12914, 2024.

[597] A. Amayuelas, L. Pan, W. Chen, and W. Wang, "Knowledge of knowledge: Exploring known-unknowns uncertainty with large language models," arXiv preprint arXiv:2305.13712, 2023.

[598] C. Chen, Z. Zhang, I. Khalilov, B. Guo, S. A. Gebreegziabher, Y. Ye, Z. Xiao, Y. Yao, T. Li, and T. J.-J. Li, "Toward a human-centered evaluation framework for trustworthy llm-powered gui agents," arXiv preprint arXiv:2504.17934, 2025.

[599] C. Y. Kim, C. P. Lee, and B. Mutlu, "Understanding large-language model (Ilm)-powered human-robot interaction," in Proceedings of the 2024 ACM/IEEE International Conference on Human-Robot Interaction, 2024, pp. 371-380.

[600] Y. Lu, Y. Yang, Q. Zhao, C. Zhang, and T. J.-J. Li, "Ai assistance for ux: A literature review through human-centered ai," arXiv preprint arXiv:2402.06089, 2024.

[601] J. Wester, T. Schrills, H. Pohl, and N. van Berkel, ""as an ai language model, i cannot": Investigating llm denials of user requests," in Proceedings of the CHI Conference on Human Factors in Computing Systems, 2024, pp. 1-14.

[602] J. Wang, W. Ma, P. Sun, M. Zhang, and J.-Y. Nie, "Understanding user experience in large language model interactions," arXiv preprint arXiv:2401.08329, 2024.

[603] E. Cambria, L. Malandri, F. Mercorio, N. Nobani, and A. Seveso, "XAI meets Ilms: A survey of the relation between explainable ai and large language models," 2024. [Online]. Available: https://arxiv.org/abs/2407.15248

[604] X. Wu, H. Zhao, Y. Zhu, Y. Shi, F. Yang, T. Liu, X. Zhai, W. Yao, J. Li, M. Du et al., "Usable xai: 10 strategies towards exploiting explainability in the Ilm era," arXiv preprint arXiv:2403.08946, 2024.

[605] H. Cai, Y. Li, W. Wang, F. Zhu, X. Shen, W. Li, and T.-S. Chua, "Large language models empowered personalized web agents," arXiv preprint arXiv:2410.17236, 2024.

[606] H. Li, C. Yang, A. Zhang, Y. Deng, X. Wang, and T.-S. Chua, "Hello again! Ilm-powered personalized agent for long-term dialogue," arXiv preprint arXiv:2406.05925, 2024.

[607] H. Li, H. Jiang, T. Zhang, Z. Yu, A. Yin, H. Cheng, S. Fu, Y. Zhang, and W. He, "Traineragent: Customizable and efficient model training through llm-powered multi-agent system," arXiv preprint arXiv:2311.06622, 2023.

[608] Z. Tan and M. Jiang, "User modeling in the era of large language models: Current research and future directions," arXiv preprint arXiv:2312.11518, 2023.

[609] G. Gao, A. Taymanov, E. Salinas, P. Mineiro, and D. Misra, "Aligning Ilm agents by learning latent preference from user edits," arXiv preprint arXiv:2404.15269, 2024.

[610] T. Kaufmann, P. Weng, V. Bengs, and E. Hüllermeier, "A survey of reinforcement learning from human feedback," arXiv preprint arXiv:2312.14925, 2023.

[611] S. Kim, H. Kang, S. Choi, D. Kim, M. Yang, and C. Park, "Large language models meet collaborative filtering: An efficient all-round Ilm-based recommender system," in Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, 2024, pp. 1395-1406.

[612] W. Talukdar and A. Biswas, "Improving large language model (Ilm) fidelity through context-aware grounding: A systematic approach to reliability and veracity," arXiv preprint arXiv:2408.04023, 2024.

[613] X. Xiao and Y. Tao, "Personalized privacy preservation," in Proceedings of the 2006 ACM SIGMOD international conference on Management of data, 2006, pp. 229-240.

[614] N. Hojo, K. Shinoda, Y. Yamazaki, K. Suzuki, H. Sugiyama, K. Nishida, and K. Saito, "Generativegui: Dynamic gui generation leveraging llms for enhanced user interaction on chat interfaces," in Proceedings of the Extended Abstracts of the CHI Conference on Human Factors in Computing Systems, 2025, pp. 1-9.

[615] I. H. Sarker, "LIm potentiality and awareness: a position paper from the perspective of trustworthy and responsible ai modeling," Discover Artificial Intelligence, vol. 4, no. 1, p. 40, 2024.

[616] A. Biswas and W. Talukdar, "Guardrails for trust, safety, and ethical development and deployment of large language models (llm)," Journal of Science & Technology, vol. 4, no. 6, pp. 55-82, 2023.

[617] Y. Li, M. Du, R. Song, X. Wang, and Y. Wang, "A survey on fairness in large language models," arXiv preprint arXiv:2308.10149, 2023.

[618] Z. Zhang, E. Schoop, J. Nichols, A. Mahajan, and A. Swearngin, "From interaction to impact: Towards safer ai agent through understanding and evaluating mobile ui operation impacts," in Proceedings of the 30th International Conference on Intelligent User Interfaces, 2025, pp. 727-744.

[619] E. Ferrara, "Should chatgpt be biased? challenges and risks of bias in large language models," arXiv preprint arXiv:2304.03738, 2023.

[620] Y. Yu, Y. Zhuang, J. Zhang, Y. Meng, A. J. Ratner, R. Krishna, J. Shen, and C. Zhang, "Large language model as attributed training data generator: A tale of diversity and bias," Advances in Neural Information Processing Systems, vol. 36, 2024.

[621] A. Piñeiro-Martín, C. García-Mateo, L. Docío-Fernández, and M. D. C. Lopez-Perez, "Ethical challenges in the development of virtual assistants powered by large language models," Electronics, vol. 12, no. 14, p. 3170, 2023.

[622] B. Zheng, Z. Liu, S. Salisbury, Z. Du, X. Huang, Q. Zheng, L. Davis, M. Lin, X. Jin, H. Sun et al., "Agentmonitor: Towards a generalist guardrail for web agent."

[623] C.-M. Chan, J. Yu, W. Chen, C. Jiang, X. Liu, W. Shi, Z. Liu, W. Xue, and Y. Guo, "Agentmonitor: A plug-and-play framework for predictive and secure multi-agent systems," arXiv preprint arXiv:2408.14972, 2024.

[624] L. Lin, L. Wang, J. Guo, and K.-F. Wong, "Investigating bias in Ilm-based bias detection: Disparities between Ilms and human perception," arXiv preprint arXiv:2403.14896, 2024.

[625] Y. Zhang, T. Yu, and D. Yang, "Attacking vision-language computer agents via pop-ups," 2024. [Online]. Available: https://arxiv.org/abs/2411.02391

[626] S. Shekkizhar and R. Cosentino, "Agi is coming... right after ai learns to play wordle," 2025. [Online]. Available: https://arxiv.org/abs/2504.15434

[627] R. Grosse, J. Bae, C. Anil, N. Elhage, A. Tamkin, A. Tajdini, B. Steiner, D. Li, E. Durmus, E. Perez et al., "Studying large language model generalization with influence functions," arXiv preprint arXiv:2308.03296, 2023.

[628] X. Zhang, J. Li, W. Chu, J. Hai, R. Xu, Y. Yang, S. Guan, J. Xu, and P. Cui, "On the out-of-distribution generalization of multimodal large language models," arXiv preprint arXiv:2402.06599, 2024.

[629] E. Li and J. Waldo, "Websuite: Systematically evaluating why web agents fail," arXiv preprint arXiv:2406.01623, 2024.

[630] Y. Song, W. Xiong, X. Zhao, D. Zhu, W. Wu, K. Wang, C. Li, W. Peng, and S. Li, "Agentbank: Towards generalized Ilm agents via fine-tuning on 50000+ interaction trajectories," arXiv preprint arXiv:2410.07706, 2024.

[631] K. Weiss, T. M. Khoshgoftaar, and D. Wang, "A survey of transfer learning," Journal of Big data, vol. 3, pp. 1-40, 2016.

[632] Y. Chen, R. Zhong, S. Zha, G. Karypis, and H. He, "Meta-learning via language model in-context tuning," arXiv preprint arXiv:2110.07814, 2021.

[633] Y. Zhu, S. Qiao, Y. Ou, S. Deng, N. Zhang, S. Lyu, Y. Shen, L. Liang, J. Gu, and H. Chen, "Knowagent: Knowledge-augmented planning for Ilm-based agents," arXiv preprint arXiv:2403.03101, 2024.

[634] Y. Guan, D. Wang, Y. Wang, H. Wang, R. Sun, C. Zhuang, J. Gu, and Z. Chu, "Explainable behavior cloning: Teaching large language model agents through learning by demonstration," arXiv preprint arXiv:2410.22916, 2024.

[635] C.-Y. Hsieh, S.-A. Chen, C.-L. Li, Y. Fujii, A. Ratner, C.-Y. Lee, R. Krishna, and T. Pfister, "Tool documentation enables zero-shot tool-usage with large language models," arXiv preprint arXiv:2308.00675, 2023.

[636] T. Kagaya, T. J. Yuan, Y. Lou, J. Karlekar, S. Pranata, A. Kinose, K. Oguri, F. Wick, and Y. You, "Rap: Retrieval-augmented planning with contextual memory for multimodal Ilm agents," arXiv preprint arXiv:2402.03610, 2024.