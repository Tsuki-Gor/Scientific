<!-- Meanless: indows in Ultra- sum of the list -->

# LLM-Powered GUI Agents in Phone Automation: Surveying Progress and Prospects

Guangyi Liu ${}^{1, \dagger  }$ , Pengxiang Zhao ${}^{1, \dagger  }$ , Liang Liu ${}^{2, \dagger  , \ddagger  }$ , Yaxuan Guo ${}^{2}$ , Han Xiao ${}^{3}$ , Weifeng Lin ${}^{3}$ Yuxiang Chai ${}^{3}$ , Yue Han ${}^{1}$ , Shuai Ren ${}^{2}$ ,Hao Wang ${}^{1}$ , Xiaoyu Liang ${}^{1}$ , Wenhao Wang ${}^{1}$

Tianze ${\mathrm{{Wu}}}^{1}$ ,Linghao ${\mathrm{{Li}}}^{1}$ ,Hao ${\mathrm{{Wang}}}^{2}$ ,Guanjing Xiong ${}^{2}$ ,Yong Liu ${}^{1,}{}^{\left\lbrack  \infty \right\rbrack  }$ ,Hongsheng ${\mathrm{{Li}}}^{3,} \; {}^{1}$ Zhejiang University ${}^{2}$ vivo Al Lab ${}^{3}$ CUHK MMLab

${}^{ \dagger  }$ Equal Contribution, ${}^{ \ddagger  }$ Project Lead, ${}^{ \boxtimes  }$ Corresponding Authors

yongliu@iipc.zju.edu.cn; hsli@ee.cuhk.edu.hk

Project Homepage: github.com/PhoneLLM/Awesome-LLM-Powered-Phone-GUI-Agents

Abstract-With the rapid rise of large language models (LLMs), phone automation has undergone transformative changes. This paper systematically reviews LLM-driven phone GUI agents, highlighting their evolution from script-based automation to intelligent, adaptive systems. We first contextualize key challenges, (i) limited generality, (ii) high maintenance overhead, and (iii) weak intent comprehension, and show how LLMs address these issues through advanced language understanding, multimodal perception, and robust decision-making. We then propose a taxonomy covering fundamental agent frameworks (single-agent, multi-agent, plan-then-act), modeling approaches (prompt engineering, training-based), and essential datasets and benchmarks. Furthermore, we detail task-specific architectures, supervised fine-tuning, and reinforcement learning strategies that bridge user intent and GUI operations. Finally, we discuss open challenges such as dataset diversity, on-device deployment efficiency, user-centric adaptation, and security concerns, offering forward-looking insights into this rapidly evolving field. By providing a structured overview and identifying pressing research gaps, this paper serves as a definitive reference for researchers and practitioners seeking to harness LLMs in designing scalable, user-friendly phone GUI agents.

Index Terms-Large Language Models, GUI Agents, Phone Automation, Mobile Interfaces, Natural Language Processing

## 1 INTRODUCTION

THE core of phone GUI automation involves programmatically simulating human interactions with mobile interfaces to accomplish complex tasks. This technology has wide applications in testing and shortcut creation, enhancing efficiency and reducing manual effort [1], [2], [3], [4], [5]. Traditional approaches rely on predefined scripts and templates which, while functional, lack flexibility when confronting variable interfaces and dynamic environments [6], [7], [8], [9], [10].

In computer science, an agent perceives its environment through sensors and acts via actuators to achieve goals [11], [12], [13], [14], [15]. These range from simple scripts to complex systems capable of learning and adaptation [13], [14], [16]. Traditional phone automation agents are constrained by static scripts and limited adaptability, making them ill-suited for modern mobile interfaces' dynamic nature.

Building intelligent autonomous agents with planning, decision-making, and execution capabilities remains a long-term AI goal [17]. As technologies advanced, agents evolved from traditional forms [18], [19], [20] to AI agents [21], [22], [23] incorporating machine learning and probabilistic decision-making. However, these still struggle with complex instructions [24], [25] and dynamic environments [26], [27].

With the rapid development of Large Language Models (LLMs) like the GPT series [28], [29], [30], [31] and specialized models such as Fuyu-8B [32], LLM-based agents have demonstrated powerful capabilities across numerous domains [33], [34], [35], [36], [37], [38], [39], [40], [41], [42], [43]. As Figure 1 illustrates, conversational LLMs primarily focus on language understanding and generation, while LLM-based agents extend these capabilities by integrating perception and action components. This integration enables interaction with external environments through multimodal inputs and operational outputs [33], [34], [41], bridging language understanding and real-world interactions [11], [12], [44], [45].

Applying LLM-based agents to phone automation has created a new paradigm, making mobile interface operations more intelligent [46], [47], [48], [49]. LLM-powered phone GUI agents are intelligent systems that leverage large language models to understand, plan, and execute tasks on mobile devices by integrating natural language processing, multimodal perception, and action execution capabilities. These agents can recognize interfaces, understand instructions, perceive changes in real time, and respond dynamically. Unlike script-based automation, they can autonomously plan complex sequences through multimodal processing of instructions and interface information. Their adaptability and flexibility improve user experience through intent understanding, planning, and automated task execution, enhancing efficiency across scenarios from app testing to complex operations like configuring settings [50], navigating maps [51], [52], and shopping [48].

Clarifying the development trajectory of phone GUI agents is crucial. On one hand, with the support of large language models [28], [29], [30], [31], phone GUI agents can significantly enhance the efficiency of phone automation scenarios, making operations more intelligent and no longer limited to coding fixed operation paths. This enhancement not only optimizes phone automation processes but also expands the application scope of automation. On the other hand, phone GUI agents can understand and execute complex natural language instructions, transforming human intentions into specific operations such as automatically scheduling appointments, booking restaurants, summoning transportation, and even achieving functionalities similar to autonomous driving in advanced automation. These capabilities demonstrate the potential of phone GUI agents in executing complex tasks, providing convenience to users and laying practical foundations for AI development.


<!-- Meanless: 2 -->

<!-- Media -->

<!-- figureText: What are the best coffee beans?<br>I want a latte delivered to my office.<br>Perception<br>Ethiopian Yirgacheffe beans are often considered among the best...<br>Action: "Tap the Starbucks icon"<br>conversational LLMs<br>Phone GUI agents -->

<img src="https://cdn.noedgeai.com/bo_d41gln3ef24c73d3u2c0_1.jpg?x=139&y=126&w=741&h=348&r=0"/>

Fig. 1: Comparison between conversational LLMs and phone GUI agents. While a conversational LLM can understand queries and provide informative responses (e.g., recommending coffee beans), a Phone GUI agent can go beyond text generation to perceive the device's interface, decide on an appropriate action (like tapping an app icon), and execute it in the real environment, thus enabling tasks like ordering a latte directly on the user's phone.

<!-- Media -->

With the increasing research on large language models in phone automation [50], [51], [52], [53], [54], [55], [56], the research community's attention to this field has grown rapidly. However, there is still a lack of dedicated systematic surveys in this area, especially comprehensive explorations of phone automation from the perspective of large language models. Given the importance of phone GUI agents, the purpose of this paper is to fill this gap by systematically summarizing current research achievements, reviewing relevant literature, analyzing the application status of large language models in phone automation, and pointing out directions for future research.

To provide a comprehensive overview of the current state and future prospects of LLM-Powered GUI Agents in Phone Automation, we present a taxonomy that categorizes the field into three main areas: Frameworks of LLM-powered phone GUI agents, Large Language Models for Phone Automation, and Datasets and Evaluation Methods Figure 2. This taxonomy highlights the diversity and complexity of the field, as well as the interdisciplinary nature of the research involved.

Unlike previous literature reviews, which primarily focus on traditional phone automated testing methods, most existing surveys emphasize manual scripting or rule-based automation approaches without leveraging LLMs [6], [7], [8], [9], [10]. These traditional methods face significant challenges in coping with dynamic changes, complex user interfaces, and the scalability required for modern applications. Although recent surveys have explored broader areas of multimodal agents and foundation models for GUI automation, such as Foundations and Recent Trends in Multimodal Mobile Agents: A Survey [157], GUI Agents with Foundation Models: A Comprehensive Survey [158], and Large Language Model-Brained GUI Agents: A Survey [159], these works primarily cover general GUI-based automation and multimodal applications.

However, a dedicated and focused survey on the role of large language models in phone GUI automation remains absent in the existing literature. This paper addresses the above-mentioned gap by systematically reviewing the latest developments, challenges, and opportunities in LLM-powered phone GUI agents, thereby offering a more targeted exploration of this emerging domain. Our main contributions can be summarized as follows:

- A Comprehensive and Systematic Survey of LLM-Powered Phone GUI Agents. We provide an in-depth and structured overview of recent literature on LLM-powered phone automation, examining its developmental trajectory, core technologies, and real-world application scenarios. By comparing LLM-driven methods to traditional phone automation approaches, this survey clarifies how large models transform GUI-based tasks and enable more intelligent, adaptive interaction paradigms.

- Methodological Framework from Multiple Perspectives. Leveraging insights from existing studies, we propose a unified methodology for designing LLM-driven phone GUI agents. This encompasses framework design (e.g., single-agent vs. multi-agent vs. plan-then-act frameworks), LLM model selection and training (prompt engineering vs. training-based methods), data collection and preparation strategies (GUI-specific datasets and annotations), and evaluation protocols (benchmarks and metrics). Our systematic taxonomy and method-oriented discussion serve as practical guidelines for both academic and industrial practitioners.

- In-Depth Analysis of Why LLMs Empower Phone Automation. We delve into the fundamental reasons behind LLMs' capacity to enhance phone automation. By detailing their advancements in natural language comprehension, multimodal grounding, reasoning, and decision-making, we illustrate how LLMs bridge the gap between user intent and GUI actions. This analysis elucidates the critical role of large models in tackling issues of scalability, adaptability, and human-like interaction in real-world mobile environment.

- Insights into Latest Developments, Datasets, and Benchmarks. We introduce and evaluate the most recent progress in the field, highlighting innovative datasets that capture the complexity of modern GUIs and benchmarks that allow reliable performance assessment. These resources form the backbone of LLM-based phone automation, enabling systematic training, fair evaluation, and transparent comparisons across different agent designs.


<!-- Media -->

<!-- figureText: Single-Agent (§3.1 - §3.3)<br>e.g. DroidBot-GPT [53], Enabling Conversational [57], AutoDroid [50], LLMPA [58], TOL Agent [59], MM-Navigator [60], MobileGPT [61], CogAgent [46], OmniParser [56], GUI Narrator [62], MobileVLM [63], AppAgent [48], AppAgent v2 [64], AppAgentX [65], Auto-GUI [66], ScreenAI [67], Mobile-Agent-v [68], OS-Kairos [69], GUI-Xplore [70], CoCo-agent [71]<br>PromptRPA [75], CHOP [76], Agent S2 [77], Ask-before-Plan [78]<br>Frameworks (§3)<br>Role-Coordinated (§3.4.1)<br>e.g. MMAC-Copilot [72], Cradle [73], Mobile-Agent-v2 [52], Mobile-Agent-E [74],<br>Multi-Agent (§3.4)<br>Scenario-Based (§3.4.2)<br>e.g. MobileExperts [55], SteP [79]<br>Plan-Then-Act (§3.5)<br>e.g. SeeAct [47], UGround [80], LiMAC [81], ClickAgent [82], Ponder & Press [83]<br>Text-Based Prompt (§4.1.1)<br>e.g. MobileGPT [61], AutoDroid [50], DroidBot-GPT [53], Enabling conversational [57], PromptRPA [75], AXNav [84]<br>Prompt Engineering<br>Multimodal Prompt (§4.1.2)<br>e.g. Mobile-Agent [51], Mobile-Agent-v2 [52], OmniParser [56], VisionDroid [54], AppAgent [48], MM-Navigator [60], MobileExperts [55], VisionTasker [85] AppAgent v2 [64], GUI Narrator [62], ReuseDroid [86], VLM-Fuzzer [87], Mobile-Agent-E [74]<br>LLM-Powered GUI Agents in Phone Automation<br>General-Purpose<br>e.g. Auto-GUI [66], CogAgent [46], ScreenAI [67], CoCo-Agent [71], MobileFlow [88], ShowUI [89], Aguvis [90], ViMo [91], UI-TARS [92]<br>Models (§4)<br>Task-Specific Model Architectures (§4.2.1)<br>UI Grounding<br>e.g. MUG [93], LVG [94], UI-Hawk [95], Aria-UI [96], OS-Atlas [97], MP-GUI [98], Smoothing Grounding [99], GUI-Bee [100]<br>Phone UI-Specific<br>UI Referring<br>e.g. Ferret-UI [101], UI-Hawk [95], MP-GUI [98] Ferret-UI 2 [102], Textual Foresight [103]<br>Screen Question Answering<br>e.g. ScreenAI [67], WebVLN [104], MP-GUI [98] UI-Hawk [95]<br>Training-based Methods (§4.2)<br>Supervised Fine-Tuning (§4.2.2)<br>e.g. SeeClick [105], InfiGUIAgent [106], GUICourse [107], Agent-R [108], GUI Odyssey [109], TinyClick [110], MobileAgent [111], ReALM [112], AppVLM [113] V-Droid [114], IconDesc [115]<br>Phone Agents<br>e.g. DigiRL [116], DistRL [117], AutoGLM [118], Digi-q [119], ReachAgent [120], VSC-RL [121], Ui-r1 [122], GUI-R1 [123]<br>Reinforcement Learning (§4.2.3)<br>Web Agents<br>e.g. ETO [124], Agent Q [125], AutoWebGLM [126], GLAINTEL [127]<br>PC Agents<br>e.g. ScreenAgent [128], AssistGUI [129]<br>Datasets (§5.1)<br>e.g. Rico [130], RICO Semantics [131], PixelHelp [132], MoTIF [133], UIBert [134], Meta-GUI [135], UGIF [136], AITW [137], AITZ [138], GUI Odyssey [109], GUI-WORLD [139] AndroidControl [140], AMEX [141], MobileViews [<br>Datasets and Benchmarks (§5)<br>Benchmarks (§5.2)<br>e.g. AutoDroid [50], MobileEnv [143], AndroidArena [144], LlamaTouch [145], B-MoCA [146], AndroidWorld [147 - AUITestAgent [148], AgentStudio [149], AndroidLab [150], MobileAgentBench [151], VisualAgentBench [152], FedMABench [153], AutoEval [154], LearnAct [155], A3 [156] -->

<img src="https://cdn.noedgeai.com/bo_d41gln3ef24c73d3u2c0_2.jpg?x=133&y=109&w=1533&h=1330&r=0"/>

Fig. 2: A comprehensive taxonomy of LLM-powered phone GUI agents in phone automation. Note that only a selection of representative works is included in this categorization.

<!-- Media -->

- Identification of Key Challenges and Novel Perspectives for Future Research. Beyond discussing mainstream hurdles (e.g., dataset coverage, on-device constraints, reliability), we propose forward-looking viewpoints on user-centric adaptations, security and privacy considerations, long-horizon planning, and multi-agent coordination. These novel perspectives shed light on how researchers and developers might advance the current state of the art toward more robust, secure, and personalized phone GUI agents.

By addressing these aspects, our survey not only provides an up-to-date map of LLM-powered phone GUI automation but also offers a clear roadmap for future exploration. We hope this work will guide researchers in identifying pressing open problems and inform practitioners about promising directions to harness LLMs in designing efficient, adaptive, and user-friendly phone GUI agents.

## 2 DEVELOPMENT OF PHONE AUTOMATION

The evolution of phone automation has been marked by significant technological advancements [160], particularly with the emergence of LLMs [28], [29], [30], [31]. This section explores the historical development of phone automation, the challenges faced by traditional methods, and how LLMs have revolutionized the field.

### 2.1 Phone Automation Before the LLM Era

Before the advent of LLMs, phone automation was predominantly achieved through traditional technical methods [1], [160], [161], [162], [163], [164]. This subsection delves into the primary areas of research and application during that period, including automation testing, shortcuts, and Robotic Process Automation (RPA), highlighting their methodologies and limitations.


<!-- Meanless: 4 -->

#### 2.1.1 Automation Testing

Phone applications (apps) have become extremely popular, with approximately 1.68 million apps in the Google Play Store ${}^{1}$ . The increasing complexity of apps [165] has raised significant concerns about app quality. Moreover, due to rapid release cycles and limited human resources, developers find it challenging to manually construct test cases. Therefore, various automated phone app testing techniques have been developed and applied, making phone automation testing the main application of phone automation before the era of large models [160], [161], [163], [166]. Test cases for phone apps are typically represented by a sequence of GUI events [167] to simulate user interactions with the app. The goal of automated test generators is to produce such event sequences to achieve high code coverage or detect bugs [164].

In the development history of phone automation testing, we have witnessed several key breakthroughs and advancements. Initially, random testing (e.g., Monkey Testing [168]) was used as a simple and fundamental testing method, detecting application stability and robustness by randomly generating user actions. Although this method could cover a wide range of operational scenarios, its testing process lacked focus and was difficult to reproduce and pinpoint specific issues [160].

Subsequently, model-based testing [1], [162], [169] became a more systematic testing approach. It establishes a user interface model of the application, using predefined states and transition rules to generate test cases. This method improved testing coverage and efficiency, but the construction and maintenance of the model required substantial manual involvement, and updating the model became a challenge for highly dynamic applications.

With the development of machine learning techniques, learning-based testing methods began to emerge [2], [3], [4], [5]. These methods generate test cases by analyzing historical data to learn user behavior patterns. For example, Humanoid [4] uses deep learning to mimic human tester interaction behavior and uses the learned model to guide test generation like a human tester. However, this method relies on human-generated datasets to train the model and needs to combine the model with a set of predefined rules to guide testing.

Recently, reinforcement learning [170] has shown great potential in the field of automated testing. DinoDroid [164] is an example that uses Deep Q-Network (DQN) [171] to automate testing of Android applications. By learning behavior models of existing applications, it automatically explores and generates test cases, not only improving code coverage but also enhancing bug detection capabilities. Deep reinforcement learning methods can handle more complex state spaces and make more intelligent decisions but also face challenges such as high training costs and poor model generalization capabilities [172].

#### 2.1.2 Shortcuts

Shortcuts on mobile devices refer to predefined rules or trigger conditions that enable users to execute a series of actions automatically [173], [174], [175]. These shortcuts are designed to streamline interaction by reducing repetitive manual input. For instance, the Tasker app on the Android platform ${}^{2}$ and the Shortcuts feature on iOS ${}^{3}$ allow users to automate tasks like turning on Wi-Fi, sending text messages, or launching apps under specific conditions such as time, location, or events. These implementations leverage simple IF-THEN and manually-designed logic but are inherently limited in scope and flexibility.

#### 2.1.3 Robotic Process Automation

Robotic Process Automation(RPA) applications on phone devices aim to simulate human users performing repetitive tasks across applications [176]. Phone RPA tools generate repeatable automation processes by recording user action sequences. These tools are used in enterprise environment to automate tasks such as data entry and information gathering, reducing human errors and improving efficiency, but they struggle with dynamic interfaces and require frequent script updates [177], [178].

### 2.2 Challenges of Traditional Methods

Despite the advancements made, traditional phone automation methods faced significant challenges that hindered further development. This subsection analyzes these challenges, including lack of generality and flexibility, high maintenance costs, difficulty in understanding complex user intentions, and insufficient intelligent perception, highlighting the need for new approaches.

#### 2.2.1 Limited Generality

Traditional automation methods are often tailored to specific applications and interfaces, lacking adaptability to different apps and dynamic user environment [179], [180], [181], [182]. For example, automation scripts designed for a specific app may not function correctly if the app updates its interface or if the user switches to a different app with similar functionality. This inflexibility makes it difficult to extend automation across various usage scenarios without significant manual reconfiguration.

These methods typically follow predefined sequences of actions and cannot adjust their operations based on changing contexts or user preferences. For instance, if a user wants an automation to send a customized message to contacts who have birthdays on a particular day, traditional methods struggle because they cannot dynamically access and interpret data from the contacts app, calendar, and messaging app simultaneously. Similarly, automating tasks that require conditional logic-such as playing different music genres based on the time of day or weather conditions-poses a challenge for traditional automation tools, as they lack the ability to integrate real-time data and make intelligent decisions accordingly [183], [184].

#### 2.2.2 High Maintenance Costs

Writing and maintaining automation scripts require professional knowledge and are time-consuming and labor-intensive [185], [186], [187], [188], [189]. Taking RPA as an example, as applications continually update and iterate, scripts need frequent modifications. When an application's interface layout changes or functions are updated, RPA scripts originally written for the old version may not work properly, requiring professionals to spend considerable time and effort readjusting and optimizing the scripts [190], [191], [192].

---

<!-- Footnote -->

2. https://play.google.com.

3. https://support.apple.com.

1. https://www.statista.com.

<!-- Footnote -->

---


<!-- Meanless: 5 -->

The high entry barrier also limits the popularity of some automation features [193], [194]. For example, Apple's Shortcuts ${}^{4}$ can combine complex operations, such as starting an Apple Watch fitness workout, recording training data, and sending statistical data to the user's email after the workout. However, setting up such a complex shortcut often requires the user to perform a series of complicated operations on the phone following fixed rules. This is challenging for ordinary users, leading many to abandon usage due to the complexity of manual script writing.

#### 2.2.3 Poor Intent Comprehension

Rule-based and script-based systems can only execute predefined tasks or engage in simple natural language interactions [195], [196]. Simple instructions like "open the browser" can be handled using traditional natural language processing algorithms, but complex instructions like "open the browser, go to Amazon, and purchase a product" cannot be completed. These traditional systems are based on fixed rules and lack in-depth understanding and parsing capabilities for complex natural language [197], [198], [199].

They require users to manually write scripts to interact with the phone, greatly limiting the application of intelligent assistants that can understand complex human instructions. For example, when a user wants to check flight information for a specific time and book a ticket, traditional systems cannot accurately understand the user's intent and automatically complete the series of related operations, necessitating manual script writing with multiple steps, which is cumbersome and requires high technical skills.

#### 2.2.4 Weak Screen GUI Perception

Different applications present a wide variety of GUI elements, making it challenging for traditional methods like RPA to accurately recognize and interact with diverse controls [200], [201], [202], [203]. Traditional automation often relies on fixed sequences of actions targeting specific controls or input fields, exhibiting Weak Screen GUI Perception that limits their ability to adapt to variations in interface layouts and component types. For example, in an e-commerce app, the product details page may include dynamic content like carousels, embedded videos, or interactive size selection menus, which differ significantly from the simpler layout of a search results page. Traditional methods may fail to accurately identify and interact with the "Add to Cart" button or select product options, leading to unsuccessful automation of purchasing tasks.

Moreover, traditional automation struggles with understanding complex screen information such as dynamic content updates, pop-up notifications, or context-sensitive menus that require adaptive interaction strategies. Without the ability to interpret visual cues like icons, images, or contextual hints, these methods cannot handle tasks that involve navigating through multi-layered interfaces or responding to real-time changes. For instance, automating the process of booking a flight may involve selecting dates from a calendar widget, choosing seats from an interactive seat map, or handling security prompts-all of which require sophisticated perception and interpretation of the interface [145].

In phone automation, many apps do not provide open API interfaces, forcing solutions to rely directly on the GUI for triggering actions and retrieving information. Even when tools are used to parse the Android UI [204], non-standard controls often prevent accurate JSON parsing, further complicating automated testing and interaction. Additionally, because the GUI is a universal and consistent interface across apps regardless of their internal design, it naturally becomes the central focus of phone automation methods.

These limitations significantly impede the widespread application and deep development of traditional phone automation technologies. Without intelligent perception capabilities, automation cannot adapt to the complexities of modern app interfaces, which are increasingly dynamic and rich in interactive elements. This underscores the urgent need for new methods and technologies that can overcome these bottlenecks and achieve more intelligent, flexible, and efficient phone automation.

### 2.3 LLMs Boost Phone Automation

The advent of LLMs has marked a significant shift in the landscape of phone automation, enabling more dynamic, context-aware, and sophisticated interactions with mobile devices. As illustrated in Figure 3, the research on LLM-powered phone GUI agents has progressed through pivotal milestones, where models become increasingly adept at interpreting multimodal data, reasoning about user intents, and autonomously executing complex tasks. This section clarifies how LLMs address traditional limitations and examines why scaling laws can further propel large models in phone automation. As will be detailed in $§4$ and $§5$ ,LLM-based solutions for phone automation generally follow two routes: (1) Prompt Engineering, where pre-trained models are guided by carefully devised prompts, and (2) Training-Based Methods, where LLMs undergo additional optimization on GUI-focused datasets. The following subsections illustrate how LLMs mitigate the core challenges of traditional phone automation-ranging from contextual semantic understanding and GUI perception to reasoning and decision making-and briefly highlight the role of scaling laws in enhancing these capabilities.

Scaling Laws in LLM-Based Phone Automation. Scaling laws-originally observed in general-purpose LLMs, where increasing model capacity and training data yields emergent capabilities [205], [206], [207]-have similarly begun to manifest in phone GUI automation. As datasets enlarge and encompass more diverse apps, usage scenarios, and user behaviors, recent findings [105], [107], [109], [110] show consistent gains in step-by-step automation tasks such as clicking buttons or entering text. This data scaling not only captures broader interface layouts and device contexts but also reveals latent "emergent" competencies, allowing LLMs to handle more abstract, multi-step instructions. Empirical evidence from in-domain scenarios [140] further underscores how expanded coverage of phone apps and user patterns systematically refines automation accuracy. In essence, as model sizes and data complexity grow, phone GUI agents exploit these scaling laws to bridge the gap between user intent and real-world GUI interactions with increasing efficiency and sophistication.

---

<!-- Footnote -->

4. https://support.apple.com.

<!-- Footnote -->

---


<!-- Meanless: 6 -->

<!-- Media -->

<!-- figureText: Datasets<br>Benchmarks<br>S<br>GUI-R1<br>Agent S2<br>HUAWEI<br>OS-Kairos<br>GUIPivot<br>GUI-Xplore<br>AutoEval<br>FedMABench<br>ViMo<br>VLM-Fuzz<br>ReuseDroid<br>LearnAct<br>Apr.<br>ADA<br>HUAWEI<br>Mar.<br>V-Droid<br>AppAgentX<br>UI-R1<br>MP-GUI<br>CHOP<br>HUAWEI<br>VEM<br>VSC-RL<br>amazon<br>MI<br>Agent-R<br>InfiGUIAgent<br>HUAWEI<br>Digi-Q<br>AppVLM<br>Mobile-Agent-V<br>ReachAgent<br>Feb.<br>Jan<br>Mobile-Agent-E<br>UI-TARS<br>A3<br>GUI-Bee<br>热<br>2025<br>Aria-UI<br>Ponder & Press<br>Aguvis<br>Dec.<br>HUAWEI<br>LiMAC<br>DistRL<br>TinyClick<br>Nov.<br>AutoGLM<br>ShowUI<br>AndroidLab<br>上海人工智能实验室<br>AgentStudio<br>OS-Atlas<br>Ferret-UI 2<br>UGround<br>Oct.<br>Sep.<br>IconDesc<br>MobileViews<br>Tencent<br>HUAWEI<br>VisualAgentBench<br>AppAgent v2<br>OmniParser<br>UI-hawk<br>Aug.<br>MI<br>BOSTON university<br>Mobile-Bench<br>AMEX<br>AndroidControl<br>AUITestAgent<br>MobileAgentBench<br>Mobile-Agent-v2<br>Textual Foresight<br>Lenovo<br>Jul<br>MobileExperts<br>Security matrix<br>MobileFlow<br>VisionDroid<br>VisionTasker<br>GUI Narrator<br>GUICourse<br>GUI odyssey<br>DigiRL<br>LVG<br>Jun<br>Google<br>B-MoCA<br>LlamaTouch<br>May<br>AXNav<br>AndroidWorld<br>Ferret-UI<br>Octopus v2<br>ReALM<br>PromptRPA<br>Apr.<br>HUAWEI<br>Google<br>Mar<br>AITZ<br>MobileEnv<br>AndroidArena<br>CoCo-agent<br>ScreenAI<br>Feb.<br>Google<br>Mobile-Agent<br>MobileAgent<br>SeeClick<br>UINav<br>AITW<br>WebVLN-Net<br>(2)<br>Tencent<br>2024<br>MobileGPT<br>VisionTasker<br>AppAgent<br>CogAgent<br>Dec.<br>amazon<br>Nov.<br>MM-Navigator<br>AutoDroid<br>AutoGUI<br>Sep.<br>Google<br>Apr<br>DroidBot-GPT<br>Enabling Conversational<br>2023 -->

<img src="https://cdn.noedgeai.com/bo_d41gln3ef24c73d3u2c0_5.jpg?x=186&y=170&w=1452&h=1665&r=0"/>

Fig. 3: Milestones in the development of LLM-powered phone GUI agents. This figure divides advancements into four primary parts: Prompt Engineering, Training-Based Methods, Datasets and Benchmarks. Prompt Engineering leverages pre-trained LLMs by strategically crafting input prompts, as detailed in §4.1, to perform specific tasks without modifying model parameters. In contrast, Training-Based Methods, discussed in §4.2, involve adapting LLMs via supervised fine-tuning or reinforcement learning on GUI-specific data, thereby enhancing their ability to understand and interact with mobile UIs.

<!-- Media -->


<!-- Meanless: 7 -->

Contextual Semantic Understanding. LLMs have transformed natural language processing for phone automation by learning from extensive textual corpora [48], [50], [205], [208], [209], [210]. This training captures intricate linguistic structures and domain knowledge [199], allowing agents to parse multi-step commands and generate context-informed responses. MobileAgent [51], for example, interprets user directives like scheduling appointments or performing transactions with high precision, harnessing the Transformer architecture [208] for efficient encoding of complex prompts. Consequently, phone GUI agents benefit from stronger natural language grounding, bridging user-intent gaps once prevalent in script-based systems.

Screen GUI with Multi-Modal Perception. Screen GUI perception in earlier phone automation systems typically depended on static accessibility trees or rigid GUI element detection, which struggled to adapt to changing app interfaces. Advances in LLMs, supported by large-scale multimodal datasets [211], [212], [213], allow models to unify textual and visual signals in a single representation. Systems like UGround [80], Ferret-UI [101], and UI-Hawk [95] excel at grounding natural language descriptions to on-screen elements, dynamically adjusting as interfaces evolve. Moreover, SeeClick [105] and ScreenAI [67] demonstrate that learning directly from screenshots-rather than purely textual meta-data—can further enhance adaptability. By integrating visual cues with user language, LLM-based agents can respond more flexibly to a wide range of UI designs and interaction scenarios.

Reasoning and Decision Making. LLMs also enable advanced reasoning and decision-making by combining language, visual context, and historical user interactions. Pretraining on broad corpora equips these models with the capacity for complex reasoning [214], [215], multi-step planning [216], [217], and context-aware adaptation [218], [219]. MobileAgent-V2 [52], for instance, introduces a specialized planning agent to track task progress while a decision agent optimizes actions. Auto-GUI [66] applies a multimodal chain-of-action approach that accounts for both previous and forthcoming steps, and SteP [79] uses stacked LLM modules to solve diverse web tasks. Similarly, MobileGPT [61] leverages an app memory system to minimize repeated mistakes and bolster adaptability. Such architectures demonstrate higher success rates in complex phone operations, reflecting a new level of autonomy in orchestrating tasks that previously demanded handcrafted scripts.

Overall, LLMs are transforming phone automation by reinforcing semantic understanding, expanding multimodal perception, and enabling sophisticated decision-making strategies. The scaling laws observed in datasets like An-droidControl [140] reinforce the notion that a larger volume and diversity of demonstrations consistently elevate model accuracy. As these techniques mature, LLM-driven phone GUI agents continue to redefine how users interact with mobile devices, ultimately paving the way for a more seamless and user-centric automation experience.

### 2.4 Emerging Commercial Applications

The integration of LLMs has enabled novel commercial applications that leverage phone automation, offering innovative solutions to real-world challenges. This subsection highlights several prominent cases, presented in chronological order based on their release dates, where LLM-based GUI agents are reshaping user experiences, improving efficiency, and providing personalized services.

Apple Intelligence. On June 11, 2024, Apple introduced its personal intelligent system,Apple Intelligence ${}^{5}$ , seamlessly integrating AI capabilities into iOS, iPadOS, and macOS. It enhances communication, productivity, and focus features through intelligent summarization, priority notifications, and context-aware replies. For instance, Apple Intelligence can summarize long emails, transcribe and interpret call recordings, and generate personalized images or "Genmoji." A key aspect is on-device processing, which ensures user privacy and security. By enabling the system to operate directly on the user's device, Apple Intelligence safeguards personal information while providing an advanced, privacy-preserving phone automation experience.

vivo PhoneGPT. On October 10, 2024, vivo unveiled Origi-nOS 5 ${}^{6}$ , its newest mobile operating system, featuring an AI agent ability named PhoneGPT. By harnessing large language models, PhoneGPT can understand user instructions, preferences, and on-screen information, autonomously engaging in dialogues and detecting GUI states to operate the smart-phone. Notably, it allows users to order coffee or takeout with ease and can even carry out a full phone reservation process at a local restaurant through extended conversations. By integrating the capabilities of large language models with native system states and APIs, PhoneGPT illustrates the great potential of phone GUI agents.

Honor YOYO Agent. Released on October 24, 2024, the Honor YOYO Agent ${}^{7}$ exemplifies an phone automation assistant that adapts to user habits and complex instructions. With just one voice or text command, YOYO can automate multi-step processes-such as comparing prices to secure discounts when shopping, automatically filling out forms, ordering beverages aligned with user preferences, or silencing notifications during online meetings. By learning from user behaviors, YOYO reduces the complexity of human-device interaction, offering a more effortless and intelligent phone experience.

Anthropic Claude Computer Use. On October 22, 2024, Anthropic unveiled the Computer Use feature for its Claude 3.5 Sonnet model8. This feature allows an AI agent to interact with a computer as if a human were operating it, observing screenshots, moving the virtual cursor, clicking buttons, and typing text. Instead of requiring specialized environment adaptations, the AI can "see" the screen and perform actions that humans would, bridging the gap between language-based instructions and direct computer operations. Although initial performance is still far below human proficiency, this represents a paradigm shift in human-computer interaction. By teaching AI to mimic human tool usage, Anthropic reframes the challenge from "tool adaptation for models" to "model adaptation to existing tools." Achieving balanced performance, security, and cost-effectiveness remains an ongoing endeavor.

---

<!-- Footnote -->

5. https://www.apple.com/apple-intelligence/.

6. https://www.vivo.com.cn/originos

7. https://www.honor.com/cn/magic-os/.

8. https://www.anthropic.com/news/3-5-models-and-computer-use

<!-- Footnote -->

---


<!-- Meanless: 8 -->

Zhipu.AI AutoGLM. On October 25, 2024, Zhipu.AI introduced AutoGLM [118], an intelligent agent that simulates human operations on smartphones. With simple text or voice commands, AutoGLM can like and comment on social media posts, purchase products, book train tickets, or order takeout. Its capabilities extend beyond mere API calls-AutoGLM can navigate interfaces, interpret visual cues, and execute tasks that mirror human interaction steps. This approach streamlines daily tasks and demonstrates the versatility and practicality of LLM-driven phone automation in commercial applications.

These emerging commercial applications-from Apple's privacy-focused on-device intelligence to vivo's PhoneGPT, Honor's YOYO agent, Anthropic's Computer Use, and Zhipu.AI's AutoGLM-showcase how LLM-based agents are transcending traditional user interfaces. They enable more natural, efficient, and personalized human-device interactions. As models and methods continue to evolve, we can anticipate even more groundbreaking applications, further integrating AI into the fabric of daily life and professional workflows.

## 3 FRAMEWORKS AND COMPONENTS OF PHONE GUI AGENTS

MLLM-powered phone GUI agents can be designed using different architectural paradigms and components, ranging from straightforward, single-agent systems [48], [50], [51], [53], [57] to more elaborate multi-agent [52], [55], [220] or multi-stage [47], [80], [82] approaches. A fundamental scenario involves a single agent that operates incrementally, without precomputing an entire action sequence from the outset. Instead, the agent continuously observes the dynamically changing mobile environment-where available UI elements, device states, and relevant contextual factors may shift in unpredictable ways-and cannot be exhaustively enumerated in advance. As a result, the agent must adapt its strategy step-by-step, making decisions based on the current situation rather than following a fixed plan. This iterative decision-making process can be effectively modeled using a Partially Observable Markov Decision Process (POMDP), a well-established framework for handling sequential decision-making under uncertainty [221], [222]. By modeling the task as a POMDP, we capture its dynamic nature, the impossibility of pre-planning all actions, and the necessity of adjusting the agent's approach at each decision point.

As illustrated in Figure 4, consider a simple example: the agent's goal is to order a latte through the Starbucks app. The app's interface may vary depending on network latency, promotions displayed, or the user's last visited screen. The agent cannot simply plan all steps in advance; it must observe the current screen, identify which UI elements are present, and then choose an action (like tapping the Starbucks icon, swiping to a menu, or selecting the latte). After each action, the state changes, and the agent re-evaluates its options. This dynamic, incremental decision-making is precisely why POMDPs are a suitable framework. In the POMDP formulation for phone automation:

States (S). At each decision point, the agent's perspective is described as a state, a comprehensive snapshot of all relevant information that could potentially influence the decision-making process. This state encompasses the current UI information (e.g., screenshots, UI trees, OCR-extracted text, icons), the phone's own status (network conditions, battery level, location), and the task context (the user's goal—"order a latte"—and the agent's progress toward it). The state ${S}_{t}$ represents the complete,underlying situation of the environment at time $t$ ,which may not be directly observable in its entirety.

Actions $\left( A\right)$ . Given the state ${S}_{t}$ at time $t$ ,the agent selects from available actions (taps, swipes, typing text, launching apps) that influence the subsequent state. The details of how phone GUI agents make decisions are introduced in § 3.2, and the design of the action space is discussed in § 3.3.

Transition Dynamics $\left( {P\left( {{s}^{\prime } \mid  s,a}\right) }\right)$ . When the agent executes an action ${a}_{t}$ at time $t$ ,it leads to a new state ${S}_{t + 1}$ . Some transitions may be deterministic (e.g., tapping a known button reliably opens a menu), while others are uncertain (e.g., network delays, unexpected pop-ups). Mathematically, we have the transition probability $P\left( {{s}^{\prime } \mid  s,a}\right)$ which describes the likelihood of transitioning from state ${S}_{t}$ to state ${S}_{t + 1}$ given action ${a}_{t}$ .

Observations $\left( O\right)$ . The agent receives observations ${O}_{t}$ at time $t$ which are partial and imperfect reflections of the true state ${S}_{t}$ . In the phone automation context,these observations could be, for example, a glimpse of the visible UI elements (not the entire UI tree), a brief indication of the network status (such as a signal icon without detailed connection parameters), or a partial view of the battery level indicator. These observations ${O}_{t}$ provide the agent with some,but not all,of the information relevant to the state ${S}_{t}$ . The agent must infer and make decisions based on these limited observations, attempting to reach the desired goal state despite the partial observability. The details of phone GUI agent perception are discussed in § 3.1.

Under this POMDP-based paradigm, the agent aims to make decisions that lead to the goal state by observing the current state and choosing appropriate actions. It continuously re-evaluates its strategy as conditions evolve, promoting real-time responsiveness and dynamic adaptation. The agent observes the state ${S}_{t}$ at time $t$ ,chooses an action ${a}_{t}$ ,and then based on the resulting observation ${O}_{t + 1}$ and new state ${S}_{t + 1}$ ,refines its strategy.

As illustrated in Figure 5, frameworks of phone GUI agents aim to integrate perception, reasoning, and action capabilities into cohesive agents that can interpret user intentions, understand complex UI states, and execute appropriate operations within mobile environment. By examining these frameworks, we can identify best practices, guide future advancements, and choose the right approach for various applications and contexts.

To address limitations in adaptability and scalability, §3.4 introduces multi-agent frameworks, where specialized agents collaborate, enhance efficiency, and handle more diverse tasks in parallel. Finally, §3.5 presents the Plan-Then-Act Framework, which explicitly separates the planning phase from the execution phase. This approach allows agents to refine their conceptual plans before acting, potentially improving both accuracy and robustness.


<!-- Meanless: 9<br>I want a latte delivered to my office. -->

<!-- Media -->

<!-- figureText: ${O}_{1}$ : Main menu with "Latte" button visible<br>${\mathrm{O}}_{2}$ : Latte product details visible<br>X<br>${A}^{0}$ : Tap Latte Directly<br>0: Starbucks app icon visible<br>${A}^{0}$ : Tap Starbucks Icon<br>${S}_{1}$ : Starbucks App<br>卷<br>${S}_{2}$ : Latte Selected<br>${A}^{0}$ : Tap Wrong App<br>So: Home Screen<br>1: Swipe to Menu<br>Tap "Order"<br>-<br>${A}^{0}$ : Tap Latte Directly<br>${A}^{2}$ : Task as completed.<br>${A}^{0}$ : Tap Wrong App<br>${A}^{1}$ : Swipe Again<br>(STOP)<br>${S}_{3}$ : Menu Open<br>${S}_{4}$ : Order Confirmed<br>X<br>${O}_{3}$ : Beverage list Qwith "Latte" option<br>${O}_{4}$ : Summary of order with "Confirm" button -->

<img src="https://cdn.noedgeai.com/bo_d41gln3ef24c73d3u2c0_8.jpg?x=170&y=273&w=1438&h=593&r=0"/>

Fig. 4: POMDP model for ordering a latte. Each circle represents a state (e.g., Home Screen, App Homepage, Latte Details Page,Customize Order,Order Confirmation,Order Complete). The agent starts at the initial state ${S}_{0}$ (Home Screen) and makes decisions at each step (e.g., tapping the Starbucks app icon, selecting the "Latte" button, viewing latte details). Due to partial observability,the agent receives limited information at each decision point (e.g., ${O}_{0}$ : Starbucks app icon visible, ${O}_{1}$ : "Latte" button visible, ${O}_{2}$ : Latte product details visible). Some actions correctly advance towards the goal,while others may cause errors requiring corrections. The final goal is to confirm the order.

<!-- Media -->

### 3.1 Perception in Phone GUI Agents

Perception is a fundamental component of the basic framework for MLLM-powered phone GUI agents. It is responsible for capturing and interpreting the state of the mobile environment, enabling the agent to understand the current context and make informed decisions. In the overall pipeline, perception serves as the initial step in the POMDP, providing the necessary input for the reasoning and action modules to operate effectively.

#### 3.1.1 UI Information Perception

UI information is crucial for agents to interact seamlessly with the mobile interface. It can be further categorized into UI tree-based and screenshot-based approaches, supplemented by techniques like Set-of-Marks (SoM) and Icon & OCR enhancements.

UI tree is a structured, hierarchical representation of the UI elements on a mobile screen [223], [224]. Each node in the UI tree corresponds to a UI component, containing attributes such as class type, visibility, and resource identifiers. ${}^{9}$ Early datasets like PixelHelp [132], MoTIF [133], and UIBert [134] utilized UI tree data to enable tasks such as mapping natural language instructions to UI actions and performing interactive visual environment interactions. DroidBot-GPT [53] was the first work to investigate how pre-trained language models can be applied to app automation without modifying the app or the model. DroidBot-GPT uses the UI tree as its primary perception information. The challenge lies in converting the structured UI tree into a format that LLMs can process effectively. DroidBot-GPT addresses this by transforming the UI tree into natural language sentences. Specifically, it extracts all user-visible elements, generates prompts like "A view <name>that can..." for each element, and combines them into a cohesive description of the current UI state. This approach mitigates the issue of excessively long and complex UI trees by presenting the information in a more natural and concise format suitable for LLMs. Subsequent developments, such as Enabling Conversational Interaction [57] and AutoDroid [50], further refined this approach by representing the view hierarchy as HTML. Enabling Conversational Interaction introduces a method to convert the view hierarchy into HTML syntax, mapping Android UI classes to corresponding HTML tags and preserving essential attributes such as class type, text, and resource identifiers. This representation aligns closely with the training data distribution of LLMs, enhancing their ability to perform few-shot learning and improving overall UI understanding. AutoDroid extends this work by developing a GUI parsing module that converts the GUI into a simplified HTML representation using specific HTML tags like <button>, <checkbox>, <scroller>, <input>, and <p>. Additionally, AutoDroid implements automatic scrolling of scrollable components to ensure that comprehensive UI information is available to the LLM, thereby enhancing decision-making accuracy and reducing computational overhead. Furthermore, LLMPA [58] employs object detection models to comprehend page layouts and optimizes the grouping of UI elements for potential actions. This approach reduces redundant information in the UI tree, thereby enhancing the accuracy and speed of decision making. Similar to this approach, the TOL Agent [59] introduces a variant of the UI tree, known as the Hierarchical Layout Tree, to represent the hierarchical layout of screen captures. In this tree, nodes represent different levels of regions. This structure, combined with a trained DINO model, aids in generating more accurate screen descriptions for MLLM.

---

<!-- Footnote -->

9. https://developer.android.com/reference/android/view/View.

<!-- Footnote -->

---


<!-- Meanless: 10 -->

<!-- Media -->

<!-- figureText: Enviroment<br>口秀<br>I want a latte delivered to my office.<br>( $\curvearrowright$<br>Perception<br>Intent Comprehension<br>GOOGA<br>Action<br>User<br>Single/Multi-Agent<br>Phone<br>Perception<br>Action<br>UI Info<br>Touch Interactions<br>Brain<br>1.2<br>345<br>screenshoot<br>SOM<br>UI tree<br>Storage<br>click<br>type<br>back<br>OCR<br>memory<br>knowledge<br>...<br>ocr info<br>ocr info<br>...<br>Phone State<br>home<br>swip<br>wifi<br>data<br>location<br>Decision Making<br>Atomic Skills<br>田<br>...<br>Tools<br>APIs<br>battery keyboard<br>planning<br>ng reasoning reflection -->

<img src="https://cdn.noedgeai.com/bo_d41gln3ef24c73d3u2c0_9.jpg?x=166&y=114&w=1463&h=1106&r=0"/>

Fig. 5: Overview of MLLM-powered phone GUI agent framework. The user's intent, expressed through natural language, is mapped to UI operations. By perceiving UI information and phone state(§3.1) , the agent leverages stored knowledge and memory to plan, reason, and reflect (§3.2). Finally, it executes actions to fulfill the user's goals(§3.3).

<!-- Media -->

Screenshots provide a visual snapshot of the current UI, capturing the appearance and layout of UI elements. Unlike UI trees, which require API access and can become unwieldy with complex hierarchies, screenshots offer a more flexible and often more comprehensive representation of the UI. Additionally, UI trees present challenges such as missing or overlapping controls and the inability to directly reference UI elements programmatically, making screenshots a more practical and user-friendly alternative for quickly assessing and sharing the state of a user interface. Auto-GUI [66] introduced a multimodal agent that relies on screenshots for GUI control, eliminating the dependency on UI trees. This approach allows the agent to interact with the UI directly through visual perception, enabling more natural and human-like interactions. Auto-GUI employs a chain-of-action technique that uses both previously executed actions and planned future actions to guide decision-making, achieving high action type prediction accuracy and efficient task execution. Following Auto-GUI, a series of multimodal solutions emerged, including MM-Navigator [60], CogA-gent [46], AppAgent [48], VisionTasker [85], MobileGPT [61], GUI Narrator [62], MobileVLM [63], AdaptAgent [225], WebVoyager [226] and Steward [227]. These frameworks leverage screenshots in combination with supplementary information to enhance UI understanding and interaction capabilities.

Set-of-Mark (SoM) is a prompting technique used to annotate screenshots with OCR, icon, and UI tree information, thereby enriching the visual data with textual descriptors [228]. For example, MM-Navigator [60] uses SoM to label UI elements with unique identifiers, allowing the LLM to reference and interact with specific components more effectively. This method has been widely adopted in subsequent works such as AppAgent [48], VisionDroid [54], OmniParser [56] and VisualWebArena [229], which utilize SoM to enhance the agent's ability to interpret and act upon UI elements based on visual, textual, and structural cues.


<!-- Meanless: 11 -->

Icon & OCR enhancements provide additional layers of information that complement the visual data, enabling more precise action decisions. For instance, Mobile-Agent-v2 [52] integrates OCR and icon data with screenshots to provide a richer context for the LLM, allowing it to interpret and execute more complex instructions that require understanding both text and visual icons. Icon & OCR enhancements are employed in various works, including VisionTasker [85], MobileGPT [61], OmniParser [56], and WindowsAgentArena [230], to improve the accuracy and reliability of phone GUI agents.

#### 3.1.2 Phone State Perception

Phone state information, such as keyboard status and location data, further contextualizes the agent's interactions. For example, Mobile-Agent-v2 [52] uses keyboard status to determine when text input is required. Location data, while not currently utilized, represents a potential form of phone state information that could be used to recommend nearby services or navigate to specific addresses. This additional state information enhances the agent's ability to perform context-aware actions, making interactions more intuitive and efficient.

The perception information gathered through UI trees, screenshots, SoM, OCR, and phone state is converted into prompt tokens that the LLM can process. This conversion is crucial for enabling seamless interaction between the perception module and the reasoning and action modules. Detailed methodologies for transforming perception data into prompt formats are discussed in § 4.1.

### 3.2 Brain in Phone GUI Agents

The brain of an LLM-based phone automation agent is its cognitive core, primarily constituted by a LLM. The LLM serves as the agent's reasoning and decision-making center, enabling it to interpret inputs, generate appropriate responses, and execute actions within the mobile environment [231], [232]. Leveraging the extensive knowledge embedded within LLMs, agents benefit from advanced language understanding, contextual awareness, and the ability to generalize across diverse tasks and scenarios.

#### 3.2.1 Storage

Storage encompasses both memory and knowledge, which are critical for maintaining context and informing the agent's decision-making processes.

Memory refers to the agent's ability to retain information from past interactions with users and the environment [44]. This is particularly useful for cross-application operations, where continuity and coherence are essential for completing multi-step tasks. For example, Mobile-Agent-v2 [52] integrates a memory unit that records task-related focus content from historical screens. This memory is accessed by the decision-making module when generating operations, ensuring that the agent can reference and update relevant information dynamically. The Self-MAP framework [233] establishes a memory repository based on the history of conversational interactions. It utilizes a multifaceted matching approach to retrieve the top-K memory snippets that are semantically relevant to the current dialogue state and have similar trajectories. This assists the agent in effectively utilizing limited context space during multi-turn interactions, thereby enhancing its ability to comprehend and execute user instructions.

Knowledge pertains to the agent's understanding of phone automation tasks and the functionalities of various apps. This knowledge can originate from multiple sources:

- Pre-trained Knowledge. LLMs are inherently equipped with a vast amount of general knowledge, including common-sense reasoning and familiarity with programming and markup languages such as HTML. This preexisting knowledge allows the agent to interpret and generate meaningful actions based on the UI representations.

- Domain-Specific Training. Some agents enhance their knowledge by training on phone automation-specific datasets. Works such as Auto-GUI [66], CogAgent [46], ScreenAI [67], CoCo-agent [71], and Ferret-UI [101] have trained LLMs on datasets tailored for mobile UI interactions, thereby improving their capability to understand and manipulate mobile interfaces effectively. For a more detailed discussion of knowledge acquisition through model training, see § 4.2.

- Knowledge Injection. Agents can enhance their decision-making by incorporating knowledge derived from exploratory interactions and stored contextual information. This involves utilizing data collected during offline exploration phases or from observed human demonstrations to inform the LLM's reasoning process. For instance, AutoDroid [50] explores app functionalities and records UI transitions in a UI Transition Graph (UTG) memory, which are then used to generate task-specific prompts for the LLM. Similarly, AppAgent [48] compiles knowledge from autonomous interactions and human demonstrations into structured documents, enabling the LLM to make informed decisions based on comprehensive UI state information and task requirements. AppAgent v2 [64] introduces a more efficient mechanism for knowledge base construction and updating. It leverages Retrieval-Augmented Generation (RAG) technology to achieve real-time dynamic updates of knowledge base information. This significantly enhances the agent's adaptability in new environments.AppAgentX [65] introduces an evolutionary mechanism that enables dynamic learning from past interactions and replaces inefficient low-level operations with high-level actions. Other similar works include AdaptAgent [225], Mobile-Agent-V [68], LearnAct [155] and others.

#### 3.2.2 Decision Making

Decision Making is the process by which the agent determines the appropriate actions to perform based on the current perception and stored information [44]. The LLM processes the input prompts, which include the current UI state, historical interactions from memory, and relevant knowledge, to generate action sequences that accomplish the assigned tasks.


<!-- Meanless: 12 -->

Planning involves devising a sequence of actions to achieve a specific task goal [44], [216]. Effective planning is essential for decomposing complex tasks into manageable steps and adapting to changes in the environment. For instance, Mobile-Agent-v2 [52] incorporates a planning agent that generates task progress based on historical operations, ensuring effective operation generation by the decision agent. Additionally, approaches like Dynamic Planning of Thoughts (D-PoT) have been proposed to dynamically adjust plans based on environmental feedback and action history, significantly improving accuracy and adaptability in task execution [220]. Simultaneously, by reducing the number of calls to LLMs and employing a phased planning strategy, the agent can plan all actions in a given state at once, thereby enhancing planning efficiency [234].

Reasoning enables the agent to interpret and analyze information to make informed decisions [235], [236], [237]. It involves understanding the context, evaluating possible actions, and selecting the most appropriate ones to achieve the desired outcome. By leveraging chain-of-thought(COT) [238], LLMs enhance their reasoning capabilities, allowing them to think step-by-step and handle intricate decision-making processes. This structured approach facilitates the generation of coherent and logical action sequences, ensuring that the agent can navigate complex UI interactions effectively. The best-first tree search algorithm is utilized in real-world environments to iteratively construct, explore, and prune trajectory graphs, thereby enhancing the reasoning and decision-making capabilities of agents. A value function serves as a reward signal to guide agents in conducting efficient searches [239]. Additionally, research indicates that LLMs to estimate the latent states of agents, in combination with reasoning methods, can further improve the agents' reasoning performance [240].

Reflection allows the agent to assess the outcomes of its actions and make necessary adjustments to improve performance [241]. It involves evaluating whether the executed actions meet the expected results and identifying any discrepancies or errors. For example, Mobile-Agent-v2 [52] includes a reflection agent that evaluates whether the decision agent's operations align with the task goals. If discrepancies are detected, the reflection agent generates appropriate remedial measures to correct the course of action. This continuous feedback loop enhances the agent's reliability and ensures that it can recover from unexpected states or errors during task execution. Furthermore, structured self-reflection identifies initial erroneous actions, which prevents agents from repeating the same mistakes. It also draws on reflective memory to avoid known unsuccessful actions [234]. Additionally, regular reflection through automated evaluation methods significantly enhances the performance of agents [242], [243].

By integrating robust planning, advanced reasoning, and reflective capabilities, the Decision Making component of the Brain ensures that MLLM-powered phone GUI agents can perform tasks intelligently and adaptively. These mechanisms enable the agents to handle a wide range of scenarios, maintain task continuity, and improve their performance over time through iterative learning and adjustment.

### 3.3 Action in Phone GUI Agents

The Action component is a critical part of MLLM-powered phone GUI agents, responsible for executing decisions made by the Brain within the mobile environment. By bridging high-level commands generated by the LLM with low-level device operations, the agent can effectively interact with the phone's UI and system functionalities. Actions encompass a wide variety of operations, ranging from simple interactions like tapping a button to complex tasks such as launching applications or modifying device settings. Execution mechanisms leverage tools like Android's UI Automator [244], iOS's XCTest [245], or popular automation frameworks such as Appium [246] and Selenium [247], [248] to send precise commands to the phone. Through these mechanisms, the agent ensures that decisions are translated into tangible, reliable operations on the device.

The types of actions in phone GUI agents are diverse and can be broadly categorized based on their functionalities. Table 1 summarizes these actions, providing a clear overview of the operations agents can perform.

The above categories reflect the key interactions required for phone automation. Touch interactions form the foundation of UI navigation, while gesture-based actions add flexibility for dynamic control. Typing and input enable text-based operations, whereas system operations and media controls extend the agent's capabilities to broader device functionalities. By combining these actions, phone GUI agents can achieve high accuracy and adaptability in executing user tasks, ensuring a seamless experience even in complex and dynamic environment.

### 3.4 Multi-Agent Framework

While single-agent frameworks based on LLMs have achieved significant progress in screen understanding and reasoning, they operate as isolated entities [249], [250], [251]. This isolation limits their flexibility and scalability in complex tasks that may require diverse, coordinated skills and adaptive capabilities. Single-agent systems may struggle with tasks that demand continuous adjustments based on real-time feedback, multi-stage decision-making, or specialized knowledge in different domains. Furthermore, they lack the ability to leverage shared knowledge or collaborate with other agents, reducing their effectiveness in dynamic environment [44], [52], [72], [73].

Multi-agent frameworks address these limitations by facilitating collaboration among multiple agents, each with specialized functions or expertise [252], [253], [254], [255], [256], [257], [258], [259]. This collaborative approach enhances task efficiency, adaptability, and scalability, as agents can perform tasks in parallel or coordinate their actions based on their specific capabilities. As illustrated in Figure 6, multi-agent frameworks in phone automation can be categorized into two primary types: the Role-Coordinated Multi-Agent Framework and the Scenario-Based Task Execution Framework. These frameworks enable more flexible, efficient, and robust solutions in phone automation by either organizing agents based on general functional roles or dynamically assembling specialized agents according to specific task scenarios.


<!-- Meanless: 13 -->

<!-- Media -->

TABLE 1: Types of actions in phone GUI agents

<table><tr><td>Action Type</td><td>Description</td></tr><tr><td>Touch Interactions</td><td>Tap: Select a specific UI element. <br> Double Tap: Quickly tap twice to trigger an action. <br> Long Press: Hold a touch for extended interaction, triggering contextual options or menus.</td></tr><tr><td>Gesture-Based Actions</td><td>Swipe: Move a finger in a direction (left, right, up, down). <br> Pinch: Zoom in/out by bringing fingers together/apart. <br> Drag: Move UI elements to a new location.</td></tr><tr><td>Typing and Input</td><td>Type Text: Enter text into input fields. <br> Select Text: Highlight text for editing or copying.</td></tr><tr><td>System Operations</td><td>Launch Application: Open a specific app. <br> Change Settings: Modify system settings (e.g., Wi-Fi, brightness). <br> Navigate Menus: Access app sections or system menus.</td></tr><tr><td>Media Control</td><td>Play/Pause: Control media playback. <br> Adjust Volume: Increase or decrease device volume.</td></tr></table>

<!-- figureText: The user's goal is clear. I'll break it down into actionable steps.<br>Which scenario are we tackling-shopping, traveling, or coding today?<br>Planning Agent<br>Host Agent<br>Got it. the settings menu to begin.<br>Looking for a deal? I can compare prices and add items to your cart!<br>Planning a trip? Let me find flights, hotels, and build your itinerary!<br>LEEPLE<br>Action Agent<br>Shopping Agent<br>Travel Agent<br>O<br>Check complete. The previous step succeeded. Let's continue to the next action.<br>0<br>Need a code fix or a snippet review? Send it my way, I'll help optimize it!<br>Reflection Agent<br>Code Agent<br>Role-Coordinated Multi-Agent Framework<br>Scenario-Based Task Execution Framework -->

<img src="https://cdn.noedgeai.com/bo_d41gln3ef24c73d3u2c0_12.jpg?x=175&y=943&w=1461&h=541&r=0"/>

Fig. 6: Comparison of the role-coordinated and scenario-based multi-agent frameworks. The Role-Coordinated framework organizes agents based on general functional roles with a fixed workflow, while the Scenario-Based framework dynamically assigns tasks to specialized agents tailored for specific scenarios, allowing for increased flexibility and adaptability in handling diverse tasks.

<!-- Media -->

#### 3.4.1 Role-Coordinated Multi-Agent

In the Role-Coordinated Multi-Agent Framework, agents are assigned general functional roles such as planning, decision-making, memory management, reflection, or tool invocation. These agents collaborate through a predefined workflow, with each agent focusing on its specific function to collectively achieve the overall task. This approach is particularly beneficial for tasks that require a combination of these general capabilities, allowing each agent to specialize and optimize its role within the workflow.

For example, in MMAC-Copilot [72], multiple agents with distinct general functions collaborate as an OS copilot. The Planner strategically manages and allocates tasks to other agents, optimizing workflow efficiency. Meanwhile, the Librarian handles information retrieval and provides foundational knowledge, and the Programmer is responsible for coding and executing scripts, directly interacting with the software environment. The Viewer interprets complex visual information and translates it into actionable commands, while the Video Analyst processes and analyzes video content. Additionally, the Mentor offers strategic oversight and troubleshooting support. Each agent contributes its specialized function to the collaborative workflow, thereby enhancing the system's overall capability to handle complex interactions with the operating system.

Similarly, in Mobile-Agent-v2 [52], three agents with general roles are utilized: a planning agent, a decision agent, and a reflection agent. The planning agent compresses historical actions and state information to provide a concise representation of task progress. The decision agent uses this information to navigate the task effectively, while the reflection agent monitors the outcomes of actions and corrects any errors, ensuring accurate task completion. This role-based collaboration reduces context length, improves task progression, and enhances focus content retention through a memory unit managed by the decision agent.


<!-- Meanless: 14 -->

In contrast, Mobile-Agent-E [74] decomposes tasks into high-level planning and low-level action execution, creating a system with a Manager Agent responsible for high-level planning and four subordinate agents: the Perceptor Agent, Operator Agent, Action Reflector Agent, and Notetaker Agent. The Perceptor Agent is responsible for fine-grained visual perception. The Operator Agent determines the next specific actions based on task and perception information. The Action Reflector Agent checks the screenshots before and after operations to verify if the expected outcomes are achieved and provides feedback to the Manager and Operator Agents. The Notetaker Agent extracts task-related information for use in subsequent steps. Additionally, Mobile-Agent-E incorporates a Self-Evolution Module, using two specialized agents, AES and AET, to update long-term memory after each task completion. AES summarizes lessons learned, while AET records reusable operational sequences, helping the agent in efficiently completing common subtasks and making better decisions in similar future tasks.

CHOP [76] introduces a mobile operating assistant with Constrained High-frequency Optimized subtask Planning. This approach addresses challenges in the subtask level, which links high-level goals with low-level executable actions. CHOP overcomes VLM's deficiency in GUI scenario planning by using human-planned subtasks as basis vectors, significantly improving both effectiveness and efficiency across multiple applications in both English and Chinese contexts. The framework specifically targets two common issues: ineffective subtasks that lower-level agents cannot execute and inefficient subtasks that fail to contribute to higher-level task completion.

In general computer automation, Cradle [73] leverages foundational agents with general roles to achieve versatile computer control. Agents specialize in functions like command generation or state monitoring, enabling Cradle to tackle general-purpose tasks across multiple software environment. Additionally, studies such as Ask-before-Plan [78], PromptRPA [75], LUMOS [260], and WebPilot [261] also utilize general-purpose role agents to execute tasks and excel in complex tasks like planning. Among these, LUMOS provides high-quality training data and methods for future intelligent agent research. Agent S2 [77] presents a compositional generalist-specialist framework for computer use agents that delegates cognitive responsibilities across various models. It introduces a Mixture-of-Grounding technique for precise GUI localization and Proactive Hierarchical Planning that dynamically refines action plans at multiple temporal scales based on evolving observations.

#### 3.4.2 Scenario-Based Task Execution

In the Scenario-Based Task Execution Framework, tasks are dynamically assigned to specialized agents based on specific task scenarios or application domains. Each agent is endowed with capabilities tailored to a particular scenario, such as shopping, code editing, or navigation. By assigning tasks to agents specialized in the relevant domain, the system improves task success rates and efficiency.

For instance, MobileExperts [55] forms different expert agents through an Expert Exploration phase. In the exploration phase, each agent receives tailored tasks broken down into sub-tasks to streamline the exploration process. Upon completion of a sub-task, the agent extracts three types of memories from its trajectory: interface memories, procedural memories (tools), and insight memories for use in subsequent execution phases. When a new task arrives, the system dynamically forms an expert team by selecting agents whose expertise matches the task requirements, enabling them to collaboratively execute the task more effectively. Similarly, in the SteP [79] framework, agents are specialized based on specific web scenarios such as shopping, GitLab, maps, Reddit, or CMS platforms. Each scenario agent possesses specific capabilities and knowledge relevant to its domain. When a task is received, it is dynamically assigned to the appropriate scenario agent, which executes the task leveraging its specialized expertise. This approach enhances flexibility and adaptability, allowing the system to handle a wide range of tasks across different domains more efficiently.

Through dynamic task assignment and specialization, the Scenario-Based Task Execution Framework optimizes multi-agent systems to adapt to diverse and evolving contexts, significantly enhancing both the efficiency and effectiveness of task execution. As illustrated in Figure 6, the Role-Coordinated Framework relies on agents with general functional roles collaborating through a fixed workflow, suitable for tasks requiring a combination of general capabilities. In contrast, the Scenario-Based Framework dynamically assigns tasks to specialized agents tailored to specific scenarios, providing a flexible structure that adapts to the varying complexity and requirements of real-world tasks.

Despite the potential of multi-agent frameworks in phone automation, several challenges remain. In the Role-Coordinated Framework, coordinating agents with general functions requires efficient workflow design and may introduce overhead in communication and synchronization. In the Scenario-Based Framework, maintaining and updating a diverse set of specialized agents can be resource-intensive, and dynamically assigning tasks requires effective task recognition and agent selection mechanisms. Future research could explore hybrid frameworks that combine the strengths of both approaches, leveraging general functional agents while also incorporating specialized scenario agents as needed. Additionally, developing advanced algorithms for agent collaboration, learning, and adaptation can further enhance the intelligence and robustness of multi-agent systems. Integrating external knowledge bases, real-time data sources, and user feedback can also improve agents' decision-making capabilities and adaptability in dynamic environment.

### 3.5 Plan-Then-Act Framework

While single-agent and multi-agent frameworks enhance adaptability and scalability, some tasks benefit from explicitly separating high-level planning from low-level execution. This leads to what we term the Plan-Then-Act Framework. In this paradigm, the agent first formulates a conceptual plan—often expressed as human-readable instructions—before grounding and executing these instructions on the device's UI.


<!-- Meanless: 15 -->

<!-- Media -->

<!-- figureText: Prompt Engineering<br>Supervised Fine-Tuning for GUI Tasks<br>GUI-Specific LLM Architectures<br>4<br>Reinforcement Learning for Phone Agents<br>Task Guidance<br>Domain Adaptation<br>Customized Design<br>Reward-Driven Optimization<br>UI Contextualization<br>Labeled Data Dependency<br>Improved Generalization<br>Sequential Decision-Making<br>No Training Required<br>Parameter Optimization<br>Task-Specific Modules<br>Adaptive Learning Strategies<br>Task<br>e.g., LoRA:<br>e.g., CogAgent:<br>e.g., DigiRL:<br>"Your task is..."<br>Inputs X<br>Pretraining<br>Step II: Online RL<br>UI infomation<br>Tasks are sampled from task dataset<br>"Here is the current UI infomation of the Phone..."<br>WA<br>㉘<br>Model executes tasks in parallel and produce trajectories<br>Pre-trained weights $W$<br>Step I: Offline RL<br>Action Space<br>WB<br>"You must choose one of the actions below..."<br>are used to update the<br>model through online<br>XX<br>Embeddings h<br>...<br>Pretrained LLM<br>Prompt engineering method<br>Training-based method -->

<img src="https://cdn.noedgeai.com/bo_d41gln3ef24c73d3u2c0_14.jpg?x=192&y=119&w=1435&h=728&r=0"/>

Fig. 7: Differences between training-based methods and prompt engineering in phone automation. Training-based methods adapt the model's parameters through additional training, enhancing its ability to perform specific tasks, whereas prompt engineering leverages the existing capabilities of pre-trained models by guiding them with well-designed prompts.

<!-- Media -->

The Plan-Then-Act approach addresses a fundamental challenge: although LLMs and multimodal LLMs (MLLMs) excel at interpreting instructions and reasoning about complex tasks, they frequently struggle to precisely map their textual plans to concrete UI actions. By decoupling these stages, the agent can focus on what should be done (planning) and then handle how to do it on the UI (acting). Recent works highlight the effectiveness of this approach:

- SeeAct [47] demonstrates that GPT-4V(ision) [31] can generate coherent plans for navigating websites. However, bridging the gap between textual plans and underlying UI elements remains challenging. By clearly delineating planning from execution, the system can better refine its plan before finalizing actions.

- UGround [80] and related efforts [95], [101] emphasize advanced visual grounding. Under a Plan-Then-Act framework, the agent first crafts a task solution plan, then relies on robust visual grounding models to locate and manipulate UI components. This modular design enhances performance across diverse GUIs and platforms, as the grounding model can evolve independently of the planning mechanism.

- LiMAC (Lightweight Multi-modal App Control) [81] also embodies a Plan-Then-Act spirit. LiMAC's Action Transformer (AcT) determines the required action type (the plan), and a specialized VLM is invoked only for natural language needs. By structuring decision-making and text generation into distinct stages, LiMAC improves responsiveness and reduces compute overhead, ensuring that reasoning and UI interaction are cleanly separated.

- ClickAgent [82] similarly employs a two-phase approach. The MLLM handles reasoning and action planning, while a separate UI location model pinpoints the relevant coordinates on the screen. Here, the MLLM's plan of which element to interact with is formed first, and only afterward is the element's exact location identified and the action executed.

- Ponder & Press [83] employs a general MLLM to decompose user instructions into executable actions. It then uses a GUI-specific MLLM to map the target elements in the action descriptions to pixel coordinates, thereby constructing a Plan-Then-Act Framework based solely on visual input. This framework is adaptable across various software environments without relying on supplementary information such as HTML or UI Trees.

The Plan-Then-Act Framework offers several advantages. Modularity allows improvements in planning without requiring changes to the UI grounding and execution modules, and vice versa. Error Mitigation enables the agent to revise its plan before committing to actions; if textual instructions are ambiguous or infeasible, they can be corrected, reducing wasted actions and improving reliability. Additionally, improved visual grounding models, OCR enhancements, and scenario-specific knowledge can further refine the Plan-Then-Act approach, making agents more adept at handling intricate, real-world tasks. In summary, the Plan-Then-Act Framework represents a natural evolution in designing MLLM-powered phone GUI agents. By separating planning from execution, agents can achieve clearer reasoning, improved grounding, and ultimately more effective and reliable task completion.

## 4 LLMs for Phone Automation

LLMs [28], [29], [30], [31] have emerged as a transformative technology in phone automation, bridging natural language inputs with executable actions. By leveraging their advanced language understanding, reasoning, and generalization capabilities, LLMs enable agents to interpret complex user intents, dynamically interact with diverse mobile applications, and effectively manipulate GUIs.


<!-- Meanless: 16 -->

In this section, we explore two primary approaches to leveraging LLMs for phone automation: Training-Based Methods and Prompt Engineering. Figure 7 illustrates the differences between these two approaches in the context of phone automation. Training-Based Methods involve adapting LLMs specifically for phone automation tasks through techniques like supervised fine-tuning [105], [107], [109], [110] and reinforcement learning [116], [117], [124]. These methods aim to enhance the models' capabilities by training them on GUI-specific data, enabling them to understand and interact with GUIs more effectively. Prompt Engineering, on the other hand, focuses on designing input prompts to guide pre-trained LLMs to perform desired tasks without additional training [238], [262], [263]. By carefully crafting prompts that include relevant information such as task descriptions, interface states, and action histories, users can influence the model's behavior to achieve specific automation goals [48], [49], [53].

### 4.1 Prompt Engineering

LLMs like the GPT series [28], [29], [30] have demonstrated remarkable capabilities in understanding and generating human-like text. These models have revolutionized natural language processing by leveraging massive amounts of data to learn complex language patterns and representations.

Prompt engineering is the practice of designing input prompts to effectively guide LLMs to produce desired outputs for specific tasks [238], [262], [263]. By carefully crafting the prompts, users can influence the model's behavior without the need for additional training or fine-tuning. This approach allows for leveraging the general capabilities of pre-trained models to perform a wide range of tasks by simply providing appropriate instructions or examples in the prompt.

In the context of phone automation, prompt engineering enables the utilization of general-purpose LLMs to perform automation tasks on mobile devices. Recently, a plethora of works have emerged that apply prompt engineering to achieve phone automation [48], [49], [51], [52], [53], [54], [55], [56], [60], [75], [84], [264]. These works leverage the strengths of LLMs in natural language understanding and reasoning to interpret user instructions and generate corresponding actions on mobile devices.

The fundamental approach to achieving phone automation through prompt engineering entails the creation of prompts that encapsulate a comprehensive set of information. These prompts should include a detailed task description, such as searching for the best Korean restaurant on Yelp. They also integrate the current UI information of the phone, which may encompass screenshots, SoM, UI tree structures, icon details, and OCR data. Additionally, the prompts should account for the phone's real-time state, including its location, battery level, and keyboard status, as well as any pertinent action history and the range of possible actions (action space). The COT prompt [238], [265] is also a crucial component, guiding the thought process for the next operation. The LLM then analyzes this rich prompt and determines the subsequent action to execute. This methodical process is vividly depicted in Figure 8.

This section explores the application of prompt engineering in phone automation, categorizing related works based on the type of prompts used: Text-Based Prompt and Multimodal Prompt. As illustrated in Figure 9, the approach to automation significantly diverges between these two prompt types. Table 2 summarizes notable methods, highlighting their main UI information, the type of model used, and other relevant details such as task types and grounding strategies.

#### 4.1.1 Text-Based Prompting

In the domain of text-based prompt automation, the primary architecture involves a single text-modal LLM serving as the agent for mobile device automation. This agent operates by interpreting UI information presented in the form of a UI tree. It is important to note that, to date, the approaches discussed have primarily utilized UI tree data and have not extensively incorporated OCR text and icon information. We believe that solely relying on OCR and icon information is insufficient for fully representing screen UI information; instead, as demonstrated in Mobile-agent-v2 [52], they are best used as auxiliary information alongside screenshots. These text-based prompt agents make decisions by selecting elements from a list of candidates based on the textual description of the UI elements. For instance, to initiate a search, the LLM would identify and select the search button by its index within the UI tree rather than its screen coordinates, as depicted in Figure 9.

The study by Enabling Conversational [57] marked a significant step in this field. It explored the use of task descriptions, action spaces, and UI trees to map instructions to UI actions. However, it focused solely on the execution of individual instructions without delving into sequential decision-making processes. DroidBot-GPT [53] is a landmark in applying pre-trained language models to app automation. It is the first to explore the use of LLMs for app automation without requiring modifications to the app or the model. DroidBot-GPT perceives UI trees, which are structural representations of the app's UI, and integrates user-provided tasks along with action spaces and output requirements. This allows the model to engage in sequential decision-making and automate tasks effectively. AutoDroid [50] takes this concept further. It employs a UI Transition Graph (UTG) generated through random exploration to create an App Memory. This memory, combined with the commonsense knowledge of LLMs, enhances decision-making and significantly advances the capabilities of phone GUI agents. MobileGPT [61] introduces a hierarchical decision-making process. It simulates human cognitive processes-exploration, selection, derivation, and recall-to augment the efficiency and reliability of LLMs in mobile task automation. Lastly, AXNav [84] showcases an innovative application of Prompt Engineering in accessibility testing. AXNav interprets natural language instructions and executes them through an LLM, streamlining the testing process and improving the detection of accessibility issues, thus aiding the manual testing workflows of QA professionals.


<!-- Meanless: 17 -->

<!-- Media -->

<!-- figureText: Task<br>< 4 0 0 .<br>< 1<br>Your task is: Search for the best Korean restaurant near me on Yelp<br>Korean Restaurant<br>✘<br>UI Info<br>Draper, UT<br>Licon descrebtionlcoordinate 1<br>Korean Restaurant<br>Action History<br>Accon descrebtion/coordinate 2<br>icon descrebfion.coordinate 3<br>Korean Restaurants Near Me<br><icon/><br>[1]<br>text1 coordinate1<br>Korean Restaurant BBQ<br><icon/><br><icon/><br>text2 coordinate 2<br>text3 coordinate3<br>Overstruccise<br>SOM<br>UI tree<br>OCR info<br>Phone State<br>Pretrained LLM<br>Click(1020,140)<br>股 Amazon<br>Here is the current UI state of the Phone.<br>Bluen<br><icon/><br>The phone's location is at Hangzhou East Railway Station.<br>q' w' e' r' t' y' u i ' o ' p '<br>as di g h j k l<br>The battery is charging and is now fully charged.<br>- 2 . 2 . 2 . v b n w<br>22 。<br>The phone keyboard is activated.<br>Step1: open Yelp app<br>Step2: click search bar<br>Restaurant<br>Restaurants<br>Restaurant's<br>Let's think step by step!<br>COT<br>w<br>e<br>y<br>U<br>p<br>Think about the requirements that need to be completed in the next one operation.<br>Analyze the UI infomation and identify relevant elements for the next action.<br>d<br>f<br>g<br>j<br>k<br>Action Space<br>合<br>Z<br>X<br>C<br>V<br>n<br>m<br>☒<br>You must choose one of the actions below:<br>Touch Interactions<br>Atomic Skills<br>?123<br><icon/><br>柒<br>"Click (x,y):Tap the position (x,y)in current page."<br>"Ues tool (tool name): Use the \\"tool name\\"."<br>[] "Type (text):Type the \\"text\\"in the input box."<br>"Call api (api name):Call the \\"api name\\"."<br>·<br>型<br>necessary prompt<br>poptional prompt<br>flexible prompt -->

<img src="https://cdn.noedgeai.com/bo_d41gln3ef24c73d3u2c0_16.jpg?x=175&y=117&w=1463&h=869&r=0"/>

Fig. 8: Schematic of prompt engineering for phone automation. The necessary prompt is mandatory, initiating the task, e.g., searching for a Korean restaurant. The optional prompt are supplementary, enhancing tasks without being mandatory. The flexible prompt must include one or more elements from the UI Info, like a screenshot or OCR info, to adapt to task needs.

<!-- Media -->

Each of these contributions, while unique in their approach, is united by the common thread of Prompt Engineering. They demonstrate the versatility and potential of text-based prompt automation in enhancing the interaction between LLMs and mobile applications.

#### 4.1.2 Multimodal Prompting

With the advancement of large pre-trained models, Multimodal Large Language Models (MLLMs) have demonstrated exceptional performance across various domains [31], [229], [266], [267], [268], [269], [270], [271], [272], [273], [274], significantly contributing to the evolution of phone automation. Unlike text-only models, multimodal models integrate visual and textual information, addressing limitations such as the inability to access UI trees, missing control information, and inadequate global screen representation. By leveraging screenshots for decision-making, multimodal models facilitate a more natural simulation of human interactions with mobile devices, enhancing both accuracy and robustness in automated operations.

The fundamental framework for multimodal phone automation is illustrated in Figure 9. Multimodal prompts integrate visual perception (e.g., screenshots) and textual information (e.g., UI tree, OCR, and icon data) to guide MLLMs in generating actions. The action outputs can be categorized into two methods: SoM-Based Indexing Methods and Direct Coordinate Output Methods. These methods define how the agent identifies and interacts with UI elements, either by referencing annotated indices or by pinpointing precise coordinates. SoM-Based Indexing Methods. SoM-based methods involve annotating UI elements with unique identifiers within the screenshot, allowing the MLLM to reference these elements by their indices when generating actions. This approach mitigates the challenges associated with direct coordinate outputs, such as precision and adaptability to dynamic interfaces. MM-Navigator [60] represents a breakthrough in zero-shot GUI navigation using GPT-4V [31]. By employing SoM prompting [228], MM-Navigator annotates screenshots through OCR and icon recognition, assigning unique numeric IDs to actionable widgets. This enables GPT-4V to generate indexed action descriptions rather than precise coordinates, enhancing action execution accuracy. Building upon the SoM-based approach, AppAgent [48] integrates autonomous exploration and human demonstration observation to construct a comprehensive knowledge base. This framework allows the agent to navigate and operate smartphone applications through simplified action spaces, such as tapping and swiping, without requiring backend system access. Tested across 10 different applications and 50 tasks, AppAgent showcases superior adaptability and efficiency in handling diverse high-level tasks, further advancing multimodal phone automation. OmniParser [56] enhances the SoM-based method by introducing a robust screen parsing technique. It combines fine-tuned interactive icon detection models and functional captioning models to convert UI screenshots into structured elements with bounding boxes and labels. This comprehensive parsing significantly improves GPT-4V's ability to generate accurately grounded actions, ensuring reliable operation across multiple platforms and applications. GUI Narrator [62] utilizes video captioning to guide the VLM, aiding in the deeper understanding of GUI operations. The framework uses the mouse cursor as a visual prompt, highlighting it with a green bounding box to enhance the VLM's interpretative abilities with high-resolution screenshots. By extracting screenshots from before and after GUI actions occur in the video as keyframes, it provides temporal and spatial logic to the action screenshots. These are combined into prompts to further guide the VLM in producing accurate action descriptions, thereby improving its performance.


<!-- Meanless: 18 -->

<!-- Media -->

<!-- figureText: UI infomation<br>[OCR]<br>Additional layers of UI info.<br>Hierarchical representation.<br>Click or type<br>[OCR]<br>Additional layers of UI info.<br>恭<br>or ... ?<br>action type<br>ocr & icon info<br>UI tree<br>Text Perception<br>ocr & icon info<br>The location is $\left( {x,y}\right)$ .<br>Hierarchical representation.<br>coordinate<br>agent<br>LLM<br>UI tree<br>MLLM<br>Click or type<br>米<br>or ... ?<br>action<br>Click or type or ... ?<br>面三<br>面<br>The target element is 1/2/3..<br>Visual Perception<br>345<br>Set-of-Marks.<br>action type<br>The target<br>义<br>Screenshoot of current UI.<br>element is<br>1/2/3..<br>index<br>action type<br>index<br>UI infomation<br>agent<br>action<br>Text-Based Prompt<br>Multimodal Prompt -->

<img src="https://cdn.noedgeai.com/bo_d41gln3ef24c73d3u2c0_17.jpg?x=195&y=123&w=1434&h=724&r=0"/>

Fig. 9: Comparison between text-based prompt and multimodal prompt. In Text-Based Prompt, the LLM processes textual UI information, such as UI tree structures and OCR data, to determine the action type (index). In contrast, Multimodal Prompt integrates screenshot data with supplementary UI information to facilitate decision-making by the agent. The MLLM can then pinpoint the action location using either coordinates or indices.

<!-- Media -->

Direct Coordinate Output Methods. Direct coordinate output methods enable MLLMs to determine the exact $(x$ , y) positions of UI elements from screenshots, facilitating precise interactions without relying on indexed references. This approach leverages the advanced visual grounding capabilities of MLLMs to interpret and interact with the UI elements directly. VisionTasker [85] introduces a two-stage framework that combines vision-based UI understanding with LLM task planning. Utilizing models like YOLOv8 [275] and PaddleOCR [276], VisionTasker parses screenshots to identify widgets and textual information, transforming them into natural language descriptions. This structured semantic representation allows the LLM to perform step-by-step task planning, enhancing the accuracy and practicality of automated mobile task execution. The Mobile-Agent series [51], [52] leverages visual perception tools to accurately identify and locate both visual and textual UI elements within app screenshots. Mobile-Agent-v1 utilizes coordinate-based actions, enabling precise interaction with UI elements. Mobile-Agent-v2 extends this by introducing a multi-agent architecture comprising planning, decision, and reflection agents. Mobile-Agent-E [74] optimizes the multi-agent architecture by detailing the responsibilities of each agent. It also introduces a long-term memory mechanism through the design of a Self-Evolution Module, which accumulates experience and enables agents to evolve, thereby enhancing adaptability to new tasks. MobileExperts [55] advances the direct coordinate output method by incorporating tool formulation and multi-agent collaboration. This dynamic, tool-enabled agent team employs a dual-layer planning mechanism to efficiently execute multi-step operations while reducing reasoning costs by approximately 22%. By dynamically assembling specialized agents and utilizing reusable code block tools, MobileExperts demonstrates enhanced intelligence and operational efficiency in complex phone automation tasks. Unlike AppAgent, AppAgent v2 [64] integrates parsers with visual features and employs UI element coordinates along with Index information, creating a more flexible action space. This allows the agent to manage dynamic interfaces and non-standard UI elements more adeptly, thereby enhancing its adaptability to various complex tasks. VisionDroid [54] applies MLLMs to automated GUI testing, focusing on detecting non-crash functional bugs through vision-based UI understanding. By aligning textual and visual information, VisionDroid enables the MLLM to comprehend GUI semantics and operational logic, employing step-by-step task planning to enhance bug detection accuracy. Evaluations across multiple datasets and real-world applications highlight VisionDroid's superior performance in identifying and addressing functional bugs.

While multimodal prompt strategies have significantly advanced phone automation by integrating visual and textual data, they still face notable challenges. Approaches that do not utilize SoM maps and instead directly output coordinates rely heavily on the MLLM's ability to accurately ground UI elements for precise manipulation. Although recent innovations [52], [54], [55] have made progress in addressing the limitations of MLLMs' grounding capabilities, there remains considerable room for improvement. Enhancing the robustness and accuracy of UI grounding is essential to achieve more reliable and scalable phone automation.


<!-- Meanless: 19 -->

<!-- Media -->

TABLE 2: Summary of prompt engineering methods for phone GUI agents

<table><tr><td>Method</td><td>Date</td><td>Task Type</td><td>Model</td><td>Screenshot</td><td>$\mathbf{{SoM}}$</td><td>UI tree</td><td>Icon & OCR</td><td>Grounding</td></tr><tr><td>DroidBot-GPT [53] :</td><td>2023.04</td><td>General</td><td>ChatGPT</td><td>✘</td><td>✘</td><td>✓</td><td>✘</td><td>Index</td></tr><tr><td>Enabling conversational [57]</td><td>2023.04</td><td>Screen Understanding, QA</td><td>PaLM</td><td>✘</td><td>✘</td><td>✓</td><td>✘</td><td>Index</td></tr><tr><td>AutoDroid [50] !</td><td>2023.09</td><td>General</td><td>GPT-4, GPT-3.5</td><td>✘</td><td>✘</td><td>✓</td><td>✘</td><td>Index</td></tr><tr><td>MM-Navigator [60] !</td><td>2023.11</td><td>General</td><td>GPT-4V</td><td>✓</td><td>✘</td><td>✘</td><td>✓</td><td>Index</td></tr><tr><td>VisionTasker [85] !</td><td>2023.12</td><td>Manual Teaching</td><td>GPT-4</td><td>✘</td><td>✓</td><td>✓</td><td>✓</td><td>Index</td></tr><tr><td>AppAgent [48] !</td><td>2023.12</td><td>General</td><td>GPT-4</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td>Index</td></tr><tr><td>MobileGPT [61] ☑</td><td>2023.12</td><td>General</td><td>GPT-3.5, GPT-4</td><td>✘</td><td>✘</td><td>✓</td><td>✘</td><td>Index</td></tr><tr><td>Mobile-Agent [51] ?</td><td>2024.01</td><td>General</td><td>GPT-4V</td><td>✓</td><td>✘</td><td>✘</td><td>✓</td><td>Coordinate</td></tr><tr><td>AXNav [84]</td><td>2024.05</td><td>Bug Testing</td><td>GPT-4</td><td>✘</td><td>✘</td><td>✓</td><td>✓</td><td>Index</td></tr><tr><td>Mobile-Agent-v2 [52] ?</td><td>2024.06</td><td>General</td><td>GPT-4V</td><td>✓</td><td>✘</td><td>✘</td><td>✓</td><td>Coordinate</td></tr><tr><td>GUI Narrator [62] ?</td><td>2024.06</td><td>GUI Video Captioning</td><td>GPT-40, QwenVL-7B</td><td>✓</td><td>✓</td><td>✘</td><td>✘</td><td>Index</td></tr><tr><td>MobileExpert [55]</td><td>2024.07</td><td>General</td><td>GPT-4V</td><td>✓</td><td>✘</td><td>✘</td><td>✘</td><td>Coordinate</td></tr><tr><td>VisionDroid [54] ?</td><td>2024.07</td><td>Non-Crash Functional Bug Detection</td><td>GPT-4</td><td>✓</td><td>✓</td><td>✓</td><td>✘</td><td>Index</td></tr><tr><td>AppAgent v2 [64]</td><td>2024.08</td><td>General</td><td>GPT-4</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td>Coordinate, Index</td></tr><tr><td>OmniParser [56]</td><td>2024.08</td><td>General</td><td>GPT-4V</td><td>✓</td><td>✓</td><td>✘</td><td>✓</td><td>Index</td></tr><tr><td>Mobile-Agent-E [74] ◇</td><td>2025.01</td><td>General</td><td>GPT-40, Claude -3.5, Gemini-1.5</td><td>✓</td><td>✘</td><td>✘</td><td>✓</td><td>Coordinate</td></tr><tr><td>Mobile-Agent-V [68] ?</td><td>2025.02</td><td>General</td><td>GPT-4o</td><td>✓</td><td>✘</td><td>✘</td><td>✓</td><td>Coordinate</td></tr><tr><td>LearnAct [155] ?</td><td>2025.02</td><td>General</td><td>Gimini-1.5</td><td>✓</td><td>✘</td><td>✘</td><td>✘</td><td>Coordinate</td></tr></table>

<!-- Media -->

### 4.2 Training-Based Models

The subsequent sections delve into these approaches, discussing the development of task-specific model architectures, supervised fine-tuning strategies and reinforcement learning techniques in both general-purpose and Phone UI-specific scenarios.

#### 4.2.1 Task-Specific LLM-based Agents

To advance AI agents for phone automation, significant efforts have been made to develop Task Specific Model Architectures that are tailored to understand and interact with GUIs by integrating visual perception with language understanding. These models address unique challenges posed by GUI environment, such as varying screen sizes, complex layouts, and diverse interaction patterns. A summary of notable Task Specific Model Architectures is presented in Figure 3, highlighting their main contributions, domains, and other relevant details. General-Purpose Models. The general-purpose GUI-specific LLMs are designed to handle a wide range of tasks across different applications and interfaces. They focus on enhancing direct GUI interaction, high-resolution visual recognition, and comprehensive perception to improve the capabilities of AI agents in understanding and navigating complex mobile GUIs. One significant challenge in this domain is enabling agents to interact directly with GUIs without relying on environment parsing or application-specific APIs, which can introduce inefficiencies and error propagation. To tackle this, Auto-GUI [66] presents a multimodal agent that directly engages with the interface. It introduces a chain-of-action technique that leverages previous action histories and future action plans, enhancing the agent's decision-making process and leading to improved performance in GUI control tasks. High-resolution input is essential for recognizing tiny UI elements and text prevalent in GUIs. CogAgent [46] addresses this by employing both low-resolution and high-resolution image encoders within its architecture. Supporting input resolutions up to ${1120} \times  {1120}$ , CogAgent effectively recognizes small page elements and text. Understanding UIs and infographics requires models to interpret complex visual languages and design principles. ScreenAI [67] improves upon existing architectures by introducing a flexible patching strategy and a novel textual representation for UIs. During pre-training, this representation teaches the model to interpret UI elements effectively. Leveraging large language models, ScreenAI automatically generates training data at scale, covering a wide spectrum of tasks in UI and infographic understanding. Enhancing both perception and action response is crucial for comprehensive GUI automation. CoCo-Agent [71] proposes two novel approaches: comprehensive environment perception (CEP) and conditional action prediction (CAP). CEP enhances GUI perception through multiple aspects, including visual channels (screenshots and detailed layouts) and textual channels (historical actions). CAP decomposes action prediction into determining the action type first, then identifying the action target conditioned on the action type. Addressing the need for effective GUI agents in applications featuring extensive Mandarin content, MobileFlow [88] introduces a multimodal LLM specifically designed for mobile GUI agents. MobileFlow employs a hybrid visual encoder trained on a vast array of GUI pages, enabling it to extract and comprehend information across diverse interfaces. The model incorporates a Mixture of Experts (MoE) and specialized modality alignment training tailored for GUIs. ShowUI [89] employs the UI-Guided visual tokens selection method, which randomly selects a subset of tokens from each component during training. This approach retains the original positional information while reducing redundant tokens by 33%, thereby accelerating training speed by 1.4 times. Furthermore, by using interleaved vision-language-action streaming combined with high-quality training data, it significantly improves the training speed and performance of GUI visual agents. Aguvis [90] employs a two-stage training method to enhance the generalization and efficiency of GUI agents. It uses single-step task data to train the model's grounding abilities and multi-step task data to develop the model's planning and reasoning capabilities. This approach significantly improves the overall performance of the agents. UI-TARS [92] employs a more in-depth and structurally robust System-2 reasoning method, combined with online bootstrapping and reflection tuning strategies. This combination effectively assists the model in handling complex tasks in dynamic environments and continuously optimizes overall performance. V-Droid [114] introduces a novel verifier-driven architecture where the LLM does not generate actions directly but instead scores and selects from a finite set of extracted actions, improving task success rates and significantly reducing latency. Collectively, these general-purpose Task Specific Model Architectures address key challenges in phone automation by enhancing direct GUI interaction, high-resolution visual recognition, comprehensive environment perception, and conditional action prediction. By leveraging multimodal inputs and innovative architectural designs, these models significantly advance the capabilities of AI agents in understanding and navigating complex mobile GUIs, paving the way for more intelligent and autonomous phone automation solutions. Phone UI-Specific Models. Phone UI-Specific Model Architectures have primarily focused on screen understanding


<!-- Meanless: 20 -->

<!-- Media -->

TABLE 3: Summary of task-specific model architectures

<table><tr><td>Method</td><td>Date</td><td>Task Type</td><td>Backbone</td><td>Size</td><td>Contributions</td></tr><tr><td>Auto-GUI [66] ?</td><td>2023.09</td><td>General</td><td>N/A</td><td>60M / 200M / 700M</td><td>Direct screen interaction; Chain-of-action; Action histories and future plans</td></tr><tr><td>CogAgent [46] ?</td><td>2023.12</td><td>General</td><td>CogVLM</td><td>18B</td><td>High-res input (1120 × 1120); Specialized in GUI understanding</td></tr><tr><td>WebVLN-Net [104] ?</td><td>2023.12</td><td>Screen Understanding, QA</td><td>N/A</td><td>N/A</td><td>Web navigation with visual and HTML content</td></tr><tr><td>ScreenAI [67] ?</td><td>2024.02</td><td>Screen Understanding, QA</td><td>N/A</td><td>4.6B</td><td>UI and infographic understanding; Flexible patching</td></tr><tr><td>CoCo-Agent [71] ?</td><td>2024.02</td><td>General</td><td>LLaVA (LLaMA-2- chat-7B, CLIP)</td><td>N/A</td><td>Comprehensive perception; Conditional action prediction; Enhanced automation</td></tr><tr><td>Ferret-UI [101] ?</td><td>2024.04</td><td>Screen Understanding, Referring</td><td>Ferret</td><td>N/A</td><td>"Any resolution" tech-niques; Precise referring and grounding</td></tr><tr><td>LVG [94]</td><td>2024.06</td><td>Screen Understanding, Grounding</td><td>SWIN Transformer, BERT</td><td>N/A</td><td>Visual UI grounding; Layout-guided contrastive learning</td></tr><tr><td>Textual Foresight [103]</td><td>2024.06</td><td>Screen Understanding, Referring</td><td>BLIP-2</td><td>N/A</td><td>Predict UI state; UI representation learning</td></tr><tr><td>MobileFlow [88]</td><td>2024.07</td><td>General</td><td>Qwen-VL-Chat</td><td>21B</td><td>Hybrid visual encoders; Variable resolutions; Multilingual support</td></tr><tr><td>UI-Hawk [95]</td><td>2024.08</td><td>Screen Understanding, Grounding</td><td>N/A</td><td>N/A</td><td>History-aware encoder; Screen stream processing; FunUI benchmark</td></tr><tr><td>Ferret-UI 2 [102]</td><td>2024.10</td><td>Screen Understanding, Referring</td><td>Ferret</td><td>N/A</td><td>Multi-platform; High-resolution encoding</td></tr><tr><td>OS-Atlas [97] :</td><td>2024.10</td><td>Screen Understanding, Grounding</td><td>Owen2-VL, InternVL-2</td><td>4B / 7B</td><td>Grounding data synthesis; Largest GUI grounding corpus</td></tr><tr><td>ShowUI [89] :</td><td>2024.11</td><td>General</td><td>Qwen2-VL</td><td>2B</td><td>Visual tokens selection; Cross-modal understanding</td></tr><tr><td>Aguvis [90] :</td><td>2024.12</td><td>General</td><td>Qwen2-VL</td><td>7B / 72B</td><td>Comprehensive data pipeline; Two-stage training;Cross-platform</td></tr><tr><td>Aria-UI [96] :</td><td>2024.12</td><td>Screen Understanding, Grounding</td><td>Aria</td><td>3.9B</td><td>Diversified dataset pipeline; Multimodal dynamic action history</td></tr><tr><td>UI-TARS [92] ☒</td><td>2025.01</td><td>General</td><td>Qwen2-VL</td><td>2B / 7B / 72B</td><td>System-2 Reasoning; Online bootstrapping; Reflection tuning</td></tr><tr><td>GUI-Bee [100] ❒</td><td>2025.01</td><td>Screen Understanding, Grounding</td><td>SeeClick, UIX-7B, Qwen-GUI</td><td>N/A</td><td>Model-Environment alignment; Self-exploratory Data</td></tr><tr><td>V-Droid [114]</td><td>2025.03</td><td>General</td><td>Llama-3.1-8B</td><td>8b</td><td>Verifier-driven framework</td></tr><tr><td>MP-GUI [98] ❄</td><td>2025.03</td><td>General</td><td>InternVL2-8B</td><td>8B</td><td>Screen Understanding, Referring</td></tr></table>

<!-- Media -->


<!-- Meanless: 21 -->

tasks, which are essential for enabling AI agents to interact effectively with graphical user interfaces. These tasks can be categorized into three main types: UI grounding, UI referring, and screen question answering (QA). Figure 10 illustrates the differences between these categories.

- UI Grounding involves identifying and localizing UI elements on a screen that correspond to a given natural language description. This task is critical for agents to perform precise interactions with GUIs based on user instructions. MUG [93] proposes guiding agent actions through multi-round interactions with users, improving the execution accuracy of UI grounding in complex or ambiguous instruction scenarios. It also leverages user instructions and previous interaction history to predict the next agent action. LVG (Layout-guided Visual Grounding) [94] addresses UI grounding by unifying detection and grounding of UI elements within application interfaces. LVG tackles challenges such as application sensitivity, where UI elements with similar appearances have different functions across applications, and context sensitivity, where the functionality of UI elements depends on their context within the interface. By introducing layout-guided contrastive learning, LVG learns the semantics of UI objects from their visual organization and spatial relationships, improving grounding accuracy. UI-Hawk [95] enhances UI grounding by incorporating a history-aware visual encoder and an efficient resampler to process screen sequences during GUI navigation. By understanding historical screens, UI-Hawk improves the agent's ability to ground UI elements accurately over time. An automated data curation method generates training data for UI grounding, contributing to the creation of the FunUI benchmark for evaluating screen understanding capabilities. Aria-UI [96] leverages strong MLLMs such as GPT-40 to generate diverse and high-quality element instructions for grounding training. It employs a two-stage training method that incorporates action history in textual or interleaved text-image formats, enabling the model to develop both single-step localization capabilities and multi-step context awareness. This approach demonstrates robust performance and generalization ability across various tasks. Similar research includes GUI-Bee [100], which autonomously explores environments to collect high-quality data, thereby aligning GUI action grounding models with new environments and significantly enhancing model performance. OS-Atlas [97] unifies the action space, enabling models to adapt to UI grounding tasks across multiple platforms. Additionally, TAG (Tuning-free Attention-driven Grounding) [277] introduces a method that leverages the inherent attention mechanisms of pre-trained MLLMs to accurately identify and locate elements within a GUI without the need for tuning. Validation shows that this method performs comparably to or even surpasses tuned approaches across multiple benchmark datasets, demonstrating exceptional generalization capabilities. This offers a new perspective for the application of MLLMs in UI grounding.

- UI Referring focuses on generating natural language descriptions for specified UI elements on a screen. This task enables agents to explain UI components to users or other agents, facilitating better communication and interaction. Ferret-UI [101] is a multimodal LLM designed for enhanced understanding of mobile UI screens, emphasizing precise referring and grounding tasks. By incorporating any resolution techniques to handle various screen aspect ratios and dividing screens into sub-images for detailed analysis, Ferret-UI generates accurate descriptions of UI elements. Training on a curated dataset of elementary UI tasks, Ferret-UI demonstrates strong performance in UI referring tasks. Leveraging the Ferret-UI framework, Ferret-UI 2 [102] integrates an adaptive N-grid partitioning mechanism. This system enhances image feature extraction by dynamically resizing grids, thereby improving the model's efficiency and accuracy without sacrificing resolution. Additionally, Ferret-UI 2 demonstrates remarkable cross-platform portability. Textual Foresight [103] uses user actions as a bridge, requiring the model to predict the global textual description of the next UI state based on the current UI screen and a local action. With limited training data, the Textual Foresight method achieves superior performance compared to similar models, demonstrating exceptional data efficiency. UI-Hawk [95] also contributes to UI referring by defining tasks that require the agent to generate descriptions for UI elements based on their role and context within the interface. By processing screen sequences and understanding the temporal relationships between screens, UI-Hawk improves the agent's ability to refer to UI elements accurately.


<!-- Meanless: 22 -->

<!-- Media -->

<!-- figureText: 1 UI Grounding<br>2 Screen QA<br>3 GUI Navigation<br>4 UI Referring<br>Q Zyunlp / LLMAgentPapers Public<br>11:43 00<br><icon/><br>1:44<br><icon/><br>合<br>③<br>$\vdots$<br>google.com/sϵ<br>③<br>Open this paper.<br>Must-read Papers on LLM Agents.<br>4<br>Tap[969, 1049]<br>2 2k stars $\$$ 106 forks<br>Q<br>Ilm agent paper<br>✘<br>Paranches Q Tags Y Activity<br>1:45 * 00<br>MOTI<br>Star<br>A Notifications<br>Search or ty...<br>Ⓡ<br>All<br>Images<br>Shopping<br>Videos<br>Shor<br>③<br>$< >$ Code<br><icon/><br>...<br>25 github.com/zju<br>口<br>GitHub<br>https://github.com/zjun1p > LLM...<br>口<br>Sign in<br>P<br>main<br>Codes<br>...<br>9<br>zjun1p/LLMAge.ltPapers: Must-read Papers or I<br>百月<br>We sincerely invite you to dive into these collections of papers and resources, each offering a distinct journey of exploration and...<br>口 zjunlp / LLMAgentPapers<br>Rolnand last month<br>⑨<br>Public<br>口<br>DS_Store<br>2 months ago<br>Must-read Papers on LLM A<br>On this page, to click on the search bar, where should I navigate to?<br>arXiv<br>2 Stars<br>README.md<br>last month<br>https://arxiv.org<br>Parancines Dese- Activity<br>[488, 658, 1448, 866]<br>The Rise and Potential of Large Language Model Based Agents<br>Star<br>A Notifications<br>CO README<br>Crazies<br>by ${ZX}$ to2023. Cited by 617 - In this paper, we perform a comprehensive survey on LLM-based agents. We start by tracing the conce.<br>< > Code<br>Sussues<br>...<br>Google<br>11:44<br>Search or ty.<br>эти<br>What's the title of the first link in this page?<br>What is function of this area[475, 495, 1254, 632]?<br>Ilm agent paper<br>/<br>Rolnand last month<br>Э<br>Ilm agent paper<br>The title is "ziunlp/LLMAgentPapersMust - read Papers on LLM Agents."<br>DS_Store<br>2 months ago<br>Stars show likes, forks versions, tags for releases, activity for updates. -->

<img src="https://cdn.noedgeai.com/bo_d41gln3ef24c73d3u2c0_21.jpg?x=180&y=123&w=1423&h=845&r=0"/>

Fig. 10: Illustration of screen understanding tasks. (a) UI Grounding involves identifying UI elements corresponding to a given description; (b) UI Referring focuses on generating descriptions for specified UI elements; (c) Screen Question Answering requires answering questions based on the content of the screen.

<!-- Media -->

- Screen Question Answering involves answering questions about the content and functionality of a screen based on visual and textual information. This task requires agents to comprehend complex screen layouts and extract relevant information to provide accurate answers. ScreenAI [67] specializes in understanding screen UIs and infographics, leveraging the common visual language and design principles shared between them. By introducing a flexible patching strategy and a novel textual representation for UIs, ScreenAI pre-trains models to interpret UI elements effectively. Using large language models to automatically generate training data, ScreenAI covers tasks such as screen annotation and screen QA. WebVLN [104] extends vision-and-language navigation to websites, where agents navigate based on question-based instructions and answer questions using information extracted from target web pages. By integrating visual inputs, linguistic instructions, and web-specific content like HTML, WebVLN enables agents to understand both the visual layout and underlying structure of web pages, enhancing screen QA capabilities. UI-Hawk [95] further enhances screen QA by enabling agents to process screen sequences and answer questions based on historical interactions. By incorporating screen question answering as one of its fundamental tasks, UI-Hawk improves the agent's ability to comprehend and reason about screen content over time. MP-GUI [98] introduces a specialized MLLM for GUI understanding with three dedicated perceivers for graphical, textual, and spatial modalities. Using a fusion gate to adaptively combine these modalities and an automated data collection pipeline to address training data scarcity, MP-GUI achieves strong performance on GUI understanding tasks including screen QA despite limited training data.


<!-- Meanless: 23 -->

These Phone UI-Specific Model Model Architectures demonstrate the importance of focusing on screen understanding tasks to enhance AI agents' interaction with complex user interfaces. By categorizing these tasks into UI grounding, UI referring, and screen question answering, researchers have developed specialized models that address the unique challenges within each category. Integrating innovative techniques such as layout-guided contrastive learning, history-aware visual encoding, and flexible patching strategies has led to significant advancements in agents' abilities to understand, navigate, and interact with GUIs effectively.

#### 4.2.2 Supervised Fine-Tuning

Supervised fine-tuning has emerged as a crucial technique for enhancing the capabilities of LLMs in GUI tasks within phone automation. By tailoring models to specific tasks through fine-tuning on curated datasets, researchers have significantly improved models' abilities in GUI grounding, optical character recognition (OCR), cross-application navigation, and efficiency. A summary of notable works in this area is presented in Table 4, highlighting their main contributions, domains, and other relevant details.

Supervised fine-tuning has been effectively applied to develop more versatile and efficient GUI agents by enhancing their fundamental abilities and GUI knowledge. One of the fundamental challenges in developing visual GUI agents is enabling accurate interaction with screen elements based solely on visual inputs, known as GUI grounding. SeeClick [105] addresses this challenge by introducing a visual GUI agent that relies exclusively on screenshots for task automation, circumventing the need for extracted structured data like HTML, which can be lengthy and sometimes inaccessible. Recognizing that GUI grounding is a key hurdle, SeeClick enhances the agent's capability by incorporating GUI grounding pre-training. The authors also introduce ScreenSpot, the first realistic GUI grounding benchmark encompassing mobile, desktop, and web environment. Experimental results demonstrate that improving GUI grounding through supervised fine-tuning directly correlates with enhanced performance in downstream GUI tasks. InfiGUIAgent [106] is trained using a supervised fine-tuning method and employs the Reference-Augmented Annotation approach to fully leverage spatial information, establishing bidirectional connections between GUI elements and text descriptions, thereby enhancing the model's understanding of GUI visual language. Additionally, the model incorporates Hierarchical Reasoning and Expectation-Reflection Reasoning capabilities, enabling the agent to perform complex reasoning natively, which improves its grounding ability. Beyond grounding, agents require robust OCR capabilities and comprehensive knowledge of GUI components and interactions to function effectively across diverse applications. GUICourse [107] tackles these challenges by presenting a suite of datasets designed to train visual-based GUI agents from general VLMs. The GUIEnv dataset strengthens OCR and grounding abilities by providing 10 million website page-annotation pairs for pre-training and 0.7 million region-text QA pairs for supervised fine-tuning. To enrich the agent's understanding of GUI components and interactions, the GUIAct and GUIChat datasets offer extensive single-step and multi-step action instructions and conversational data with text-rich images and bounding boxes. As users frequently navigate across multiple applications to complete complex tasks, enabling cross-app GUI navigation becomes essential for practical GUI agents.GUI Odyssey [109] addresses this need by introducing a comprehensive dataset specifically designed for training and evaluating cross-app navigation agents. The GUI Odyssey dataset comprises 7,735 episodes from six mobile devices, covering six types of cross-app tasks, 201 apps, and 1,399 app combinations. By fine-tuning the Qwen-VL model with a history resampling module on this dataset, they developed OdysseyAgent, a multimodal cross-app navigation agent. Extensive experiments show that OdysseyAgent achieves superior accuracy compared to existing models, significantly improving both in-domain and out-of-domain performance on cross-app navigation tasks. Efficiency and scalability are also critical considerations, especially for deploying GUI agents on devices with limited computational resources. TinyClick [110] demonstrates that even compact models can achieve strong performance on GUI automation tasks through effective supervised fine-tuning strategies. Utilizing the Vision-Language Model Florence-2- Base, TinyClick focuses on the primary task of identifying the screen coordinates of UI elements corresponding to user commands. By employing multi-task training and Multimodal Large Language Model-based data augmentation, TinyClick significantly improves model performance while maintaining a compact size of 0.27 billion parameters and minimal latency. MobileAgent [111] combines LoRA and SOP methods to effectively reduce computational overhead through low-rank adaptive supervised fine-tuning, while breaking down complex tasks into subtasks to enhance the model's understanding and execution efficiency. At the same time, this approach does not impose additional burdens on inference speed, significantly improving the model's performance and responsiveness. The performance of agents is often limited by their inability to recover from errors. Agent-R [108] identifies the first error step in an erroneous trajectory and combines it with a correct trajectory to create a corrected path, thus enabling real-time error correction. By training on self-generated corrected trajectories and using an iterative supervised fine-tuning approach, Agent-R dynamically identifies and rectifies errors, gradually enhancing decision-making abilities. Moreover, under a multi-task training strategy, its training outcomes improve significantly. This method offers new directions for developing more intelligent and adaptable GUI agents. Supervised fine-tuning has also been applied to domain-

specific tasks to address specialized challenges in particular contexts, such as reference resolution and accessibility. In the context of Reference Resolution in GUI Contexts, ReALM [112] formulates reference resolution as a language modeling problem, enabling the model to handle various types of references, including on-screen entities, conversational entities, and background entities. By converting reference resolution into a multiple-choice task for the LLM, ReALM significantly improves the model's ability to resolve references in GUI contexts. For Accessibility and UI Icons Alt-Text Generation, IconDesc [115] addresses the challenge of generating informative alt-text for mobile UI icons, which is essential for users relying on screen readers. Traditional deep learning approaches require extensive datasets and struggle with the diversity and imbalance of icon types. IconDesc introduces a novel method using Large Language Models to autonomously generate alt-text with partial UI data, such as class, resource ID, bounds, and contextual information from parent and sibling nodes. By fine-tuning an off-the-shelf LLM on a small dataset of approximately 1.4k icons, IconDesc demonstrates significant improvements in generating relevant alt-text, aiding developers in enhancing UI accessibility during app development.


<!-- Meanless: 24 -->

<!-- Media -->

TABLE 4: Summary of supervised fine-tuning methods for phone GUI agents

<table><tr><td>Method</td><td>Date</td><td>Task Type</td><td>Backbone</td><td>Size</td><td>Contributions</td></tr><tr><td>MobileAgent [111] ?</td><td>2024.01</td><td>General</td><td>Qwen</td><td>7B</td><td>Standard Operating Procedure; Human-machine interaction</td></tr><tr><td>SeeClick [105] ?</td><td>2024.01</td><td>General</td><td>Qwen-VL</td><td>9.6B</td><td>GUI grounding pre-training; ScreenSpot benchmark</td></tr><tr><td>ReALM [112]</td><td>2024.04</td><td>Reference Resolution</td><td>FLAN-T5</td><td>80M-3B</td><td>Formulated reference resolution as language modeling; Improved performance on resolving references</td></tr><tr><td>GUICourse [107] ?</td><td>2024.06</td><td>General</td><td>Qwen-VL, Fuyu-8B, MiniCPM-V</td><td>N/A</td><td>Suite of datasets (GUIEnv, GUIAct, GUIChat); Enhanced OCR and grounding</td></tr><tr><td>GUI Odyssey [109] ?</td><td>2024.06</td><td>General</td><td>Qwen-VL</td><td>N/A</td><td>Cross-app navigation dataset; Agent with history resampling</td></tr><tr><td>IconDesc [115]</td><td>2024.09</td><td>Alt-Text Generation</td><td>GPT-3.5</td><td>N/A</td><td>Generated alt-text for UI icons using partial UI data; Improved accessibility</td></tr><tr><td>TinyClick [110] ?</td><td>2024.10</td><td>General</td><td>Florence-2</td><td>0.27B</td><td>Single-turn agent; Multitask training; MLLM-based data augmentation</td></tr><tr><td>InfiGUIAgent [106] ?</td><td>2025.01</td><td>General</td><td>Qwen2-VL</td><td>2B</td><td>Model-Environment alignment; Self-exploratory Data</td></tr><tr><td>Agent-R [108] ?</td><td>2025.01</td><td>General</td><td>LLama-3.1</td><td>8B</td><td>Self-reflection capabilities; Real-time error correction</td></tr></table>

<!-- Media -->

These works collectively demonstrate that supervised fine-tuning is instrumental in advancing GUI agents for phone automation. By addressing specific challenges through targeted datasets and training strategies—whether enhancing GUI grounding, improving OCR and GUI knowledge, enabling cross-app navigation, or optimizing for accessibility-researchers have significantly enhanced the performance and applicability of GUI agents. The advancements summarized in Figure 4 highlight the ongoing efforts and progress in this field, paving the way for more intelligent, versatile, and accessible phone automation solutions capable of handling complex tasks in diverse environment.

#### 4.2.3 Reinforcement Learning

Reinforcement Learning (RL) [278] has emerged as a powerful technique for training agents to interact autonomously with GUIs across various platforms, including phones, web browsers, and desktop environment. Although RL-based approaches for phone GUI agents are relatively few, significant progress has been made in leveraging RL to enhance agent capabilities in dynamic and complex GUI environment. In this section, we discuss RL approaches for GUI agents across different platforms, highlighting their unique challenges, methodologies, and contributions. A summary of notable RL-based methods is presented in Figure 5, which includes specific RL-related features such as the type of RL used (online or offline) and the targeted platform.

Phone Agents. Training phone GUI agents using RL presents unique challenges due to the dynamic and complex nature of mobile applications. Agents must adapt to real-world stochasticity and handle the intricacies of interacting with diverse mobile environment. Recent works have addressed these challenges by developing RL frameworks that enable agents to learn from interactions and improve over time. DigiRL [116] andDistRL [117] both tackle the limitations of pre-trained vision-language models (VLMs) in decision-making tasks for device control through GUIs. Recognizing that static demonstrations are insufficient due to the dynamic nature of real-world mobile environment, these works introduce RL approaches to train agents capable of in-the-wild device control. DigiRL proposes an autonomous RL framework that employs a two-stage training process: an initial offline RL phase to initialize the agent using existing data, followed by an offline-to-online RL phase that fine-tunes the model based on its own interactions. By building a scalable Android learning environment with a VLM-based evaluator, DigiRL identifies key design choices for effective RL in mobile GUI domains. The agent learns to handle real-world stochasticity and dynamism, achieving significant improvements over supervised fine-tuning, with a 49.5% absolute increase in success rate on the Android-in-the-Wild dataset. Similarly, DistRL introduces an asynchronous distributed RL framework specifically designed for on-device control agents on mobile devices. To address inefficiencies in online fine-tuning and the challenges posed by dynamic mobile environment, DistRL employs centralized training and decentralized data acquisition. Leveraging an off-policy RL algorithm tailored for distributed and asynchronous data utilization, DistRL improves training efficiency and agent performance by prioritizing significant experiences and encouraging exploration. Experiments show that DistRL achieves a 20% relative improvement in success rate compared to state-of-the-art methods on general Android tasks. Building upon these advancements, AutoGLM [118] extends the application of RL to both phone and web platforms. AutoGLM presents a series of foundation agents based on the ChatGLM model family, aiming to serve as autonomous agents for GUI control. A key insight from this work is the design of an intermediate interface that separates planning and grounding behaviors, allowing for more agile development and enhanced performance. By employing self-evolving online curriculum RL, AutoGLM enables agents to learn from environmental interactions and adapt to dynamic GUI environment. The approach demonstrates impressive success rates on various benchmarks, showcasing the potential of RL in creating versatile GUI agents across platforms.


<!-- Meanless: 25 -->

<!-- Media -->

TABLE 5: Summary of reinforcement learning methods for phone GUI agents

<table><tr><td>Method</td><td>Date</td><td>Platform</td><td>RL Type</td><td>Backbone</td><td>Size</td></tr><tr><td>DigiRL [116] ❄</td><td>2024.06</td><td>Phone</td><td>Online RL</td><td>AutoUI-Base</td><td>200M</td></tr><tr><td>DistRL [117] ❗</td><td>2024.10</td><td>Phone</td><td>Online RL</td><td>T5-based</td><td>1.3B</td></tr><tr><td>AutoGLM [118] ⤵</td><td>2024.11</td><td>Phone, Web</td><td>Online RL</td><td>GLM-4-9B-Base</td><td>9B</td></tr><tr><td>ScreenAgent [128] ⤵</td><td>2024.02</td><td>PC OS</td><td>N/A</td><td>CogAgent</td><td>18B</td></tr><tr><td>ETO [124] ⤵</td><td>2024.03</td><td>Web</td><td>Offline-to-Online RL</td><td>LLaMA-2-7B-Chat</td><td>7B</td></tr><tr><td>AutoWebGLM [126] ⤵</td><td>2024.04</td><td>Web</td><td>RL (Curriculum Learning, Bootstrapped RL)</td><td>ChatGLM3-6B</td><td>6B</td></tr><tr><td>Agent Q [125] ⤵</td><td>2024.08</td><td>Web</td><td>Offline RL with MCTS</td><td>LLaMA-3-70B</td><td>70B</td></tr><tr><td>GLAINTEL [127] ⚗</td><td>2024.11</td><td>Web</td><td>RL (Offline-to-Online, Hybrid RL)</td><td>Flan-T5</td><td>0.78B</td></tr><tr><td>ReachAgent [120]</td><td>2025.02</td><td>Phone</td><td>Hybrid RL</td><td>MobileVLM [63]</td><td>N/A</td></tr><tr><td>VEM [279] ?</td><td>2025.02</td><td>Phone</td><td>Environment-Free RL</td><td>N/A</td><td>N/A</td></tr><tr><td>Digi-Q [119] ?</td><td>2025.02</td><td>Phone</td><td>O-Function Based RL</td><td>N/A</td><td>N/A</td></tr><tr><td>VSC-RL [121] ?</td><td>2025.02</td><td>Phone</td><td>Variational Subgoal-Conditioned RL</td><td>N/A</td><td>N/A</td></tr><tr><td>UI-R1 [122] ☒</td><td>2025.03</td><td>Phone</td><td>Rule-Based RL</td><td>Qwen2.5-VL</td><td>3B</td></tr></table>

<!-- Media -->

Recent advances have brought several innovative approaches to reinforcement learning for phone GUI agents. ReachAgent [120] decomposes mobile agent tasks into two sub-tasks: page reaching and page operation, utilizing a two-stage fine-tuning strategy. In the first stage, supervised fine-tuning enables the agent to better perform each subtask. In the second stage, reinforcement learning is applied to further optimize the agent's overall task completion capabilities, thereby enhancing its performance in complex tasks. VEM [279] introduces an environment-free RL framework that decouples value estimation from policy optimization using a pretrained Value Environment Model. Unlike traditional RL methods that require costly environment interactions, VEM predicts state-action values directly from offline data, distilling human-like priors about GUI interaction outcomes. This approach avoids compounding errors and enhances resilience to UI changes by focusing on semantic reasoning. Digi-Q [119] presents an approach to train VLM-based action-value Q-functions for device control. Instead of using on-policy RL with actual environment rollouts, Digi-Q trains the Q-function using offline temporal-difference learning on frozen, intermediate-layer features of a VLM. This approach enhances scalability and reduces computational costs compared to fine-tuning the entire VLM. The trained Q-function then uses a Best-of-N policy extraction operator to imitate the best action without requiring environment interaction. VSC-RL [121] addresses the learning inefficiencies in tackling complex sequential decision-making tasks with sparse rewards and long-horizon dependencies. By reformulating vision-language sequential tasks as a variational goal-conditioned RL problem, VSC-RL optimizes the SubGoal Evidence Lower BOund (SGC-ELBO). This approach maximizes subgoal-conditioned return via RL while minimizing the difference with the reference policy. UI-R1 [122] explores how rule-based RL can enhance reasoning capabilities of multimodal large language models for GUI action prediction. Using a small yet high-quality dataset of 136 challenging tasks, UI-R1 introduces a unified rule-based action reward enabling model optimization via Group Relative Policy Optimization (GRPO).


<!-- Meanless: 26 -->

Web Agents. Web navigation tasks involve interacting with complex and dynamic web environment, where agents must interpret web content and perform actions to achieve user-specified goals. RL has been employed to train agents that can adapt to these challenges by learning from interactions and improving decision-making capabilities. ETO [124] (Exploration-based Trajectory Optimization) and Agent Q [125] both focus on enhancing the performance of LLM-based agents in web environment through RL techniques. ETO introduces a learning method that allows agents to learn from their exploration failures by iteratively collecting failure trajectories and using them to create contrastive trajectory pairs for training. By leveraging contrastive learning methods like Direct Preference Optimization (DPO), ETO enables agents to improve performance through an iterative cycle of exploration and training. Experiments on tasks such as WebShop demonstrate that ETO consistently outperforms baselines, highlighting the effectiveness of learning from failures. Agent Q combines guided Monte Carlo Tree Search (MCTS) with a self-critique mechanism and iterative fine-tuning using an off-policy variant of DPO. This framework allows LLM agents to learn from both successful and unsuccessful trajectories, improving generalization in complex, multi-step reasoning tasks. Evaluations on the WebShop environment and real-world booking scenarios show that Agent Q significantly improves success rates, outperforming behavior cloning and reinforcement learning fine-tuned baselines. AutoWebGLM [126] contributes to this domain by developing an LLM-based web navigating agent built upon ChatGLM3-6B. To address the complexity of HTML data and the versatility of web actions, AutoWebGLM introduces an HTML simplification algorithm to represent webpages succinctly. The agent is trained using a hybrid human-AI method to build web browsing data for curriculum training and is further enhanced through reinforcement learning and rejection sampling. AutoWebGLM demonstrates performance superiority on general webpage browsing tasks, achieving practical usability in real-world services. GLAIN-TEL [127] effectively utilizes human experience and the adaptive capabilities of reinforcement learning by integrating human demonstrations with reinforcement learning methods. This approach achieves superior performance in complex product search tasks. Collectively, these works demonstrate how RL techniques can be applied to web agents to improve their ability to navigate and interact with complex web environment. By learning from interactions, failures, and leveraging advanced planning methods, these agents exhibit enhanced reasoning and decision-making capabilities.

PC OS Agents. In desktop environment, agents face the challenge of interacting with complex software applications and operating systems, requiring precise control actions and understanding of GUI elements. RL approaches in this domain focus on enabling agents to perform multistep tasks and adapt to the intricacies of desktop GUIs. ScreenAgent [128] constructs an environment where a Vision Language Model (VLM) agent interacts with a real computer screen via the VNC protocol. By observing screenshots and manipulating the GUI through mouse and keyboard actions, the agent operates within an automated control pipeline that includes planning, acting, and reflecting phases. This design allows the agent to continuously interact with the environment and complete multi-step tasks. ScreenAgent introduces the ScreenAgent Dataset, which collects screen-shots and action sequences for various daily computer tasks. The trained model demonstrates computer control capabilities comparable to GPT-4V and exhibits precise UI positioning capabilities, highlighting the potential of RL in desktop GUI automation. AssistGUI [129] develops an LLM-based reinforcement learning framework called Actor-Critic Embodied Agent (ACE). This framework automates desktop GUI through visual analysis, reasoning, and action generation, significantly improving task success rates. Additionally, it introduces a novel benchmarking framework to evaluate a model's ability to complete complex tasks on desktop platforms using mouse and keyboard operations. This advancement offers a new direction for future research in desktop GUI automation.

Reinforcement Learning has proven to be a valuable approach for training GUI agents across various platforms, enabling them to learn from interactions with dynamic environment and improve their performance over time. By leveraging RL techniques, these agents can adapt to real-world stochasticity, handle complex decision-making tasks, and exhibit enhanced autonomy in phone, web, and desktop environment. The works discussed in this section showcase the progress made in developing intelligent and versatile GUI agents through RL, paving the way for enhanced automation and user interaction across diverse platforms.


<!-- Meanless: 27 -->

## 5 DATASETS AND BENCHMARKS

The rapid evolution of mobile technology has transformed smartphones into indispensable tools for communication, productivity, and entertainment. This shift has spurred a growing interest in developing intelligent agents capable of automating tasks and enhancing user interactions with mobile devices. These agents rely on a deep understanding of GUIs and the ability to interpret and execute instructions effectively. However, the development of such agents presents significant challenges, including the need for diverse datasets, standardized benchmarks, and robust evaluation methodologies.

Datasets serve as the backbone for training and testing phone GUI agents, offering rich annotations and task diversity to enable these agents to learn and adapt to complex environment. Complementing these datasets, benchmarks provide structured environment and evaluation metrics, allowing researchers to assess agent performance in a consistent and reproducible manner. Together, datasets and benchmarks form the foundation for advancing the capabilities of GUI-based agents.

This section delves into the key datasets and benchmarks that have shaped the field. Subsection 5.1 reviews notable datasets that provide the training data necessary for enabling agents to perform tasks such as language grounding, UI navigation, and multimodal interaction. Subsection 5.2 discusses benchmarks that facilitate the evaluation of agent performance, focusing on their contributions to reproducibility, generalization, and scalability. Through these resources, researchers and developers gain the tools needed to push the boundaries of intelligent phone automation, moving closer to creating agents that can seamlessly assist users in their daily lives.

### 5.1 Datasets

The development of phone automation and GUI-based agents has been significantly propelled by the availability of diverse and richly annotated datasets. These datasets provide the foundation for training and evaluating models that can understand and interact with mobile user interfaces using natural language instructions. In this subsection, we review several key datasets, highlighting their unique contributions and how they collectively advance the field. Table 6 summarizes these datasets, providing an overview of their characteristics.

Rico [130] is the largest dataset from the early stage of GUI automation development, providing a solid foundation for understanding modern mobile interfaces and developing GUI agents. It includes various types of data, such as UI screenshots, view hierarchies, and UI metadata, offering valuable references for researchers and developers. Based on this, subsequent studies like RICO Semantics [131], GUI-WORLD [139], and MobileViews [142] have emerged, expanding the types and coverage of datasets and driving the growth of GUI agent research. Among them, MobileViews is currently the largest GUI dataset.

Early efforts in dataset creation focused on mapping natural language instructions to UI actions. PixelHelp [132] pioneered this area by introducing a problem of grounding natural language instructions to mobile UI action sequences. It decomposed the task into action phrase extraction and grounding, enabling models to interpret instructions like "Turn on flight mode" and execute corresponding UI actions. Building on this, UGIF [136] extended the challenge to a multilingual and multimodal setting. UGIF addressed cross-modal and cross-lingual retrieval and grounding, providing a dataset with instructions in English and UI interactions across multiple languages, thus highlighting the complexities of multilingual UI instruction following.

Addressing task feasibility and uncertainty, MoTIF [133] introduced a dataset that includes natural language commands which may not be satisfiable within the given UI context. By incorporating feasibility annotations and followup questions, MoTIF encourages research into how agents can recognize and handle infeasible tasks, enhancing robustness in interactive environment.

For advancing UI understanding through pre-training, UIBert [134] proposed a Transformer-based model that jointly learns from image and text representations of UIs. By introducing novel pre-training tasks that leverage the correspondence between different UI features, UIBert demonstrated improvements across multiple downstream UI tasks, setting a foundation for models that require a deep understanding of GUI layouts and components.

In the realm of multimodal dialogues and interactions, Meta-GUI [135] proposed a GUI-based task-oriented dialogue system. This work collected dialogues paired with GUI operation traces, enabling agents to perform tasks through conversational interactions and direct GUI manipulations. It bridges the gap between language understanding and action execution within mobile applications.

Recognizing the need for large-scale datasets to train more generalizable agents, several works introduced extensive datasets capturing a wide range of device interactions. Android In The Wild (AITW) [137] released a dataset containing hundreds of thousands of episodes with human demonstrations of device interactions. It presents challenges where agents must infer actions from visual appearances and handle precise gestures. Building upon AITW, Android In The Zoo (AITZ) [138] provided fine-grained semantic annotations using the Chain-of-Action-Thought (CoAT) paradigm, enhancing agents' ability to reason and make decisions in GUI navigation tasks.

To address the complexities of cross-application navigation, GUI Odyssey [109] introduced a dataset specifically designed for training and evaluating agents that navigate across multiple apps. By covering diverse apps, tasks, and devices, GUI Odyssey enables the development of agents capable of handling real-world scenarios that involve integrating multiple applications and transferring context between them.

Understanding how data scale affects agent performance, AndroidControl [140] studied the impact of training data size on computer control agents. By collecting demonstrations with both high-level and low-level instructions across numerous apps, this work analyzed in-domain and out-of-domain generalization, providing insights into the scalability of fine-tuning approaches for device control agents.

Focusing on detailed annotations to enhance agents' understanding of UI elements, AMEX [141] introduced a comprehensive dataset with multi-level annotations. It includes GUI interactive element grounding, functionality descriptions, and complex natural language instructions with stepwise GUI-action chains. AMEX aims to align agents more closely with human users by providing fundamental knowledge and understanding of the mobile GUI environment from multiple levels, thus facilitating the training of agents with a deeper understanding of page layouts and UI element functionalities.


<!-- Meanless: 28 -->

<!-- Media -->

TABLE 6: Summary of datasets for phone GUI agents. "Actions" refers to the number of distinct actions available; "Demos" refers to the number of demonstration sequences; "Apps" refers to the number of applications covered; "Instr." refers to the number of natural language instructions; "Avg. Steps" refers to the average number of steps per task.

<table><tr><td>Dataset</td><td>Date</td><td>Screenshots</td><td>UI Trees</td><td>Actions</td><td>Demos</td><td>Apps</td><td>Instr.</td><td>Avg. Steps</td><td>Contributions</td></tr><tr><td>Rico [130] ?</td><td>2017.10</td><td>✓</td><td>✓</td><td>N/A</td><td>10,811</td><td>9,772</td><td>N/A</td><td>N/A</td><td>Large-scale mobile dataset</td></tr><tr><td>PixelHelp [132] ?</td><td>2020.05</td><td>✓</td><td>✓</td><td>4</td><td>187</td><td>4</td><td>187</td><td>4.2</td><td>Grounding instruc tions to actions</td></tr><tr><td>MoTIF [133] ?</td><td>2021.04</td><td>✓</td><td>✓</td><td>6</td><td>4,707</td><td>125</td><td>276</td><td>4.5</td><td>Interactive visual environment</td></tr><tr><td>UIBert [134] ?</td><td>2021.07</td><td>✓</td><td>✓</td><td>N/A</td><td>N/A</td><td>N/A</td><td>16,660</td><td>1</td><td>Pre-training task</td></tr><tr><td>Meta-GUI [135] ?</td><td>2022.05</td><td>✘</td><td>✓</td><td>7</td><td>4,684</td><td>11</td><td>1,125</td><td>5.3</td><td>Multi-turn dialogues</td></tr><tr><td>UGIF [136] ?</td><td>2022.11</td><td>✓</td><td>✓</td><td>8</td><td>523</td><td>12</td><td>523</td><td>5.3</td><td>Multilingual UI-grounded instructions</td></tr><tr><td>AITW [137] ?</td><td>2023.12</td><td>✓</td><td>✘</td><td>7</td><td>715,142</td><td>357</td><td>30,378</td><td>6.5</td><td>Large-scale interactions</td></tr><tr><td>AITZ [138] ☒</td><td>2024.03</td><td>✓</td><td>✘</td><td>7</td><td>18,643</td><td>70</td><td>2,504</td><td>7.5</td><td>Chain-of-Action-Thought annotations</td></tr><tr><td>GUI Odyssey [109] ☑</td><td>2024.06</td><td>✘</td><td>✓</td><td>9</td><td>7,735</td><td>201</td><td>7,735</td><td>15.4</td><td>Cross-app navigation</td></tr><tr><td>AndroidControl [140] :</td><td>2024.07</td><td>✓</td><td>✓</td><td>8</td><td>15,283</td><td>833</td><td>15,283</td><td>4.8</td><td>UI task scaling law</td></tr><tr><td>AMEX [141] ⤵</td><td>2024.07</td><td>✓</td><td>✓</td><td>8</td><td>2,946</td><td>110</td><td>2,946</td><td>12.8</td><td>Multi-level detailed annotations</td></tr><tr><td>MobileViews [142] ⤵</td><td>2024.09</td><td>✓</td><td>✓</td><td>N/A</td><td>N/A</td><td>21,053</td><td>N/A</td><td>N/A</td><td>Largest-scale mobile dataset</td></tr></table>

<!-- Media -->

Finally, we should focus on methods for generating, collecting, and annotating high-quality datasets. Dream-Struct [280] leverages LLMs to generate data design concept descriptions based on target tasks. It then produces HTML code with target labels, embedding semantic tags within. In the post-processing phase, Bing Search API or DALL-E is used to replace placeholder graphic elements, resulting in the final visual content. This research offers a dataset, DreamUI, which includes 9,774 labeled UI interfaces for reference. OS-Genesis [281] utilizes the method of Reverse Task Synthesis to automatically generate task instructions and corresponding action trajectories from interactions. It then integrates these with a trajectory reward model to produce high-quality and diverse GUI agent data. Learn-by-interact [282] uses LLMs to generate data through interaction with the environment and optimizes this data via backward construction. These high-quality data generation techniques reduce the dependency on manually labeled data, facilitating agents' rapid adaptation to new environments and tasks. Ferret-UI 2 [102] uses the Set-of-Mark (SoM) visual prompt method to tag each UI component with bounding boxes and numerical labels to assist GPT-40 in recognition. Subsequently, GPT-40 generates question-and-answer task data related to UI components, covering multiple aspects of UI comprehension and thus producing high-quality training data. FedMobileAgent [283] automatically collects data during users' daily mobile usage and employs locally deployed VLM to annotate user actions, thereby generating a high-quality dataset. Furthermore, even in the absence of explicit ground truth annotations, we can infer user intentions through their interactions within the GUI to generate corresponding UI annotations [284]. This approach opens up new directions for the collection and annotation of GUI data.

Collectively, these datasets represent significant strides in advancing phone automation and GUI-based agent research. They address various challenges, from language grounding and task feasibility to large-scale device control and cross-app navigation. By providing rich annotations and diverse scenarios, they enable the training and evaluation of more capable, robust, and generalizable agents, moving closer to the goal of intelligent and autonomous phone automation solutions.

### 5.2 Benchmarks

The development of mobile GUI-based agents is not only reliant on the availability of diverse datasets but is also significantly influenced by the presence of robust benchmarks. These benchmarks offer standardized environment, tasks, and evaluation metrics, which are essential for consistently and reproducibly assessing the performance of agents. They enable researchers to compare different models and approaches under identical conditions, thus facilitating collaborative progress. In this subsection, we will review some of the notable benchmarks that have been introduced to evaluate phone GUI agents, highlighting their unique features and contributions. A summary of these benchmarks is provided in Table 7, which allows for a comparative understanding of their characteristics.


<!-- Meanless: 29 -->

<!-- Media -->

TABLE 7: Summary of benchmarks for phone GUI agents

<table><tr><td>Benchmark</td><td>Date</td><td>Tasks</td><td>Task Completion</td><td>Action Quality</td><td>Resource Efficiency</td><td>Task Understanding</td><td>Format Compliance</td><td>Completion Awareness</td><td>$\mathbf{{Reward}}$</td><td>Eval Accuracy</td></tr><tr><td>MobileEnv [143] ☑</td><td>2023.05</td><td>74</td><td>✓</td><td>✘</td><td>✘</td><td>✘</td><td>✘</td><td>✘</td><td>✓</td><td>✘</td></tr><tr><td>AutoDroid [50] ☒</td><td>2023.09</td><td>N/A</td><td>✓</td><td>✓</td><td>✘</td><td>✘</td><td>✘</td><td>✘</td><td>✘</td><td>✘</td></tr><tr><td>AndroidArena [144] ?</td><td>2024.02</td><td>N/A</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td>✘</td></tr><tr><td>LlamaTouch [145] ?</td><td>2024.04</td><td>496</td><td>✓</td><td>✓</td><td>✘</td><td>✓</td><td>✘</td><td>✓</td><td>✘</td><td>✓</td></tr><tr><td>B-MoCA [146] ?</td><td>2024.04</td><td>131</td><td>✓</td><td>✘</td><td>✓</td><td>✘</td><td>✘</td><td>✘</td><td>✘</td><td>✘</td></tr><tr><td>AndroidWorld [147] ?</td><td>2024.05</td><td>116</td><td>✓</td><td>✘</td><td>✘</td><td>✘</td><td>✘</td><td>✘</td><td>✓</td><td>✘</td></tr><tr><td>MobileAgent Bench [151]</td><td>2024.06</td><td>100</td><td>✓</td><td>✓</td><td>✓</td><td>✘</td><td>✘</td><td>✘</td><td>✓</td><td>✓</td></tr><tr><td>AUITestAgent [148] ?</td><td>2024.07</td><td>N/A</td><td>✓</td><td>✓</td><td>✘</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td></tr><tr><td>VisualAgent Bench [152] ?</td><td>2024.08</td><td>119</td><td>✓</td><td>✘</td><td>✓</td><td>✘</td><td>✘</td><td>✘</td><td>✘</td><td>✘</td></tr><tr><td>AgentStudio [149] ?</td><td>2024.10</td><td>205</td><td>✓</td><td>✓</td><td>✘</td><td>✓</td><td>✘</td><td>✓</td><td>✓</td><td>✓</td></tr><tr><td>AndroidLab [150] ?</td><td>2024.11</td><td>138</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td>✘</td><td>✘</td><td>✓</td><td>✓</td></tr><tr><td>A3 [156] ?</td><td>2025.01</td><td>201</td><td>✓</td><td>✘</td><td>✘</td><td>✘</td><td>✘</td><td>✘</td><td>✓</td><td>✓</td></tr><tr><td>AutoEval [154]</td><td>2025.03</td><td>93</td><td>✓</td><td>✘</td><td>✘</td><td>✘</td><td>✘</td><td>✘</td><td>✓</td><td>✓</td></tr><tr><td>LearnGUI [155] ?</td><td>2025.04</td><td>2,353</td><td>✓</td><td>✘</td><td>✘</td><td>✘</td><td>✘</td><td>✘</td><td>✓</td><td>✘</td></tr></table>

<!-- Media -->

#### 5.2.1 Evaluation Pipelines

Early benchmarks in the field of phone GUI agents focused on creating controlled environment for training and evaluating these agents. MobileEnv [143], for example, introduced a universal platform for the training and evaluation of mobile interactions. It provided an isolated and controllable setting, with support for intermediate instructions and rewards. This emphasis on reliable evaluations and the ability to more naturally reflect real-world usage scenarios was a significant step forward.

To address the challenges presented by the complexities of modern operating systems and their vast action spaces, AndroidArena [144] was developed. This benchmark was designed to evaluate large language model (LLM) agents within a complex Android environment. It introduced scalable and semi-automated methods for benchmark construction, with a particular focus on cross-application collaboration and user constraints such as security concerns.

Current research primarily focuses on the overall task success rate and often overlooks the evaluation of core capabilities such as GUI grounding of agents in real-world scenarios. AgentStudio [149] provides a comprehensive platform that spans the entire development cycle, from environment setup and data collection to agent evaluation and visualization. AgentStudio also introduces three benchmark datasets: GroundUI, IDMBench, and CriticBench. These datasets are designed to evaluate agents' capabilities in GUI grounding, learning from videos, and success detection, respectively. Additionally, it introduces a benchmark suite comprising 205 real-world tasks to comprehensively evaluate agents' practical capabilities from multiple perspectives.

Recognizing the limitations in scalability and faithfulness of existing evaluation approaches, LlamaTouch [145] presented a novel testbed. This testbed enabled on-device mobile UI task execution and provided a means for faithful and scalable task evaluation. It introduced fine-grained UI component annotation and a multi-level application state matching algorithm. These features allowed for the accurate detection of critical information in each screen, enhancing the evaluation's accuracy and adaptability to dynamic UI changes.

B-MoCA [146] expanded the focus of benchmarking to include mobile device control agents across diverse configurations. By incorporating a randomization feature that could change device configurations such as UI layouts and language settings, B-MoCA was able to more effectively assess agents' generalization performance. It provided a realistic benchmark with 131 practical tasks, highlighting the need for agents to handle a wide range of real-world scenarios.

To provide a dynamic and reproducible environment for autonomous agents, AndroidWorld [147] introduced an Android environment with 116 programmatic tasks across 20 real-world apps. This benchmark emphasized the importance of ground-truth rewards and the ability to dynamically construct tasks that were parameterized and expressed in natural language. This enabled testing on a much larger and more realistic suite of tasks.


<!-- Meanless: 30 -->

For the specific evaluation of mobile LLM agents, Mo-bileAgentBench [151] proposed an efficient and user-friendly benchmark. It addressed challenges in scalability and usability by offering 100 tasks across 10 open-source apps. The benchmark also simplified the extension process for developers and ensured that it was fully autonomous and reliable.

In the domain of GUI function testing, AUITestA-gent [148] introduced the first automatic, natural language-driven GUI testing tool for mobile apps. By decoupling interaction and verification into separate modules and employing a multi-dimensional data extraction strategy, it enhanced the automation and accuracy of GUI testing. The practical usability of this tool was demonstrated in real-world deployments.

AndroidLab [150] presented a systematic Android agent framework. This framework included an operation environment with different modalities and a reproducible benchmark. Supporting both LLMs and large multimodal models (LMMs), it provided a unified platform for training and evaluating agents. Additionally, it came with an Android Instruction dataset that significantly improved the performance of open-source models.

LearnGUI [155] offers a novel approach by introducing the first comprehensive benchmark specifically designed for demonstration-based learning in mobile GUI agents. Rather than pursuing universal generalization through larger datasets, it focuses on improving agent performance in unseen scenarios through human demonstrations. The benchmark comprises 2,252 offline tasks and 101 online tasks with high-quality human demonstrations.

Finally, to evaluate the practical performance of mobile GUI agents in complex real-world environments, VisualA-gentBench [152] constructs a series of cross-domain tasks. This benchmark examines the agents' abilities in dynamic interaction and decision-making and provides abundant training trajectory data to support further performance improvement via behavior cloning. A3 (Android Agent Arena) [156] integrates 201 tasks from 21 widely-used third-party applications, covering common real-world user scenarios. It supports an extended action space compatible with any dataset annotation style. Additionally, the use of business-level LLMs automates task evaluation, reducing the need for manual assessment and enhancing scalability.

AutoEval [154] addresses the practicality and scalability challenges in mobile agent evaluation by introducing a framework that requires no manual effort to define task reward signals or implement evaluation codes. It employs a Structured Substate Representation to describe UI state changes during agent execution and utilizes a Judge System that can autonomously evaluate agent performance with over 94% accuracy compared to human verification.

Collectively, these benchmarks have made substantial contributions to the advancement of phone GUI agents. They have achieved this by providing diverse environment, tasks, and evaluation methodologies. They have addressed various challenges, including scalability, reproducibility, generalization across configurations, and the integration of advanced models like LLMs and LMMs. By facilitating rigorous testing and comparison, they have played a crucial role in driving the development of more capable and robust phone GUI agents.

#### 5.2.2 Evaluation Metrics

Evaluation metrics are crucial for measuring the performance of phone GUI agents, providing quantitative indicators of their effectiveness, efficiency, and reliability. This section categorizes and explains the various metrics used across different benchmarks based on their primary functions.

Task Completion Metrics. Task Completion Metrics assess how effectively an agent finishes assigned tasks. Task Completion Rate indicates the proportion of successfully finished tasks, with AndroidWorld [147] exemplifying its use for real-device assessments. Sub-Goal Success Rate further refines this by examining each sub-goal within a larger task, as employed by AndroidLab [150], making it particularly relevant for complex tasks that require segmentation. End-to-end Task Completion Rate, used by LlamaTouch [145], offers a holistic measure of whether an agent can see an entire multi-step task through to completion without interruption.

Action Execution Quality Metrics. These metrics evaluate the agent's precision and correctness when performing specific actions. Action Accuracy, adopted by AUITestAgent [148] and AutoDroid [66], compares each executed action to the expected one. Correct Step measures the fraction of accurate steps in an action sequence, whereas Correct Trace quantifies the alignment of the entire action trajectory with the ground truth. Operation Logic checks if the agent follows logical procedures to meet task objectives, as AndroidArena [144] demonstrates. Reasoning Accuracy, highlighted in AUITestA-gent [148], gauges how well the agent logically interprets and responds to task requirements.

Resource Utilization and Efficiency Metrics. These indicators measure how efficiently an agent handles system resources and minimizes redundant operations. Resource Consumption, tracked by AUITestAgent [148] via Completion Tokens and Prompt Tokens, reveals how much computational cost is incurred. Step Efficiency, applied by AUITestAgent and MobileAgentBench [151], compares actual steps to an optimal lower bound, while Reversed Redundancy Ratio, used by AndroidArena [144] and AndroidLab [150], evaluates unnecessary detours in the action path.

Task Understanding and Reasoning Metrics. These metrics concentrate on the agent's comprehension and analytical skills. Oracle Accuracy and Point Accuracy, used by AUITestA-gent [148], assess how well the agent interprets task instructions and verification points. Reasoning Accuracy indicates the correctness of the agent's logical deductions during execution, and Nuggets Mining, employed by AndroidArena [144], measures the ability to extract key contextual information from the UI environment.

Format and Compliance Metrics. These metrics verify whether the agent operates within expected format constraints. Invalid Format and Invalid Action, for example, are tracked in AndroidArena [144] to confirm that an agent's outputs adhere to predefined structures and remain within permissible action ranges.


<!-- Meanless: 31 -->

Completion Awareness and Reflection Metrics. Such metrics evaluate the agent's recognition of task boundaries and its capacity to learn from prior steps. Awareness of Completion, explored in AndroidArena [144], ensures the agent terminates at the correct time. Reflexion@K measures adaptive learning by examining how effectively the agent refines its performance over multiple iterations.

Evaluation Accuracy and Reliability Metrics. These indicators measure the consistency and reliability of the evaluation process. Accuracy, as used in LlamaTouch [145], validates alignment between the evaluation approach and manual verification, ensuring confidence in performance comparisons across agents.

Reward and Overall Performance Metrics. These metrics combine various performance facets into aggregated scores. Task Reward, employed by AndroidArena [144], provides a single effectiveness measure encompassing several factors. Average Reward, used in MobileEnv [143], further reflects consistent performance across multiple tasks, indicating the agent's stability and reliability.

These evaluation metrics together provide a comprehensive framework for assessing various dimensions of phone GUI agents. They cover aspects such as effectiveness, efficiency, reliability, and the ability to adapt and learn. By using these metrics, benchmarks can objectively compare the performance of different agents and systematically measure improvements. This enables researchers to identify strengths and weaknesses in different agent designs and make informed decisions about future development directions.

## 6 CHALLENGES AND FUTURE DIRECTIONS

Integrating LLMs into phone automation has propelled significant advancements but also introduced numerous challenges. Overcoming these challenges is essential for fully unlocking the potential of intelligent phone GUI agents. This section outlines key issues and possible directions for future work, encompassing dataset development, scaling fine-tuning, lightweight on-device deployment, user-centric adaptation, improving model capabilities, standardizing benchmarks, and ensuring reliability and security.

Dataset Development and Fine-Tuning Scalability. The performance of LLMs in phone automation heavily depends on datasets that capture diverse, real-world scenarios. Existing datasets often lack the breadth needed for comprehensive coverage. Future efforts should focus on developing large-scale, annotated datasets covering a wide range of applications, user behaviors, languages, and device types [137], [138]. Incorporating multimodal inputs—e.g., screenshots, UI trees, and natural language instructions—can help models better understand complex user interfaces. In addition, VideoGUI [285] proposes using instructional videos to demonstrate complex visual tasks to models, helping them to learn how to transition from an initial state to a target state. Video datasets are expected to evolve into a new form for future GUI datasets. However, scaling fine-tuning to achieve robust out-of-domain performance remains a challenge. As shown by AndroidControl [140], obtaining reliable results for high-level tasks outside the training domain may require one to two orders of magnitude more data than currently feasible. Fine-tuning alone may not suffice. Future directions should explore hybrid training methodologies, unsupervised learning, transfer learning, and auxiliary tasks to improve generalization without demanding prohibitively large datasets.

Lightweight and Efficient On-Device Deployment. Deploying LLMs on mobile devices confronts substantial computational and memory constraints. Current hardware often struggles to support large models with minimal latency and power consumption. Approaches such as model pruning, quantization, and efficient transformer architectures can address these constraints [111]. Recent innovations demonstrate promising progress. Octopus v2 [286] shows that a 2-billion parameter on-device model can outpace GPT-4 in accuracy and latency, while Lightweight Neural App Control [81] achieves substantial speed and accuracy improvements by distributing tasks efficiently. AppVLM [113], a lightweight vision-language model, matches GPT-40 in online task completion success rate while being up to ten times faster, making it practical for real-world deployment. Moreover, specialized hardware accelerators and edge computing solutions can further reduce dependency on the cloud, enhance privacy, and improve responsiveness [52]. Consider leveraging the powerful code generation capabilities of small language models (SLMs) to transform GUI task automation into a code generation problem. This approach fully utilizes the strengths of SLMs, significantly enhancing the efficiency and performance of GUI agents on mobile devices [287], [288].

User-Centric Adaptation: Interaction and Personalization. Current agents often rely on extensive human intervention to correct errors or guide task execution, undermining seamless user experiences. Enhancing the agent's ability to understand user intent and reducing manual adjustments is crucial. Future research should improve natural language understanding, incorporate voice commands and gestures, and enable agents to learn continuously from user feedback [51], [52], [61], [289]. Personalization is equally important. One-size-fits-all solutions are insufficient given users' diverse preferences and usage patterns. Agents should quickly adapt to new tasks and user-specific contexts without costly retraining. Integrating manual teaching, zero-shot learning, and few-shot learning can help agents generalize from minimal user input [61], [79], [85], [290], making them more flexible and universally applicable. For example, AdaptAgent [225] is capable of adapting to entirely new domains with as few as two human demonstrations. This not only proves the efficiency of limited human input, but also paves a new path for the development of multi-modal agents with broad adaptability. Similarly, LearnAct [155] demonstrates the power of human demonstrations in mobile GUI agents, using a multi-agent framework to automatically extract knowledge from demonstrations to enhance task completion. It establishes demonstration-based learning as a promising direction for creating more personalized and adaptive mobile agents.

Advancing Model Capabilities: Grounding, Reasoning, and Beyond. Accurately grounding language instructions in specific UI elements is a major hurdle. Although LLMs excel at language understanding, mapping instructions to precise UI interactions requires improved multimodal grounding. Future work should integrate advanced vision models, large-scale annotations, and more effective fusion techniques [80], [95], [101], [105]. Beyond grounding, improving reasoning, long-horizon planning, and adaptability in complex scenarios remains essential. Agents must handle intricate workflows, interpret ambiguous instructions, and dynamically adjust strategies as contexts evolve. Achieving these goals will likely involve new architectures, memory mechanisms, and inference algorithms that extend beyond current LLM capabilities. Standardizing Evaluation Benchmarks. Objective and reproducible benchmarks are imperative for comparing model performance. Existing benchmarks often target narrow tasks or limited domains, complicating comprehensive evaluations. Unified benchmarks covering diverse tasks, app types, and interaction modalities would foster fair comparisons and encourage more versatile and robust solutions [109], [137], [150], [151]. These benchmarks should provide standardized metrics, scenarios, and evaluation protocols, enabling researchers to identify strengths, weaknesses, and paths for improvement with greater clarity.


<!-- Meanless: 32 -->

Ensuring Reliability and Security. As agents gain access to sensitive data and perform critical tasks, reliability and security are paramount. Current systems may be susceptible to adversarial attacks, data breaches, and unintended actions [291]. At the same time, LLM agents are also susceptible to backdoor attacks [292], [293]. Recent research like AEIA-MN [294] has demonstrated that multimodal LLM-powered mobile agents are highly vulnerable to Active Environmental Injection Attacks, where attackers manipulate environmental elements (e.g., notifications) to mislead agents, achieving attack success rates up to 93% in benchmark tests. Robust security protocols, error-handling techniques, and privacy-preserving methods are needed to protect user information and maintain user trust [71], [116]. Employing techniques such as data localization, encrypted communication, and anonymization can effectively protect user privacy while collecting data [283]. FedMABench [153] addresses the challenges of distributed training using federated learning, providing a comprehensive benchmark for evaluating mobile agents across heterogeneous environments. Continuous monitoring and validation processes can detect vulnerabilities and mitigate risks in real-time [61]. Ensuring that agents behave predictably, respect user privacy, and maintain consistent performance under challenging conditions will be crucial for widespread adoption and long-term sustainability.

Addressing these challenges involves concerted efforts in data collection, model training strategies, hardware optimization, user-centric adaptation, improved grounding and reasoning, standardized benchmarks, and strong security measures. By advancing these areas, the next generation of LLM-powered phone GUI agents can become more efficient, trustworthy, and capable, ultimately delivering seamless, personalized, and secure experiences for users in dynamic mobile environment.

## 7 CONCLUSION

In this paper, we have presented a comprehensive survey of recent developments in LLM-driven phone automation technologies, illustrating how large language models can catalyze a paradigm shift from static script-based approaches to dynamic, intelligent systems capable of perceiving, reasoning about, and operating on mobile GUIs. We examined a variety of frameworks, including single-agent architectures, multi-agent collaborations, and plan-then-act pipelines, demonstrating how each approach addresses specific challenges in task complexity, adaptability, and scalability. In parallel, we analyzed both prompt engineering and training-based techniques (such as supervised fine-tuning and reinforcement learning), underscoring their roles in bridging user intent and device action.

Beyond clarifying these technical foundations, we also spotlighted emerging research directions and provided a critical appraisal of persistent obstacles. These include ensuring robust dataset coverage, optimizing LLM deployments under resource constraints, meeting real-world demand for user-centric personalization, and maintaining security and reliability in sensitive applications. We further emphasized the need for standardized benchmarks, proposing consistent metrics and evaluation protocols to fairly compare and advance competing designs.

Looking ahead, ongoing refinements in model architectures, on-device inference strategies, and multimodal data integration point to an exciting expansion of what LLM-based phone GUI agents can achieve. We anticipate that future endeavors will see the convergence of broader AI paradigms-such as embodied AI and AGI-into phone automation, thereby enabling agents to handle increasingly complex tasks with minimal human oversight. Overall, this survey not only unifies existing strands of research but also offers a roadmap for leveraging the full potential of large language models in phone GUI automation, guiding researchers toward robust, user-friendly, and secure solutions that can adapt to the evolving needs of mobile ecosystems.

## REFERENCES

[1] T. Azim and I. Neamtiu, "Targeted and depth-first exploration for systematic testing of android apps," in Proceedings of the 2013 ACM SIGPLAN international conference on Object oriented programming systems languages & applications, 2013, pp. 641-660. 1, 3, 4

[2] M. Pan, A. Huang, G. Wang, T. Zhang, and X. Li, "Reinforcement learning based curiosity-driven testing of android applications," in Proceedings of the 29th ACM SIGSOFT International Symposium on Software Testing and Analysis, 2020, pp. 153-164. 1, 4

[3] Y. Koroglu, A. Sen, O. Muslu, Y. Mete, C. Ulker, T. Tanriverdi, and Y. Donmez, "Qbe: Qlearning-based exploration of android applications," in 2018 IEEE 11th International Conference on Software Testing, Verification and Validation (ICST). IEEE, 2018, pp. 105-115. 1,4

[4] Y. Li, Z. Yang, Y. Guo, and X. Chen, "Humanoid: A deep learning-based approach to automated black-box android app testing," in 2019 34th IEEE/ACM International Conference on Automated Software Engineering (ASE). IEEE, 2019, pp. 1070-1073. 1, 4

[5] C. Degott, N. P. Borges Jr, and A. Zeller, "Learning user interface element interactions," in Proceedings of the 28th ACM SIGSOFT international symposium on software testing and analysis, 2019, pp. 296-306. 1, 4

[6] Y. L. Arnatovich and L. Wang, "A systematic literature review of automated techniques for functional gui testing of mobile applications," arXiv preprint arXiv:1812.11470, 2018. 1, 2

[7] P. S. Deshmukh, S. S. Date, P. N. Mahalle, and J. Barot, "Automated gui testing for enhancing user experience (ux): A survey of the state of the art," in International Conference on ICT for Sustainable Development. Springer, 2023, pp. 619-628. 1, 2

[8] M. Nass, "On overcoming challenges with gui-based test automation," Ph.D. dissertation, Blekinge Tekniska Högskola, 2024. 1, 2

[9] M. Nass, E. Alégroth, and R. Feldt, "Why many challenges with gui test automation (will) remain," Information and Software Technology, vol. 138, p. 106625, 2021. 1, 2


<!-- Meanless: 33 -->

[10] P. Tramontana, D. Amalfitano, N. Amatucci, and A. R. Fasolino, "Automated functional testing of mobile applications: a systematic mapping study," Software Quality Journal, vol. 27, pp. 149-201, 2019. 1, 2

[11] Y. Li, H. Wen, W. Wang, X. Li, Y. Yuan, G. Liu, J. Liu, W. Xu, X. Wang, Y. Sun et al., "Personal llm agents: Insights and survey about the capability, efficiency and security," arXiv preprint arXiv:2401.05459, 2024. 1

[12] T. Guo, X. Chen, Y. Wang, R. Chang, S. Pei, N. V. Chawla, O. Wiest, and X. Zhang, "Large language model based multi-agents: A survey of progress and challenges," arXiv preprint arXiv:2402.01680, 2024. 1

[13] L. Wang, C. Ma, X. Feng, Z. Zhang, H. Yang, J. Zhang, Z. Chen, J. Tang, X. Chen, Y. Lin et al., "A survey on large language model based autonomous agents," Frontiers of Computer Science, vol. 18, no. 6, p. 186345, 2024. 1

[14] H. Jin, L. Huang, H. Cai, J. Yan, B. Li, and H. Chen, "From llms to llm-based agents for software engineering: A survey of current, challenges and future," arXiv preprint arXiv:2408.02479, 2024. 1

[15] S. Bubeck, V. Chandrasekaran, R. Eldan, J. Gehrke, E. Horvitz, E. Kamar, P. Lee, Y. T. Lee, Y. Li, S. Lundberg et al., "Sparks of artificial general intelligence: Early experiments with gpt-4," arXiv preprint arXiv:2303.12712, 2023. 1

[16] X. Huang, W. Liu, X. Chen, X. Wang, H. Wang, D. Lian, Y. Wang, R. Tang, and E. Chen, "Understanding the planning of llm agents: A survey," arXiv preprint arXiv:2402.02716, 2024. 1

[17] S. V. Albrecht and P. Stone, "Autonomous agents modelling other agents: A comprehensive survey and open problems," Artificial Intelligence, vol. 258, pp. 66-95, 2018. 1

[18] G. Anscombe, "Intention," 2000. 1

[19] D. C. Dennett, "Précis of the intentional stance," Behavioral and brain sciences, vol. 11, no. 3, pp. 495-505, 1988. 1

[20] Y. Shoham, "Agent-oriented programming," Artificial intelligence, vol. 60, no. 1, pp. 51-92, 1993. 1

[21] D. L. Poole and A. K. Mackworth, Artificial Intelligence: foundations of computational agents. Cambridge University Press, 2010. 1

[22] B. Inkster, S. Sarda, V. Subramanian et al., "An empathy-driven, conversational artificial intelligence agent (wysa) for digital mental well-being: real-world data evaluation mixed-methods study," JMIR mHealth and uHealth, vol. 6, no. 11, p. e12106, 2018. 1

[23] J. Gao, M. Galley, and L. Li, "Neural approaches to conversational ai," in The 41st international ACM SIGIR conference on research & development in information retrieval, 2018, pp. 1371-1374. 1

[24] E. Luger and A. Sellen, "" like having a really bad pa" the gulf between user expectation and experience of conversational agents," in Proceedings of the 2016 CHI conference on human factors in computing systems, 2016, pp. 5286-5297. 1

[25] S. Amershi, M. Cakmak, W. B. Knox, and T. Kulesza, "Power to the people: The role of humans in interactive machine learning," AI magazine, vol. 35, no. 4, pp. 105-120, 2014. 1

[26] P. F. Christiano, J. Leike, T. Brown, M. Martic, S. Legg, and D. Amodei, "Deep reinforcement learning from human preferences," Advances in neural information processing systems, vol. 30, 2017. 1

[27] J. Köhl, R. Kolnaar, and W. J. Ravensberg, "Mode of action of microbial biological control agents against plant diseases: relevance beyond efficacy," Frontiers in plant science, vol. 10, p. 845, 2019. 1

[28] A. Radford, "Improving language understanding by generative pre-training," 2018. 1, 2, 3, 15, 16

[29] A. Radford, J. Wu, R. Child, D. Luan, D. Amodei, I. Sutskever et al., "Language models are unsupervised multitask learners," OpenAI blog, vol. 1, no. 8, p. 9, 2019. 1, 2, 3, 15, 16

[30] T. B. Brown, "Language models are few-shot learners," arXiv preprint arXiv:2005.14165, 2020. 1, 2, 3, 15, 16

[31] J. Achiam, S. Adler, S. Agarwal, L. Ahmad, I. Akkaya, F. L. Aleman, D. Almeida, J. Altenschmidt, S. Altman, S. Anadkat et al., "Gpt-4 technical report," arXiv preprint arXiv:2303.08774, 2023. 1, 2, 3, 15, 17

[32] R. Bavishi, E. Elsen, C. Hawthorne, M. Nye, A. Odena, A. Somani, and S. Taşırlar, "Fuyu-8b: A multimodal architecture for ai agents," 2023. 1

[33] G. Wang, Y. Xie, Y. Jiang, A. Mandlekar, C. Xiao, Y. Zhu, L. Fan, and A. Anandkumar, "Voyager: An open-ended embodied agent with large language models," arXiv preprint arXiv:2305.16291, 2023. 1

[34] S. Hong, X. Zheng, J. Chen, Y. Cheng, J. Wang, C. Zhang, Z. Wang, S. K. S. Yau, Z. Lin, L. Zhou et al., "Metagpt: Meta programming for multi-agent collaborative framework," arXiv preprint arXiv:2308.00352, 2023. 1

[35] G. Li, H. Hammoud, H. Itani, D. Khizbullin, and B. Ghanem, "Camel: Communicative agents for" mind" exploration of large language model society," Advances in Neural Information Processing Systems, vol. 36, pp. 51991-52008, 2023. 1

[36] J. S. Park, J. O'Brien, C. J. Cai, M. R. Morris, P. Liang, and M. S. Bernstein, "Generative agents: Interactive simulacra of human behavior," in Proceedings of the 36th annual acm symposium on user interface software and technology, 2023, pp. 1-22. 1

[37] D. A. Boiko, R. MacKnight, and G. Gomes, "Emergent autonomous scientific research capabilities of large language models," arXiv preprint arXiv:2304.05332, 2023. 1

[38] C. Qian, X. Cong, C. Yang, W. Chen, Y. Su, J. Xu, Z. Liu, and M. Sun, "Communicative agents for software development," arXiv preprint arXiv:2307.07924, vol. 6, no. 3, 2023. 1

[39] Y. Xia, M. Shenoy, N. Jazdi, and M. Weyrich, "Towards autonomous system: flexible modular production system enhanced with large language model agents," in 2023 IEEE 28th International Conference on Emerging Technologies and Factory Automation (ETFA). IEEE, 2023, pp. 1-8. 1

[40] I. Dasgupta, C. Kaeser-Chen, K. Marino, A. Ahuja, S. Babayan, F. Hill, and R. Fergus, "Collaborating with language models for embodied reasoning," arXiv preprint arXiv:2302.00763, 2023. 1

[41] C. Qian, W. Liu, H. Liu, N. Chen, Y. Dang, J. Li, C. Yang, W. Chen, Y. Su, X. Cong et al., "Chatdev: Communicative agents for software development," in Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 2024, pp. 15174-15186. 1

[42] Y. Dong, X. Jiang, Z. Jin, and G. Li, "Self-collaboration code generation via chatgpt," ACM Transactions on Software Engineering and Methodology, vol. 33, no. 7, pp. 1-38, 2024. 1

[43] B. Goertzel, "Artificial general intelligence: concept, state of the art, and future prospects," Journal of Artificial General Intelligence, vol. 5, no. 1, p. 1, 2014. 1

[44] Z. Xi, W. Chen, X. Guo, W. He, Y. Ding, B. Hong, M. Zhang, J. Wang, S. Jin, E. Zhou et al., "The rise and potential of large language model based agents: A survey," arXiv preprint arXiv:2309.07864, 2023. 1, 11, 12

[45] H. Furuta, Y. Matsuo, A. Faust, and I. Gur, "Exposing limitations of language model agents in sequential-task compositions on the web," in ICLR 2024 Workshop on Large Language Model (LLM) Agents, 2024. 1

[46] W. Hong, W. Wang, Q. Lv, J. Xu, W. Yu, J. Ji, Y. Wang, Z. Wang, Y. Dong, M. Ding et al., "Cogagent: A visual language model for gui agents," in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 14281-14290. 1, 3, 10, 11, 19, 20

[47] B. Zheng, B. Gou, J. Kil, H. Sun, and Y. Su, "Gpt-4v (ision) is a generalist web agent, if grounded," arXiv preprint arXiv:2401.01614, 2024. 1, 3, 8, 15

[48] C. Zhang, Z. Yang, J. Liu, Y. Han, X. Chen, Z. Huang, B. Fu, and G. Yu, "Appagent: Multimodal agents as smartphone users," arXiv preprint arXiv:2312.13771, 2023. 1, 3, 7, 8, 10, 11, 16, 17, 19

[49] Y. Song, Y. Bian, Y. Tang, and Z. Cai, "Navigating interfaces with ai for enhanced user interaction," arXiv preprint arXiv:2312.11190, 2023. 1, 16

[50] H. Wen, Y. Li, G. Liu, S. Zhao, T. Yu, T. J.-J. Li, S. Jiang, Y. Liu, Y. Zhang, and Y. Liu, "Autodroid: Llm-powered task automation in android," in Proceedings of the 30th Annual International Conference on Mobile Computing and Networking, 2024, pp. 543-557. 1, 2, 3,7,8,9,11,16,19,29

[51] J. Wang, H. Xu, J. Ye, M. Yan, W. Shen, J. Zhang, F. Huang, and J. Sang, "Mobile-agent: Autonomous multi-modal mobile device agent with visual perception," arXiv preprint arXiv:2401.16158, 2024. 1, 2, 3, 7, 8, 16, 18, 19, 31

[52] J. Wang, H. Xu, H. Jia, X. Zhang, M. Yan, W. Shen, J. Zhang, F. Huang, and J. Sang, "Mobile-agent-v2: Mobile device operation assistant with effective navigation via multi-agent collaboration," arXiv preprint arXiv:2406.01014, 2024. 1, 2, 3, 7, 8, 11, 12, 13, 16, 18, 19, 31

[53] H. Wen, H. Wang, J. Liu, and Y. Li, "Droidbot-gpt: Gpt-powered ui automation for android," arXiv preprint arXiv:2304.07061, 2023. 2,3,8,9,16,19


<!-- Meanless: 34 -->

[54] Z. Liu, C. Li, C. Chen, J. Wang, B. Wu, Y. Wang, J. Hu, and Q. Wang, "Vision-driven automated mobile gui testing via multimodal large language model," arXiv preprint arXiv:2407.03037, 2024. 2, 3, 10, 16, 18, 19

[55] J. Zhang, C. Zhao, Y. Zhao, Z. Yu, M. He, and J. Fan, "Mobile-experts: A dynamic tool-enabled agent team in mobile devices," arXiv preprint arXiv:2407.03913, 2024. 2, 3, 8, 14, 16, 18, 19

[56] Y. Lu, J. Yang, Y. Shen, and A. Awadallah, "Omniparser for pure vision based gui agent," arXiv preprint arXiv:2408.00203, 2024. 2, 3, 10, 11, 16, 17, 19

[57] B. Wang, G. Li, and Y. Li, "Enabling conversational interaction with mobile ui using large language models," in Proceedings of the 2023 CHI Conference on Human Factors in Computing Systems, 2023, pp. 1-17. 3, 8, 9, 16, 19

[58] Y. Guan, D. Wang, Z. Chu, S. Wang, F. Ni, R. Song, L. Li, J. Gu, and C. Zhuang, "Intelligent virtual assistants with Ilm-based process automation," arXiv preprint arXiv:2312.06677, 2023. 3, 10

[59] Y. Fan, L. Ding, C.-C. Kuo, S. Jiang, Y. Zhao, X. Guan, J. Yang, Y. Zhang, and X. E. Wang, "Read anywhere pointed: Layout-aware gui screen reading with tree-of-lens grounding," 2024. [Online]. Available: https://arxiv.org/abs/2406.19263 3, 10

[60] A. Yan, Z. Yang, W. Zhu, K. Lin, L. Li, J. Wang, J. Yang, Y. Zhong, J. McAuley, J. Gao et al., "Gpt-4v in wonderland: Large multimodal models for zero-shot smartphone gui navigation," arXiv preprint arXiv:2311.07562, 2023. 3, 10, 16, 17, 19

[61] S. Lee, J. Choi, J. Lee, M. H. Wasi, H. Choi, S. Y. Ko, S. Oh, and I. Shin, "Explore, select, derive, and recall: Augmenting llm with human-like memory for mobile task automation," arXiv preprint arXiv:2312.03003, 2023. 3, 7, 10, 11, 16, 19, 31, 32

[62] Q. Wu, D. Gao, K. Q. Lin, Z. Wu, X. Guo, P. Li, W. Zhang, H. Wang, and M. Z. Shou, "Gui action narrator: Where and when did that action take place?" arXiv preprint arXiv:2406.13719, 2024. 3, 10, 18, 19

[63] Q. Wu, W. Xu, W. Liu, T. Tan, J. Liu, A. Li, J. Luan, B. Wang, and S. Shang, "Mobilevlm: A vision-language model for better intra-and inter-ui understanding," arXiv preprint arXiv:2409.14818, 2024. 3, 10, 25

[64] Y. Li, C. Zhang, W. Yang, B. Fu, P. Cheng, X. Chen, L. Chen, and Y. Wei, "Appagent v2: Advanced agent for flexible mobile interactions," arXiv preprint arXiv:2408.11824, 2024. 3, 11, 18, 19

[65] W. Jiang, Y. Zhuang, C. Song, X. Yang, J. T. Zhou, and C. Zhang, "Appagentx: Evolving gui agents as proficient smartphone users," arXiv preprint arXiv:2503.02268, 2025. 3, 11

[66] Z. Zhang and A. Zhang, "You only look at screens: Multimodal chain-of-action agents," arXiv preprint arXiv:2309.11436, 2023. 3, 7, 10, 11, 19, 20, 30

[67] G. Baechler, S. Sunkara, M. Wang, F. Zubach, H. Mansoor, V. Etter, V. Cărbune, J. Lin, J. Chen, and A. Sharma, "Screenai: A vision-language model for ui and infographics understanding," arXiv preprint arXiv:2402.04615, 2024. 3, 7, 11, 19, 20, 22

[68] J. Wang, H. Xu, X. Zhang, M. Yan, J. Zhang, F. Huang, and J. Sang, "Mobile-agent-v: Learning mobile device operation through video-guided multi-agent collaboration," arXiv preprint arXiv:2502.17110, 2025. 3, 11, 19

[69] P. Cheng, Z. Wu, Z. Wu, A. Zhang, Z. Zhang, and G. Liu, "Os-kairos: Adaptive interaction for mllm-powered gui agents," arXiv preprint arXiv:2503.16465, 2025. 3

[70] Y. Sun, S. Zhao, T. Yu, H. Wen, S. Va, M. Xu, Y. Li, and C. Zhang, "Gui-xplore: Empowering generalizable gui agents with one exploration," arXiv preprint arXiv:2503.17709, 2025. 3

[71] X. Ma, Z. Zhang, and H. Zhao, "Coco-agent: A comprehensive cognitive mllm agent for smartphone gui automation," in Findings of the Association for Computational Linguistics ACL 2024, 2024, pp. 9097-9110. 3, 11, 20, 21, 32

[72] Z. Song, Y. Li, M. Fang, Z. Chen, Z. Shi, and Y. Huang, "Mmac-copilot: Multi-modal agent collaboration operating system copilot," arXiv preprint arXiv:2404.18074, 2024. 3, 12, 13

[73] W. Tan, W. Zhang, X. Xu, H. Xia, Z. Ding, B. Li, B. Zhou, J. Yue, J. Jiang, Y. Li et al., "Cradle: Empowering foundation agents towards general computer control," in NeurIPS 2024 Workshop on Open-World Agents. 3, 12, 14

[74] Z. Wang, H. Xu, J. Wang, X. Zhang, M. Yan, J. Zhang, F. Huang, J and H. Ji, "Mobile-agent-e: Self-evolving mobile assistant for complex tasks," arXiv preprint arXiv:2501.11733, 2025. 3, 14, 18, 19

[75] T. Huang, C. Yu, W. Shi, Z. Peng, D. Yang, W. Sun, and Y. Shi, "Promptrpa: Generating robotic process automation on smart-phones from textual prompts," arXiv preprint arXiv:2404.02475, 2024. 3, 14, 16

[76] Y. Zhou, S. Wang, S. Dai, Q. Jia, Z. Du, Z. Dong, and J. Xu, "Chop: Mobile operating assistant with constrained high-frequency optimized subtask planning," arXiv preprint arXiv:2503.03743, 2025. 3, 14

[77] S. Agashe, K. Wong, V. Tu, J. Yang, A. Li, and X. E. Wang, "Agent s2: A compositional generalist-specialist framework for computer use agents," arXiv preprint arXiv:2504.00906, 2025. 3, 14

[78] X. Zhang, Y. Deng, Z. Ren, S.-K. Ng, and T.-S. Chua, "Ask-before-plan: Proactive language agents for real-world planning," arXiv preprint arXiv:2406.12639, 2024. 3, 14

[79] P. Sodhi, S. Branavan, Y. Artzi, and R. McDonald, "Step: Stacked llm policies for web actions," in First Conference on Language Modeling, 2024. 3, 7, 14, 31

[80] B. Gou, R. Wang, B. Zheng, Y. Xie, C. Chang, Y. Shu, H. Sun, and Y. Su, "Navigating the digital world as humans do: Universal visual grounding for gui agents," arXiv preprint arXiv:2410.05243, 2024. 3, 7, 8, 15, 31

[81] F. Christianos, G. Papoudakis, T. Coste, J. Hao, J. Wang, and K. Shao, "Lightweight neural app control," arXiv preprint arXiv:2410.17883, 2024. 3, 15, 31

[82] J. Hoscilowicz, B. Maj, B. Kozakiewicz, O. Tymoshchuk, and A. Janicki, "Clickagent: Enhancing ui location capabilities of autonomous agents," arXiv preprint arXiv:2410.11872, 2024. 3, 8, 15

[83] Y. Wang, H. Zhang, J. Tian, and Y. Tang, "Ponder & press: Advancing visual gui agent towards general computer control," arXiv preprint arXiv:2412.01268, 2024. 3, 15

[84] M. Taeb, A. Swearngin, E. Schoop, R. Cheng, Y. Jiang, and J. Nichols, "Axnav: Replaying accessibility tests from natural language," in Proceedings of the CHI Conference on Human Factors in Computing Systems, 2024, pp. 1-16. 3, 16, 19

[85] Y. Song, Y. Bian, Y. Tang, G. Ma, and Z. Cai, "Visiontasker: Mobile task automation using vision based ui understanding and llm task planning," in Proceedings of the 37th Annual ACM Symposium on User Interface Software and Technology, 2024, pp. 1-17. 3, 10, 11, 18, 19, 31

[86] X. Li, J. Cao, Y. Liu, S.-C. Cheung, and H. Wang, "Reusedroid: A vlm-empowered android ui test migrator boosted by active feedback," arXiv preprint arXiv:2504.02357, 2025. 3

[87] B. F. Demissie, Y. N. Tun, L. K. Shar, and M. Ceccato, "Vlm-fuzz: Vision language model assisted recursive depth-first search exploration for effective ui testing of android apps," arXiv preprint arXiv:2504.11675, 2025. 3

[88] S. Nong, J. Zhu, R. Wu, J. Jin, S. Shan, X. Huang, and W. Xu, "Mobileflow: A multimodal llm for mobile gui agent," arXiv preprint arXiv:2407.04346, 2024. 3, 20, 21

[89] K. Q. Lin, L. Li, D. Gao, Z. Yang, S. Wu, Z. Bai, W. Lei, L. Wang, and M. Z. Shou, "Showui: One vision-language-action model for gui visual agent," arXiv preprint arXiv:2411.17465, 2024. 3, 20, 21

[90] Y. Xu, Z. Wang, J. Wang, D. Lu, T. Xie, A. Saha, D. Sahoo, T. Yu, and C. Xiong, "Aguvis: Unified pure vision agents for autonomous gui interaction," arXiv preprint arXiv:2412.04454, 2024. 3, 20, 21

[91] D. Luo, B. Tang, K. Li, G. Papoudakis, J. Song, S. Gong, J. Hao, J. Wang, and K. Shao, "Vimo: A generative visual gui world model for app agent," arXiv preprint arXiv:2504.13936, 2025. 3

[92] Y. Qin, Y. Ye, J. Fang, H. Wang, S. Liang, S. Tian, J. Zhang, J. Li, Y. Li, S. Huang et al., "Ui-tars: Pioneering automated gui interaction with native agents," arXiv preprint arXiv:2501.12326, 2025. 3, 20, 21

[93] T. Li, G. Li, J. Zheng, P. Wang, and Y. Li, "Mug: Interactive multimodal grounding on user interfaces," arXiv preprint arXiv:2209.15099, 2022. 3, 21

[94] Y. Qian, Y. Lu, A. G. Hauptmann, and O. Riva, "Visual grounding for user interfaces," in Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 6: Industry Track), 2024, pp. 97-107. 3, 20, 21

[95] J. Zhang, Y. Yu, M. Liao, W. Li, J. Wu, and Z. Wei, "Ui-hawk: Unleashing the screen stream understanding for gui agents," 2024. 3, 7, 15, 20, 21, 22, 31

[96] Y. Yang, Y. Wang, D. Li, Z. Luo, B. Chen, C. Huang, and J. Li, "Aria-ui: Visual grounding for gui instructions," arXiv preprint arXiv:2412.16256, 2024. 3, 20, 21

[97] Z. Wu, Z. Wu, F. Xu, Y. Wang, Q. Sun, C. Jia, K. Cheng, Z. Ding, L. Chen, P. P. Liang et al., "Os-atlas: A foundation action model for generalist gui agents," arXiv preprint arXiv:2410.23218, 2024. 3, 20, 21


<!-- Meanless: 35 -->

[98] Z. Wang, W. Chen, L. Yang, S. Zhou, S. Zhao, H. Zhan, J. Jin, L. Li, Z. Shao, and J. Bu, "Mp-gui: Modality perception with mllms for gui understanding," arXiv preprint arXiv:2503.14021, 2025. 3, 20, 22

[99] Z. Wu, P. Cheng, Z. Wu, T. Ju, Z. Zhang, and G. Liu, "Smoothing grounding and reasoning for mllm-powered gui agents with query-oriented pivot tasks," arXiv preprint arXiv:2503.00401, 2025. 3

[100] Y. Fan, H. Zhao, R. Zhang, Y. Shen, X. E. Wang, and G. Wu, "Gui-bee: Align gui action grounding to novel environments via autonomous exploration," arXiv preprint arXiv:2501.13896, 2025. 3, 20, 21

[101] K. You, H. Zhang, E. Schoop, F. Weers, A. Swearngin, J. Nichols, Y. Yang, and Z. Gan, "Ferret-ui: Grounded mobile ui understanding with multimodal llms," arXiv preprint arXiv:2404.05719, 2024. 3,7,11,15,20,21,31

[102] Z. Li, K. You, H. Zhang, D. Feng, H. Agrawal, X. Li, M. P. S. Moorthy, J. Nichols, Y. Yang, and Z. Gan, "Ferret-ui 2: Mastering universal user interface understanding across platforms," 2024. [Online]. Available: https://arxiv.org/abs/2410.18967 3, 20, 22, 28

[103] A. Burns, K. Saenko, and B. A. Plummer, "Tell me what's next: Textual foresight for generic ui representations," arXiv preprint arXiv:2406.07822, 2024. 3, 20, 22

[104] Q. Chen, D. Pitawela, C. Zhao, G. Zhou, H.-T. Chen, and Q. Wu, "Webvln: Vision-and-language navigation on websites," in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 38, no. 2, 2024, pp. 1165-1173. 3, 20, 22

[105] K. Cheng, Q. Sun, Y. Chu, F. Xu, Y. Li, J. Zhang, and Z. Wu, "Seeclick: Harnessing gui grounding for advanced visual gui agents," arXiv preprint arXiv:2401.10935, 2024. 3, 5, 7, 16, 23, 24, 31

[106] Y. Liu, P. Li, Z. Wei, C. Xie, X. Hu, X. Xu, S. Zhang, X. Han, H. Yang, and F. Wu, "Infiguiagent: A multimodal generalist gui agent with native reasoning and reflection," arXiv preprint arXiv:2501.04575, 2025. 3, 23, 24

[107] W. Chen, J. Cui, J. Hu, Y. Qin, J. Fang, Y. Zhao, C. Wang, J. Liu, G. Chen, Y. Huo et al., "Guicourse: From general vision language models to versatile gui agents," arXiv preprint arXiv:2406.11317, 2024. 3, 5, 16, 23, 24

[108] S. Yuan, Z. Chen, Z. Xi, J. Ye, Z. Du, and J. Chen, "Agent-r: Training language model agents to reflect via iterative self-training," arXiv preprint arXiv:2501.11425, 2025. 3, 23, 24

[109] Q. Lu, W. Shao, Z. Liu, F. Meng, B. Li, B. Chen, S. Huang, K. Zhang, Y. Qiao, and P. Luo, "Gui odyssey: A comprehensive dataset for cross-app gui navigation on mobile devices," arXiv preprint arXiv:2406.08451, 2024. 3, 5, 16, 23, 24, 27, 28, 32

[110] P. Pawlowski, K. Zawistowski, W. Lapacz, M. Skorupa, A. Wiacek, S. Postansque, and J. Hoscilowicz, "Tinyclick: Single-turn agent for empowering gui automation," arXiv preprint arXiv:2410.11871, 2024. 3, 5, 16, 23, 24

[111] T. Ding, "Mobileagent: enhancing mobile control via human-machine interaction and sop integration," arXiv preprint arXiv:2401.04124, 2024. 3, 23, 24, 31

[112] J. R. A. Moniz, S. Krishnan, M. Ozyildirim, P. Saraf, H. C. Ates, Y. Zhang, H. Yu, and N. Rajshree, "Realm: Reference resolution as language modeling," arXiv preprint arXiv:2403.20329, 2024. 3, 23, 24

[113] G. Papoudakis, T. Coste, Z. Wu, J. Hao, J. Wang, and K. Shao, "Appvlm: A lightweight vision language model for online app control," arXiv preprint arXiv:2502.06395, 2025. 3, 31

[114] G. Dai, S. Jiang, T. Cao, Y. Li, Y. Yang, R. Tan, M. Li, and L. Qiu, "Advancing mobile gui agents: A verifier-driven approach to practical deployment," arXiv preprint arXiv:2503.15937, 2025. 3, 20, 21

[115] S. Haque and C. Csallner, "Infering alt-text for ui icons with large language models during app development," arXiv preprint arXiv:2409.18060, 2024. 3, 23, 24

[116] H. Bai, Y. Zhou, M. Cemri, J. Pan, A. Suhr, S. Levine, and A. Kumar, "Digirl: Training in-the-wild device-control agents with autonomous reinforcement learning," arXiv preprint arXiv:2406.11896, 2024. 3, 16, 24, 25, 32

[117] T. Wang, Z. Wu, J. Liu, J. Hao, J. Wang, and K. Shao, "Distrl: An asynchronous distributed reinforcement learning framework for on-device control agents," arXiv preprint arXiv:2410.14803, 2024. 3, 16, 24, 25

[118] X. Liu, B. Qin, D. Liang, G. Dong, H. Lai, H. Zhang, H. Zhao, I. L. Iong, J. Sun, J. Wang et al., "Autoglm: Autonomous foundation agents for guis," arXiv preprint arXiv:2411.00820, 2024. 3, 8, 25

[119] H. Bai, Y. Zhou, L. E. Li, S. Levine, and A. Kumar, "Digi-q: Learning q-value functions for training device-control agents," arXiv preprint arXiv:2502.15760, 2025. 3, 25, 26

[120] Q. Wu, W. Liu, J. Luan, and B. Wang, "Reachagent: Enhancing mobile agent via page reaching and operation," arXiv preprint arXiv:2502.02955, 2025. 3, 25

[121] Q. Wu, J. Liu, J. Hao, J. Wang, and K. Shao, "Vsc-rl: Advancing autonomous vision-language agents with variational subgoal-conditioned reinforcement learning," arXiv preprint arXiv:2502.07949, 2025. 3, 25, 26

[122] Z. Lu, Y. Chai, Y. Guo, X. Yin, L. Liu, H. Wang, G. Xiong, and H. Li, "Ui-r1: Enhancing action prediction of gui agents by reinforcement learning," arXiv preprint arXiv:2503.21620, 2025. 3, 25, 26

[123] X. Xia and R. Luo, "Gui-r1: A generalist r1-style vision-language action model for gui agents," arXiv preprint arXiv:2504.10458, 2025. 3

[124] Y. Song, D. Yin, X. Yue, J. Huang, S. Li, and B. Y. Lin, "Trial and error: Exploration-based trajectory optimization for llm agents," arXiv preprint arXiv:2403.02502, 2024. 3, 16, 25, 26

[125] P. Putta, E. Mills, N. Garg, S. Motwani, C. Finn, D. Garg, and R. Rafailov, "Agent q: Advanced reasoning and learning for autonomous ai agents," arXiv preprint arXiv:2408.07199, 2024. 3, 25, 26

[126] H. Lai, X. Liu, I. L. Iong, S. Yao, Y. Chen, P. Shen, H. Yu, H. Zhang, X. Zhang, Y. Dong et al., "Autowebglm: A large language model-based web navigating agent," in Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, 2024, pp. 5295-5306. 3, 25, 26

[127] M. Fereidouni, A. Mosharrof, and A. Siddique, "Grounded language agent for product search via intelligent web interactions," in Proceedings of the 1st Workshop on Customizable NLP: Progress and Challenges in Customizing NLP for a Domain, Application, Group, or Individual (CustomNLP4U). Association for Computational Linguistics, 2024, p. 63-75. [Online]. Available: http://dx.doi.org/10.18653/v1/2024.customnlp4u-1.73, 25, 26

[128] R. Niu, J. Li, S. Wang, Y. Fu, X. Hu, X. Leng, H. Kong, Y. Chang, and Q. Wang, "Screenagent: A vision language model-driven computer control agent," arXiv preprint arXiv:2402.07945, 2024. 3, 25, 26

[129] D. Gao, L. Ji, Z. Bai, M. Ouyang, P. Li, D. Mao, Q. Wu, W. Zhang, P. Wang, X. Guo et al., "Assistgui: Task-oriented desktop graphical user interface automation," arXiv preprint arXiv:2312.13108, 2023. 3, 26

[130] B. Deka, Z. Huang, C. Franzen, J. Hibschman, D. Afergan, Y. Li, J. Nichols, and R. Kumar, "Rico: A mobile app dataset for building data-driven design applications," in Proceedings of the 30th annual ACM symposium on user interface software and technology, 2017, pp. 845-854. 3, 27, 28

[131] S. Sunkara, M. Wang, L. Liu, G. Baechler, Y.-C. Hsiao, A. Sharma, J. Stout et al., "Towards better semantic understanding of mobile interfaces," arXiv preprint arXiv:2210.02663, 2022. 3, 27

[132] Y. Li, J. He, X. Zhou, Y. Zhang, and J. Baldridge, "Mapping natural language instructions to mobile ui action sequences," arXiv preprint arXiv:2005.03776, 2020. 3, 9, 27, 28

[133] A. Burns, D. Arsan, S. Agrawal, R. Kumar, K. Saenko, and B. A. Plummer, "Mobile app tasks with iterative feedback (motif): Addressing task feasibility in interactive visual environments," arXiv preprint arXiv:2104.08560, 2021. 3, 9, 27, 28

[134] C. Bai, X. Zang, Y. Xu, S. Sunkara, A. Rastogi, J. Chen et al., "Uibert: Learning generic multimodal representations for ui understanding," arXiv preprint arXiv:2107.13731, 2021. 3, 9, 27, 28

[135] L. Sun, X. Chen, L. Chen, T. Dai, Z. Zhu, and K. Yu, "Meta-gui: Towards multi-modal conversational agents on mobile gui," arXiv preprint arXiv:2205.11029, 2022. 3, 27, 28

[136] S. G. Venkatesh, P. Talukdar, and S. Narayanan, "Ugif: Ui grounded instruction following," arXiv preprint arXiv:2211.07615, 2022. 3, 27, 28

[137] C. Rawles, A. Li, D. Rodriguez, O. Riva, and T. Lillicrap, "An-droidinthewild: A large-scale dataset for android device control," Advances in Neural Information Processing Systems, vol. 36, 2024. 3, 27, 28, 31, 32

[138] J. Zhang, J. Wu, Y. Teng, M. Liao, N. Xu, X. Xiao, Z. Wei, and D. Tang, "Android in the zoo: Chain-of-action-thought for gui agents," arXiv preprint arXiv:2403.02713, 2024. 3, 27, 28, 31


<!-- Meanless: 36 -->

[139] D. Chen, Y. Huang, S. Wu, J. Tang, L. Chen, Y. Bai, Z. He, C. Wang, H. Zhou, Y. Li et al., "Gui-world: A dataset for gui-oriented multimodal llm-based agents," arXiv preprint arXiv:2406.10819, 2024. 3, 27

[140] W. Li, W. Bishop, A. Li, C. Rawles, F. Campbell-Ajala, D. Tyama-gundlu, and O. Riva, "On the effects of data scale on computer control agents," arXiv preprint arXiv:2406.03679, 2024. 3, 6, 7, 27, 28, 31

[141] Y. Chai, S. Huang, Y. Niu, H. Xiao, L. Liu, D. Zhang, P. Gao, S. Ren, and H. Li, "Amex: Android multi-annotation expo dataset for mobile gui agents," arXiv preprint arXiv:2407.17490, 2024. 3, 27, 28

[142] L. Gao, L. Zhang, S. Wang, S. Wang, Y. Li, and M. Xu, "Mo-bileviews: A large-scale mobile gui dataset," arXiv preprint arXiv:2409.14337, 2024. 3, 27, 28

[143] D. Zhang, L. Chen, and K. Yu, "Mobile-env: A universal platform for training and evaluation of mobile interaction," arXiv preprint arXiv:2305.08144, 2023. 3, 29, 31

[144] M. Xing, R. Zhang, H. Xue, Q. Chen, F. Yang, and Z. Xiao, "Understanding the weakness of large language model agents within a complex android environment," in Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, 2024, pp. 6061-6072. 3, 29, 30, 31

[145] L. Zhang, S. Wang, X. Jia, Z. Zheng, Y. Yan, L. Gao, Y. Li, and M. Xu, "Llamatouch: A faithful and scalable testbed for mobile ui automation task evaluation," arXiv preprint arXiv:2404.16054, 2024. 3, 5, 29, 30, 31

[146] J. Lee, T. Min, M. An, D. Hahm, H. Lee, C. Kim, and K. Lee, "Benchmarking mobile device control agents across diverse configurations," arXiv preprint arXiv:2404.16660, 2024. 3, 29

[147] C. Rawles, S. Clinckemaillie, Y. Chang, J. Waltz, G. Lau, M. Fair, A. Li, W. Bishop, W. Li, F. Campbell-Ajala et al., "Androidworld: A dynamic benchmarking environment for autonomous agents," arXiv preprint arXiv:2405.14573, 2024. 3, 29, 30

[148] Y. Hu, X. Wang, Y. Wang, Y. Zhang, S. Guo, C. Chen, X. Wang, and Y. Zhou, "Auitestagent: Automatic requirements oriented gui function testing," arXiv preprint arXiv:2407.09018, 2024. 3, 29, 30

[149] L. Zheng, Z. Huang, Z. Xue, X. Wang, B. An, and S. Yan, "Agentstudio: A toolkit for building general virtual agents," arXiv preprint arXiv:2403.17918, 2024. 3, 29

[150] Y. Xu, X. Liu, X. Sun, S. Cheng, H. Yu, H. Lai, S. Zhang, D. Zhang, J. Tang, and Y. Dong, "Androidlab: Training and systematic benchmarking of android autonomous agents," arXiv preprint arXiv:2410.24024, 2024. 3, 29, 30, 32

[151] L. Wang, Y. Deng, Y. Zha, G. Mao, Q. Wang, T. Min, W. Chen, and S. Chen, "Mobileagentbench: An efficient and user-friendly benchmark for mobile llm agents," arXiv preprint arXiv:2406.08184, 2024. 3, 29, 30, 32

[152] X. Liu, T. Zhang, Y. Gu, I. L. Iong, Y. Xu, X. Song, S. Zhang, H. Lai, X. Liu, H. Zhao et al., "Visualagentbench: Towards large multimodal models as visual foundation agents," arXiv preprint arXiv:2408.06327, 2024. 3, 29, 30

[153] W. Wang, Z. Yu, R. Ye, J. Zhang, S. Chen, and Y. Wang, "Fedmabench: Benchmarking mobile agents on decentralized heterogeneous user data," arXiv preprint arXiv:2503.05143, 2025. 3, 32

[154] J. Sun, Z. Hua, and Y. Xia, "Autoeval: A practical framework for autonomous evaluation of mobile agents," arXiv preprint arXiv:2503.02403, 2025. 3, 29, 30

[155] G. Liu, P. Zhao, L. Liu, Z. Chen, Y. Chai, S. Ren, H. Wang, S. He, and W. Meng, "Learnact: Few-shot mobile gui agent with a unified demonstration benchmark," arXiv preprint arXiv:2504.13805, 2025. 3, 11, 19, 29, 30, 31

[156] Y. Chai, H. Li, J. Zhang, L. Liu, G. Wang, S. Ren, S. Huang, and H. Li, "A3: Android agent arena for mobile gui agents," arXiv preprint arXiv:2501.01149, 2025. 3, 29, 30

[157] B. Wu, Y. Li, M. Fang, Z. Song, Z. Zhang, Y. Wei, and L. Chen, "Foundations and recent trends in multimodal mobile agents: A survey," arXiv preprint arXiv:2411.02006, 2024. 2

[158] S. Wang, W. Liu, J. Chen, W. Gan, X. Zeng, S. Yu, X. Hao, K. Shao, Y. Wang, and R. Tang, "Gui agents with foundation models: A comprehensive survey," arXiv preprint arXiv:2411.04890, 2024. 2

[159] C. Zhang, S. He, J. Qian, B. Li, L. Li, S. Qin, Y. Kang, M. Ma, Q. Lin, S. Rajmohan et al., "Large language model-brained gui agents: A survey," arXiv preprint arXiv:2411.18279, 2024. 2

[160] P. Kong, L. Li, J. Gao, K. Liu, T. F. Bissyandé, and J. Klein, "Automated testing of android apps: A systematic literature review," IEEE Transactions on Reliability, vol. 68, no. 1, pp. 45- 66, 2018. 3, 4

[161] B. Kirubakaran and V. Karthikeyani, "Mobile application testing-challenges and solution approach through automation," in 2013 International Conference on Pattern Recognition, Informatics and Mobile Engineering. IEEE, 2013, pp. 79-84. 3, 4

[162] D. Amalfitano, A. R. Fasolino, P. Tramontana, B. D. Ta, and A. M. Memon, "Mobiguitar: Automated model-based testing of mobile apps," IEEE software, vol. 32, no. 5, pp. 53-59, 2014. 3, 4

[163] M. Linares-Vásquez, K. Moran, and D. Poshyvanyk, "Continuous, evolutionary and large-scale: A new perspective for automated mobile app testing," in 2017 IEEE International Conference on Software Maintenance and Evolution (ICSME). IEEE, 2017, pp. 399-410. 3, 4

[164] Y. Zhao, B. Harrison, and T. Yu, "Dinodroid: Testing android apps using deep q-networks," ACM Transactions on Software Engineering and Methodology, vol. 33, no. 5, pp. 1-24, 2024. 3, 4

[165] G. Hecht, O. Benomar, R. Rouvoy, N. Moha, and L. Duchien, "Tracking the software quality of android applications along their evolution (t)," in 2015 30th IEEE/ACM International Conference on Automated Software Engineering (ASE). IEEE, 2015, pp. 236-247. 4

[166] S. Zein, N. Salleh, and J. Grundy, "A systematic mapping study of mobile application testing techniques," Journal of Systems and Software, vol. 117, pp. 334-356, 2016. 4

[167] C. S. Jensen, M. R. Prasad, and A. Møller, "Automated testing with targeted event sequence generation," in Proceedings of the 2013 International Symposium on Software Testing and Analysis, 2013, pp. 67-77. 4

[168] A. Machiry, R. Tahiliani, and M. Naik, "Dynodroid: An input generation system for android apps," in Proceedings of the 2013 9th Joint Meeting on Foundations of Software Engineering, 2013, pp. 224-234. 4

[169] D. Amalfitano, A. R. Fasolino, P. Tramontana, S. De Carmine, and A. M. Memon, "Using gui ripping for automated testing of android applications," in Proceedings of the 27th IEEE/ACM International Conference on Automated Software Engineering, 2012, pp. 258-261. 4

[170] P. Ladosz, L. Weng, M. Kim, and H. Oh, "Exploration in deep reinforcement learning: A survey," Information Fusion, vol. 85, pp. 1-22, 2022. 4

[171] J. Fan, Z. Wang, Y. Xie, and Z. Yang, "A theoretical analysis of deep q-learning," in Learning for dynamics and control. PMLR, 2020, pp. 486-489. 4

[172] F.-M. Luo, T. Xu, H. Lai, X.-H. Chen, W. Zhang, and Y. Yu, "A survey on model-based reinforcement learning," Science China Information Sciences, vol. 67, no. 2, p. 121101, 2024. 4

[173] R. Bridle and E. McCreath, "Inducing shortcuts on a mobile phone interface," in Proceedings of the 11th international conference on Intelligent user interfaces, 2006, pp. 327-329. 4

[174] T. Guerreiro, R. Gamboa, and J. Jorge, "Mnemonical body shortcuts: improving mobile interaction," in Proceedings of the 15th European conference on Cognitive ergonomics: the ergonomics of cool interaction, 2008, pp. 1-8. 4

[175] C. Kennedy and S. E. Everett, "Use of cognitive shortcuts in landline and cell phone surveys," Public Opinion Quarterly, vol. 75, no. 2, pp. 336-348, 2011. 4

[176] S. Agostinelli, A. Marrella, and M. Mecella, "Research challenges for intelligent robotic process automation," in Business Process Management Workshops: BPM 2019 International Workshops, Vienna, Austria, September 1-6, 2019, Revised Selected Papers 17. Springer, 2019, pp. 12-18. 4

[177] D. Pramod, "Robotic process automation for industry: adoption status, benefits, challenges and research agenda," Benchmarking: an international journal, vol. 29, no. 5, pp. 1562-1586, 2022. 4

[178] R. Syed, S. Suriadi, M. Adams, W. Bandara, S. J. Leemans, C. Ouyang, A. H. Ter Hofstede, I. Van De Weerd, M. T. Wynn, and H. A. Reijers, "Robotic process automation: contemporary themes and challenges," Computers in Industry, vol. 115, p. 103162, 2020. 4

[179] J. Clarke, J. Proudfoot, A. Whitton, M.-R. Birch, M. Boyd, G. Parker, V. Manicavasagar, D. Hadzi-Pavlovic, A. Fogarty et al., "Therapeutic alliance with a fully automated mobile phone and web-based intervention: secondary analysis of a randomized controlled trial," JMIR mental health, vol. 3, no. 1, p. e4656, 2016. 4

[180] T. J.-J. Li, A. Azaria, and B. A. Myers, "Sugilite: creating multimodal smartphone automation by demonstration," in Proceedings of the 2017 CHI conference on human factors in computing systems, 2017, pp. 6038-6049. 4


<!-- Meanless: 37 -->

[181] S. M. Patel and S. J. Pasha, "Home automation system (has) using android for mobile phone," International Journal Of Scientific Engeneering and Technology Research, ISSN, pp. 2319-8885, 2015. 4

[182] M. Asadullah and A. Raza, "An overview of home automation systems," in 2016 2nd international conference on robotics and artificial intelligence (ICRAI). IEEE, 2016, pp. 27-31. 4

[183] R. Majeed, N. A. Abdullah, I. Ashraf, Y. B. Zikria, M. F. Mushtaq, and M. Umer, "An intelligent, secure, and smart home automation system," Scientific Programming, vol. 2020, no. 1, p. 4579291, 2020. 4

[184] X. Liu, Y. Shi, C. Yu, C. Gao, T. Yang, C. Liang, and Y. Shi, "Understanding in-situ programming for smart home automation," Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies, vol. 7, no. 2, pp. 1-31, 2023. 4

[185] R. K. Kodali, S. C. Rajanarayanan, L. Boppana, S. Sharma, and A. Kumar, "Low cost smart home automation system using smart phone," in 2019 IEEE R10 humanitarian technology conference (R10- HTC)(47129). IEEE, 2019, pp. 120-125. 4

[186] R. K. Kodali and K. S. Mahesh, "Low cost implementation of smart home automation," in 2017 International Conference on Advances in Computing, Communications and Informatics (ICACCI). IEEE, 2017, pp. 461-466. 4

[187] S. Moreira, H. S. Mamede, and A. Santos, "Process automation using rpa-a literature review," Procedia Computer Science, vol. 219, pp. 244-254, 2023. 4

[188] C. Lamberton, D. Brigo, and D. Hoy, "Impact of robotics, rpa and ai on the insurance industry: challenges and opportunities," Journal of Financial Perspectives, vol. 4, no. 1, 2017. 4

[189] A. Meironke and S. Kuehnel, "How to measure rpa's benefits? a review on metrics, indicators, and evaluation methods of rpa benefit assessment," 2022. 4

[190] A. M. Tripathi, Learning Robotic Process Automation: Create Software robots and automate business processes with the leading RPA tool-UiPath. Packt Publishing Ltd, 2018. 5

[191] X. Ling, M. Gao, and D. Wang, "Intelligent document processing based on rpa and machine learning," in 2020 Chinese Automation Congress (CAC). IEEE, 2020, pp. 1349-1353. 5

[192] S. Agostinelli, M. Lupia, A. Marrella, and M. Mecella, "Reactive synthesis of software robots in rpa from user interface logs," Computers in Industry, vol. 142, p. 103721, 2022. 5

[193] H. V. Le, S. Mayer, M. Weiß, J. Vogelsang, H. Weingärtner, and N. Henze, "Shortcut gestures for mobile text editing on fully touch sensitive smartphones," ACM Transactions on Computer-Human Interaction (TOCHI), vol. 27, no. 5, pp. 1-38, 2020. 5

[194] A. M. Roffarello, A. K. Purohit, and S. V. Purohit, "Trigger-action programming for wellbeing: Insights from 6590 ios shortcuts," IEEE Pervasive Computing, 2024. 5

[195] V. Kepuska and G. Bohouta, "Next-generation of virtual personal assistants (microsoft cortana, apple siri, amazon alexa and google home)," in 2018 IEEE 8th annual computing and communication workshop and conference (CCWC). IEEE, 2018, pp. 99-103. 5

[196] B. R. Cowan, N. Pantidi, D. Coyle, K. Morrissey, P. Clarke, S. Al-Shehri, D. Earley, and N. Bandeira, "" what can i help you with?" infrequent users' experiences of intelligent personal assistants," in Proceedings of the 19th international conference on human-computer interaction with mobile devices and services, 2017, pp. 1-12. 5

[197] D. Anicic, P. Fodor, S. Rudolph, R. Stühmer, N. Stojanovic, and R. Studer, "A rule-based language for complex event processing and reasoning," in Web Reasoning and Rule Systems: Fourth International Conference, RR 2010, Bressanone/Brixen, Italy, September 22-24, 2010. Proceedings 4. Springer, 2010, pp. 42-57. 5

[198] N. Kang, B. Singh, Z. Afzal, E. M. van Mulligen, and J. A. Kors, "Using rule-based natural language processing to improve disease normalization in biomedical text," Journal of the American Medical Informatics Association, vol. 20, no. 5, pp. 876-881, 2013. 5

[199] N. Karanikolas, E. Manga, N. Samaridi, E. Tousidou, and M. Vas-silakopoulos, "Large language models versus natural language understanding and generation," in Proceedings of the 27th PanHellenic Conference on Progress in Computing and Informatics, 2023, pp. 278-290. 5, 7

[200] J. Fu, X. Zhang, Y. Wang, W. Zeng, and N. Zheng, "Understanding mobile gui: From pixel-words to screen-sentences," Neurocomput-ing, vol. 601, p. 128200, 2024. 5

[201] I. Banerjee, B. Nguyen, V. Garousi, and A. Memon, "Graphical user interface (gui) testing: Systematic mapping and repository," Information and Software Technology, vol. 55, no. 10, pp. 1679-1694, 2013. 5

[202] C. Chen, T. Su, G. Meng, Z. Xing, and Y. Liu, "From ui design image to gui skeleton: a neural machine translator to bootstrap mobile gui implementation," in Proceedings of the 40th International Conference on Software Engineering, 2018, pp. 665-676. 5

[203] J. Brich, M. Walch, M. Rietzler, M. Weber, and F. Schaub, "Exploring end user programming needs in home automation," ACM Transactions on Computer-Human Interaction (TOCHI), vol. 24, no. 2, pp. 1-35, 2017. 5

[204] J. Wu, X. Zhang, J. Nichols, and J. P. Bigham, "Screen parsing: Towards reverse engineering of ui models from screenshots," in The 34th Annual ACM Symposium on User Interface Software and Technology, 2021, pp. 470-483. 5

[205] T. Brown, B. Mann, N. Ryder, M. Subbiah, J. D. Kaplan, P. Dhari-wal, A. Neelakantan, P. Shyam, G. Sastry, A. Askell et al., "Language models are few-shot learners," Advances in neural information processing systems, vol. 33, pp. 1877-1901, 2020. 5, 7

[206] J. Kaplan, S. McCandlish, T. Henighan, T. B. Brown, B. Chess, R. Child, S. Gray, A. Radford, J. Wu, and D. Amodei, "Scaling laws for neural language models," arXiv preprint arXiv:2001.08361, 2020. 5

[207] T. Hagendorff, "Machine psychology: Investigating emergent capabilities and behavior in large language models using psychological methods," arXiv preprint arXiv:2303.13988, vol. 1, 2023. 5

[208] A. Vaswani, "Attention is all you need," Advances in Neural Information Processing Systems, 2017. 7

[209] A. Radford, "Improving language understanding by generative pre-training," 2018. 7

[210] J. Devlin, "Bert: Pre-training of deep bidirectional transformers for language understanding," arXiv preprint arXiv:1810.04805, 2018. 7

[211] W. X. Zhao, K. Zhou, J. Li, T. Tang, X. Wang, Y. Hou, Y. Min, B. Zhang, J. Zhang, Z. Dong et al., "A survey of large language models," arXiv preprint arXiv:2303.18223, 2023. 7

[212] Y. Chang, X. Wang, J. Wang, Y. Wu, L. Yang, K. Zhu, H. Chen, X. Yi, C. Wang, Y. Wang et al., "A survey on evaluation of large language models," ACM Transactions on Intelligent Systems and Technology, vol. 15, no. 3, pp. 1-45, 2024. 7

[213] S. Minaee, T. Mikolov, N. Nikzad, M. Chenaghlu, R. Socher, X. Amatriain, and J. Gao, "Large language models: A survey," arXiv preprint arXiv:2402.06196, 2024. 7

[214] B. Wang, X. Yue, and H. Sun, "Can chatgpt defend its belief in truth? evaluating llm reasoning via debate," arXiv preprint arXiv:2305.13160, 2023. 7

[215] L. Yuan, G. Cui, H. Wang, N. Ding, X. Wang, J. Deng, B. Shan, H. Chen, R. Xie, Y. Lin et al., "Advancing llm reasoning generalists with preference trees," arXiv preprint arXiv:2404.02078, 2024. 7

[216] C. H. Song, J. Wu, C. Washington, B. M. Sadler, W.-L. Chao, and Y. Su, "Llm-planner: Few-shot grounded planning for embodied agents with large language models," in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 2998-3009. 7, 12

[217] K. Valmeekam, M. Marquez, S. Sreedharan, and S. Kambhampati, "On the planning abilities of large language models-a critical investigation," Advances in Neural Information Processing Systems, vol. 36, pp. 75993-76005, 2023. 7

[218] W. Talukdar and A. Biswas, "Improving large language model (llm) fidelity through context-aware grounding: A systematic approach to reliability and veracity," arXiv preprint arXiv:2408.04023, 2024. 7

[219] R. Koike, M. Kaneko, and N. Okazaki, "Outfox: Llm-generated essay detection through in-context learning with adversarially generated examples," in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 38, no. 19, 2024, pp. 21258-21266. 7

[220] S. Zhang, Z. Zhang, K. Chen, X. Ma, M. Yang, T. Zhao, and M. Zhang, "Dynamic planning for llm-based graphical user interface automation," arXiv preprint arXiv:2410.00467, 2024. 8, 12

[221] G. E. Monahan, "State of the art-a survey of partially observable markov decision processes: theory, models, and algorithms," Management science, vol. 28, no. 1, pp. 1-16, 1982. 8

[222] M. T. Spaan, "Partially observable markov decision processes," in Reinforcement learning: State-of-the-art. Springer, 2012, pp. 387-414. 8

[223] I. Medhi, K. Toyama, A. Joshi, U. Athavankar, and E. Cutrell, "A comparison of list vs. hierarchical uis on mobile phones for nonliterate users," in Human-Computer Interaction-INTERACT 2013: 14th IFIP TC 13 International Conference, Cape Town, South Africa, September 2-6, 2013, Proceedings, Part II 14. Springer, 2013, pp. 497-504. 9


<!-- Meanless: 38 -->

[224] O. J. Räsänen and J. P. Saarinen, "Sequence prediction with sparse distributed hyperdimensional coding applied to the analysis of mobile phone use patterns," IEEE transactions on neural networks and learning systems, vol. 27, no. 9, pp. 1878-1889, 2015. 9

[225] G. Verma, R. Kaur, N. Srishankar, Z. Zeng, T. Balch, and M. Veloso, "Adaptagent: Adapting multimodal web agents with few-shot learning from human demonstrations," arXiv preprint arXiv:2411.13451, 2024. 10, 11, 31

[226] H. He, W. Yao, K. Ma, W. Yu, Y. Dai, H. Zhang, Z. Lan, and D. Yu, "Webvoyager: Building an end-to-end web agent with large multimodal models," arXiv preprint arXiv:2401.13919, 2024. 10

[227] B. Tang and K. G. Shin, "Steward: Natural language web automation," arXiv preprint arXiv:2409.15441, 2024. 10

[228] J. Yang, H. Zhang, F. Li, X. Zou, C. Li, and J. Gao, "Set-of-mark prompting unleashes extraordinary visual grounding in gpt-4v," 2023. [Online]. Available: https://arxiv.org/abs/2310.11441 10, 17

[229] J. Y. Koh, R. Lo, L. Jang, V. Duvvur, M. C. Lim, P.-Y. Huang, G. Neu-big, S. Zhou, R. Salakhutdinov, and D. Fried, "Visualwebarena: Evaluating multimodal agents on realistic visual web tasks," arXiv preprint arXiv:2401.13649, 2024. 10, 17

[230] R. Bonatti, D. Zhao, F. Bonacci, D. Dupont, S. Abdali, Y. Li, Y. Lu, J. Wagle, K. Koishida, A. Bucker et al., "Windows agent arena: Evaluating multi-modal os agents at scale," arXiv preprint arXiv:2409.08264, 2024. 11

[231] Y. Ge, Y. Ren, W. Hua, S. Xu, J. Tan, and Y. Zhang, "Llm as os, agents as apps: Envisioning aios, agents and the aios-agent ecosystem," arXiv e-prints, pp. arXiv-2312, 2023. 11

[232] K. Mei, Z. Li, S. Xu, R. Ye, Y. Ge, and Y. Zhang, "Aios: Llm agent operating system," arXiv e-prints, pp. arXiv-2403, 2024. 11

[233] Y. Deng, X. Zhang, W. Zhang, Y. Yuan, S.-K. Ng, and T.-S. Chua, "On the multi-turn instruction following for conversational web agents," arXiv preprint arXiv:2402.15057, 2024. 11

[234] T. Li, G. Li, Z. Deng, B. Wang, and Y. Li, "A zero-shot language agent for computer control with structured reflection," arXiv preprint arXiv:2310.08740, 2023. 12

[235] K. Gandhi, J.-P. Fränken, T. Gerstenberg, and N. Goodman, "Understanding social reasoning in language models with language models," Advances in Neural Information Processing Systems, vol. 36, 2024. 12

[236] Z. Chen, Y. Li, and K. Wang, "Optimizing reasoning abilities in large language models: A step-by-step approach," Authorea Preprints, 2024. 12

[237] A. Plaat, A. Wong, S. Verberne, J. Broekens, N. van Stein, and T. Back, "Reasoning with large language models, a survey," arXiv preprint arXiv:2407.11511, 2024. 12

[238] J. Wei, X. Wang, D. Schuurmans, M. Bosma, F. Xia, E. Chi, Q. V. Le, D. Zhou et al., "Chain-of-thought prompting elicits reasoning in large language models," Advances in neural information processing systems, vol. 35, pp. 24824-24837, 2022. 12, 16

[239] J. Y. Koh, S. McAleer, D. Fried, and R. Salakhutdinov, "Tree search for language model agents," arXiv preprint arXiv:2407.01476, 2024. 12

[240] W. E. Bishop, A. Li, C. Rawles, and O. Riva, "Latent state estimation helps ui agents to reason," arXiv preprint arXiv:2405.11120, 2024. 12

[241] N. Shinn, F. Cassano, A. Gopinath, K. Narasimhan, and S. Yao, "Reflexion: Language agents with verbal reinforcement learning," Advances in Neural Information Processing Systems, vol. 36, 2024. 12

[242] J. Pan, Y. Zhang, N. Tomlin, Y. Zhou, S. Levine, and A. Suhr, "Autonomous evaluation and refinement of digital agents," arXiv preprint arXiv:2404.06474, 2024. 12

[243] P. Duan, C.-Y. Cheng, G. Li, B. Hartmann, and Y. Li, "Uicrit: Enhancing automated design evaluation with a ui critique dataset," in Proceedings of the 37th Annual ACM Symposium on User Interface Software and Technology, 2024, pp. 1-17. 12

[244] N. Patil, D. Bhole, and P. Shete, "Enhanced ui automator viewer with improved android accessibility evaluation features," in 2016 International Conference on Automatic Control and Dynamic Optimization Techniques (ICACDOT). IEEE, 2016, pp. 977-983. 12

[245] G. Lodi, "Xctest introduction," in Test-Driven Development in Swift: Compile Better Code with XCTest and TDD. Springer, 2021, pp. 13-25. 12

[246] S. Singh, R. Gadgil, and A. Chudgor, "Automated testing of mobile applications using scripting technique: A study on appium," International Journal of Current Engineering and Technology (IJCET), vol. 4, no. 5, pp. 3627-3630, 2014. 12

[247] U. Gundecha, Selenium Testing Tools Cookbook. Packt Publishing Ltd, 2015. 12

[248] C. Sinclair, "The role of selenium in mobile application testing." 12

[249] A. Torreno, E. Onaindia, A. Komenda, and M. Štolba, "Cooperative multi-agent planning: A survey," ACM Computing Surveys (CSUR), vol. 50, no. 6, pp. 1-32, 2017. 12

[250] A. Dorri, S. S. Kanhere, and R. Jurdak, "Multi-agent systems: A survey," Ieee Access, vol. 6, pp. 28573-28593, 2018. 12

[251] R. Gong, Q. Huang, X. Ma, H. Vo, Z. Durante, Y. Noda, Z. Zheng, S.-C. Zhu, D. Terzopoulos, L. Fei-Fei et al., "Mindagent: Emergent gaming interaction," arXiv preprint arXiv:2309.09971, 2023. 12

[252] F. Chen, W. Ren et al., "On the control of multi-agent systems: A survey," Foundations and Trends® in Systems and Control, vol. 6, no. 4, pp. 339-499, 2019. 12

[253] Y. Talebirad and A. Nadiri, "Multi-agent collaboration: Harnessing the power of intelligent llm agents," arXiv preprint arXiv:2306.03314, 2023. 12

[254] Q. Wu, G. Bansal, J. Zhang, Y. Wu, S. Zhang, E. Zhu, B. Li, L. Jiang, X. Zhang, and C. Wang, "Autogen: Enabling next-gen llm applications via multi-agent conversation framework," arXiv preprint arXiv:2308.08155, 2023. 12

[255] W. Chen, Y. Su, J. Zuo, C. Yang, C. Yuan, C.-M. Chan, H. Yu, Y. Lu, Y.-H. Hung, C. Qian et al., "Agentverse: Facilitating multi-agent collaboration and exploring emergent behaviors," in The Twelfth International Conference on Learning Representations, 2023. 12

[256] H. Li, Y. Q. Chong, S. Stepputtis, J. Campbell, D. Hughes, M. Lewis, and K. Sycara, "Theory of mind for multi-agent collaboration via large language models," arXiv preprint arXiv:2310.10701, 2023. 12

[257] Z. Liu, Y. Zhang, P. Li, Y. Liu, and D. Yang, "A dynamic llm-powered agent network for task-oriented agent collaboration," in First Conference on Language Modeling, 2024. 12

[258] X. Li, S. Wang, S. Zeng, Y. Wu, and Y. Yang, "A survey on llm-based multi-agent systems: workflow, infrastructure, and challenges," Vicinagearth, vol. 1, no. 1, p. 9, 2024. 12

[259] K.-T. Tran, D. Dao, M.-D. Nguyen, Q.-V. Pham, B. O'Sullivan, and H. D. Nguyen, "Multi-agent collaboration mechanisms: A survey of llms," arXiv preprint arXiv:2501.06322, 2025. 12

[260] D. Yin, F. Brahman, A. Ravichander, K. Chandu, K.-W. Chang, Y. Choi, and B. Y. Lin, "Agent lumos: Unified and modular training for open-source language agents," in Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 2024, pp. 12380-12403. 14

[261] Y. Zhang, Z. Ma, Y. Ma, Z. Han, Y. Wu, and V. Tresp, "Webpilot: A versatile and autonomous multi-agent system for web task execution with strategic exploration," arXiv preprint arXiv:2408.15978, 2024. 14

[262] S. Yao, D. Yu, J. Zhao, I. Shafran, T. Griffiths, Y. Cao, and K. Narasimhan, "Tree of thoughts: Deliberate problem solving with large language models," Advances in Neural Information Processing Systems, vol. 36, 2024. 16

[263] W. Chen, X. Ma, X. Wang, and W. W. Cohen, "Program of thoughts prompting: Disentangling computation from reasoning for numerical reasoning tasks," arXiv preprint arXiv:2211.12588, 2022. 16

[264] Y. Yang, X. Yang, S. Li, C. Lin, Z. Zhao, C. Shen, and T. Zhang, "Security matrix for multimodal agents on mobile devices: A systematic and proof of concept study," arXiv preprint arXiv:2407.09295, 2024. 16

[265] Z. Zhang, Y. Yao, A. Zhang, X. Tang, X. Ma, Z. He, Y. Wang, M. Gerstein, R. Wang, G. Liu et al., "Igniting language intelligence: The hitchhiker's guide from chain-of-thought reasoning to language agents," arXiv preprint arXiv:2311.11797, 2023. 16

[266] J. Li, D. Li, S. Savarese, and S. Hoi, "Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models," in International conference on machine learning. PMLR, 2023, pp. 19730-19742. 17

[267] Q. Ye, H. Xu, G. Xu, J. Ye, M. Yan, Y. Zhou, J. Wang, A. Hu, P. Shi, Y. Shi et al., "mplug-owl: Modularization empowers large language models with multimodality," arXiv preprint arXiv:2304.14178, 2023. 17

[268] W. Wang, Q. Lv, W. Yu, W. Hong, J. Qi, Y. Wang, J. Ji, Z. Yang, L. Zhao, X. Song et al., "Cogvlm: Visual expert for pretrained language models," arXiv preprint arXiv:2311.03079, 2023. 17


<!-- Meanless: 39 -->

[269] J. Bai, S. Bai, S. Yang, S. Wang, S. Tan, P. Wang, J. Lin, C. Zhou, and J. Zhou, "Qwen-vl: A versatile vision-language model for understanding, localization, text reading, and beyond," arXiv preprint arXiv:2308.12966, 2023. 17

[270] H. Liu, C. Li, Q. Wu, and Y. J. Lee, "Visual instruction tuning," Advances in neural information processing systems, vol. 36, 2024. 17

[271] P. Wang, S. Bai, S. Tan, S. Wang, Z. Fan, J. Bai, K. Chen, X. Liu, J. Wang, W. Ge, Y. Fan, K. Dang, M. Du, X. Ren, R. Men, D. Liu, C. Zhou, J. Zhou, and J. Lin, "Qwen2-vl: Enhancing vision-language model's perception of the world at any resolution," arXiv preprint arXiv:2409.12191, 2024. 17

[272] Z. Chen, J. Wu, W. Wang, W. Su, G. Chen, S. Xing, M. Zhong, Q. Zhang, X. Zhu, L. Lu et al., "Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks," in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 24 185-24 198. 17

[273] Z. Chen, W. Wang, H. Tian, S. Ye, Z. Gao, E. Cui, W. Tong, K. Hu, J. Luo, Z. Ma et al., "How far are we to gpt-4v? closing the gap to commercial multimodal models with open-source suites," arXiv preprint arXiv:2404.16821, 2024. 17

[274] L. Zheng, R. Wang, X. Wang, and B. An, "Synapse: Trajectory-as-exemplar prompting with memory for computer control," in The Twelfth International Conference on Learning Representations, 2023. 17

[275] R. Varghese and M. Sambath, "Yolov8: A novel object detection algorithm with enhanced performance and robustness," in 2024 International Conference on Advances in Data Engineering and Intelligent Computing Systems (ADICS). IEEE, 2024, pp. 1-6. 18

[276] Y. Du, C. Li, R. Guo, X. Yin, W. Liu, J. Zhou, Y. Bai, Z. Yu, Y. Yang, Q. Dang et al., "Pp-ocr: A practical ultra lightweight ocr system," arXiv preprint arXiv:2009.09941, 2020. 18

[277] H.-M. Xu, Q. Chen, L. Wang, and L. Liu, "Attention-driven gui grounding: Leveraging pretrained multimodal large language models without fine-tuning," arXiv preprint arXiv:2412.10840, 2024. 21

[278] L. P. Kaelbling, M. L. Littman, and A. W. Moore, "Reinforcement learning: A survey," Journal of artificial intelligence research, vol. 4, pp. 237-285, 1996. 24

[279] J. Zheng, L. Wang, F. Yang, C. Zhang, L. Mei, W. Yin, Q. Lin, D. Zhang, S. Rajmohan, and Q. Zhang, "Vem: Environment-free exploration for training gui agent with value environment model," arXiv preprint arXiv:2502.18906, 2025. 25

[280] Y.-H. Peng, F. Huq, Y. Jiang, J. Wu, X. Y. Li, J. P. Bigham, and A. Pavel, "Dreamstruct: Understanding slides and user interfaces via synthetic data generation," in European Conference on Computer Vision. Springer, 2024, pp. 466-485. 28

[281] Q. Sun, K. Cheng, Z. Ding, C. Jin, Y. Wang, F. Xu, Z. Wu, C. Jia, L. Chen, Z. Liu et al., "Os-genesis: Automating gui agent trajectory construction via reverse task synthesis," arXiv preprint arXiv:2412.19723, 2024. 28

[282] H. Su, R. Sun, J. Yoon, P. Yin, T. Yu, and S. Ö. Arık, "Learn-by-interact: A data-centric framework for self-adaptive agents in realistic environments," arXiv preprint arXiv:2501.10893, 2025. 28

[283] W. Wang, Z. Yu, W. Liu, R. Ye, T. Jin, S. Chen, and Y. Wang, "Fedmobileagent: Training mobile agents using decentralized self-sourced data from diverse users," arXiv preprint arXiv:2502.02982, 2025. 28, 32

[284] O. Berkovitch, S. Caduri, N. Kahlon, A. Efros, A. Caciularu, and I. Dagan, "Identifying user goals from ui trajectories," arXiv preprint arXiv:2406.14314, 2024. 28

[285] K. Q. Lin, L. Li, D. Gao, Q. Wu, M. Yan, Z. Yang, L. Wang, and M. Z. Shou, "Videogui: A benchmark for gui automation from instructional videos," arXiv preprint arXiv:2406.10227, 2024. 31

[286] W. Chen and Z. Li, "Octopus v2: On-device language model for super agent," arXiv preprint arXiv:2404.01744, 2024. 31

[287] H. Wen, S. Tian, B. Pavlov, W. Du, Y. Li, G. Chang, S. Zhao, J. Liu, Y. Liu, Y.-Q. Zhang et al., "Autodroid-v2: Boosting slm-based gui agents via code generation," arXiv preprint arXiv:2412.18116, 2024. 31

[288] F. Wang, Z. Zhang, X. Zhang, Z. Wu, T. Mo, Q. Lu, W. Wang, R. Li, J. Xu, X. Tang et al., "A comprehensive survey of small language models in the era of large language models: Techniques, enhancements, applications, collaboration with llms, and trustworthiness," arXiv preprint arXiv:2411.03350, 2024. 31

[289] F. Huq, J. P. Bigham, and N. Martelaro, "What's important here?: Opportunities and challenges of llm in retrieving information from web interface," R0-FoMo: Robustness of Few-shot and Zero-shot Learning in Large Foundation Models. 31

[290] W. Li, F.-L. Hsu, W. Bishop, F. Campbell-Ajala, M. Lin, and O. Riva, "Uinav: A practical approach to train on-device automation agents," in Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 6: Industry Track), 2024, pp. 36-51. 31

[291] C. H. Wu, J. Y. Koh, R. Salakhutdinov, D. Fried, and A. Raghu-nathan, "Adversarial attacks on multimodal agents," arXiv preprint arXiv:2406.12814, 2024. 32

[292] W. Yang, X. Bi, Y. Lin, S. Chen, J. Zhou, and X. Sun, "Watch out for your agents! investigating backdoor threats to llm-based agents," arXiv preprint arXiv:2402.11208, 2024. 32

[293] Y. Wang, D. Xue, S. Zhang, and S. Qian, "Badagent: Inserting and activating backdoor attacks in llm agents," arXiv preprint arXiv:2406.03007, 2024. 32

[294] Y. Chen, X. Hu, K. Yin, J. Li, and S. Zhang, "Aeia-mn: Evaluating the robustness of multimodal llm-powered mobile agents against active environmental injection attacks," arXiv preprint arXiv:2502.13053, 2025. 32