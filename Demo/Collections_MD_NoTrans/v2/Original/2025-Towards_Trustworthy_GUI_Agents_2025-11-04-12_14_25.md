# Towards Trustworthy GUI Agents: A Survey

Yucheng Shi ${}^{1}$ , Wenhao Yu ${}^{2}$ , Wenlin Yao ${}^{3}$ , Wenhu Chen ${}^{4}$ , Ninghao Liu ${}^{1}$

${}^{1}$ University of Georgia ${}^{2}$ Tencent AI Seattle Lab

${}^{3}$ Amazon ${}^{4}$ University of Waterloo

## Abstract

GUI agents, powered by large foundation models, can interact with digital interfaces, enabling various applications in web automation, mobile navigation, and software testing. However, their increasing autonomy has raised critical concerns about their security, privacy, and safety. This survey examines the trustworthiness of GUI agents in five critical dimensions: security vulnerabilities, reliability in dynamic environments, transparency and explainability, ethical considerations, and evaluation methodologies. We also identify major challenges such as vulnerability to adversarial attacks, cascading failure modes in sequential decision-making, and a lack of realistic evaluation benchmarks. These issues not only hinder real-world deployment but also call for comprehensive mitigation strategies beyond task success. As GUI agents become more widespread, establishing robust safety standards and responsible development practices is essential. This survey provides a foundation for advancing trustworthy GUI agents through systematic understanding and future research.

## 1 Introduction

Large language models (LLMs) and large multimodal models (LMMs) have rapidly evolved from question answering tools to agents capable of interacting with graphical user interfaces (GUIs) through clicks and on-screen parsing (Nguyen et al., 2024a; Wang et al., 2024c; Xie et al., 2024). Deployed on websites, desktops, mobile apps, and diverse software environments, these GUI agents promise wide-ranging applications from automated testing and e-commerce to assistive technologies for users with disabilities (Zhao et al., 2024; Cuadra et al., 2024). Their ability to interpret dynamic interfaces, understand multimodal inputs, and execute precise actions is reshaping how large foundation models assist human operators in routine digital tasks.

As GUI agents become more capable and begin to play a more significant role in real-world applications, ensuring their trustworthiness has become increasingly critical. Compared to traditional NLP tasks where inputs and outputs are relatively static and limited to textual data, GUI agents can operate in dynamic environments with inputs and outputs in different modalities. Although this flexibility improves utility, it also introduces new risks, which makes security, reliability, and transparency critical for responsible deployment (Arnold and Tilton, 2024; Ma et al., 2024). However, existing research on GUI agents focuses mainly on functional performance metrics, such as task completion rates, while often overlooking essential aspects like security, reliability, and transparency (Arnold and Tilton, 2024; Ma et al., 2024). This oversight poses significant risks, especially in high-stakes environments where these agents operate. Several emerging attacks have exposed these risks: adversarial image perturbations can deceive perception modules (Wu et al., 2025a), malicious webpage elements can manipulate agent behavior (Wu et al., 2024a), and screenshot-based navigation can inadvertently expose sensitive user data (Chen et al., 2024a).

Beyond these threats, the broader social implications are equally pressing. As the brain of GUI agents, the trustworthiness of LLMs and LMMs is crucial because it directly impacts the outcomes of GUI interactions (Liu et al., 2023c; Weidinger et al., 2022; Wu et al., 2024b). In critical domains such as finance and healthcare, trustworthy LLMs and LMMs ensure that decisions made by GUI agents are secure, ethical, transparent, and aligned with human values. Addressing these challenges requires assessing and mitigating risks from different aspects. Here, we categorize the trustworthiness in GUI agents into five key areas:

1. Security: Protecting agents against adversarial manipulation, unauthorized command execution, and data leaks. For instance, WebPI (Wu et al., 2024a) demonstrates how hidden HTML elements can mislead agents into executing unintended actions, posing security risks.

<!-- Meanless: arXiv:2503.23434v1 [cs.LG] 30 Mar 2025 -->


<!-- Media -->

<!-- figureText: 2025)<br>Authorized Agents (South et al., 2025)<br>PC-Agent<br>AEIA-MN Chenetally, 2021<br>Protecting users from themselve (Ngong et al., 2025)<br>2024)<br>CoCA<br>AdvWeb (Xueta), 2024a<br>AdvWeb<br>PUPA<br>ARE WuetaL, 2024<br>CASA<br>BC-LLMAger (Guanetal, 2020)<br>ELA<br>OS-ATLAS Musetal, 2024C)<br>Caution4E<br>(Liaoetall, 202<br>Injeckent<br>ST-WebAger<br>World Web Agent (Chae et al., 2024a)<br>WebPl<br>mprompte (Fuetal, 2024)<br>2023<br>Blockchain Agent<br>Harmlessness and Reliability<br>AHA<br>Security and Privacy<br>(2022)<br>Ethical Implications<br>Benchmarks<br>Explainability and Transparency<br>Trustworthy GUI Agents Evolutionary Tree -->

<img src="https://cdn.noedgeai.com/bo_d41gpibef24c73d3u7rg_1.jpg?x=182&y=189&w=1282&h=449&r=0"/>

Figure 1: An evolutionary tree of research on trustworthy GUI agents. Each branch represents a research direction, with notable works color-coded by their focus area, demonstrating how the field has evolved toward more comprehensive trustworthiness considerations. This figure is adapted from this repo.

<!-- Media -->

2. Reliability: Ensuring GUI agents function correctly across dynamic interfaces with reliable response. Studies on multimodal agent safety (Liu et al., 2023b) highlight risks where GUI agents may misinterpret visual cues, leading to unsafe or unintended interactions.

3. Explainability: Making agent decision-making processes more interpretable and user-friendly. Systems like EBC-LLMAgent (Guan et al., 2024) enhance transparency by learning from user demos to generate clear and interpretable action sequences with corresponding UI mappings and rationales.

4. Ethical Alignment: Ensuring agents adhere to human values and cultural norms. CASA (Qiu et al., 2024) evaluates agents on social and ethical considerations, emphasizing fairness in decision-making across diverse user populations.

5. Evaluation: Developing rigorous testing methods to assess GUI agent behavior under real-world conditions. ST-WebAgentBench (Levy et al., 2024) evaluates policy compliance and risk mitigation strategies for web-based agents.

Figure 1 illustrates the evolution of research across these dimensions. While previous surveys (Nguyen et al., 2024a; Wang et al., 2024a; Zhang et al., 2024a; Hu et al.; Liu et al., 2025b) primarily focus on the task performance of GUI agents, our work highlights less-explored issues like security, reliability, and transparency, and emerging mitigation strategies for these issues. Our discussion begins with an overview of GUI agent architectures and fundamental capabilities (Section 2). We then examine security and privacy challenges (Section 3) and strategies for enhancing reliability and harmlessness (Section 4). Next, we discuss the importance of explainability and transparency (Section 5) and outline ethical considerations for responsible deployment (Section 6). We conclude by reviewing evaluation methodologies (Section 7) and highlighting future research directions (Section 8). Overall, this survey shifts the focus from task success to holistic trustworthiness, offering researchers and developers insights into the risks, challenges, and solutions for creating secure and responsible GUI agents.

## 2 Foundation of GUI Agents

GUI agents leverage large foundation models to integrate perception, reasoning, planning, and execution, enabling interaction with user interfaces in a human-like manner. This section outlines their key components, applications, and challenges.

A standard agent pipeline includes multimodal perception, reasoning and planning (task decomposition), and interaction mechanisms (clicks, text entries, and other UI actions) (Wright, 2024; Zhou et al., 2023). For perception, some rely on accessibility APIs, while others parse HTML/DOM structures or process raw screenshots. Hybrid approaches combine these methods for more reliable understanding (Wu et al., 2024c; Nong et al., 2024; Yang et al., 2024; Deng et al., 2023). For interaction, agents perform tasks through replicating human-like interactions, such as clicking or typing (Koh et al., 2024a). Beyond the above perception and interaction, effective task decomposition is a core capability for GUI agents to navigate complex workflows and adapt to dynamic interfaces. A structured planning mechanism allows agents to decompose multi-step tasks and execute actions reliably across diverse environments (Gu et al., 2024a; Zhu et al., 2025; Koh et al., 2024b).


GUI agents can serve diverse applications. In mobile settings, they automate navigation and data entry via hierarchical planning (Nong et al., 2024; Zhu et al., 2025). On the web, they can support tasks such as automated testing, phishing detection, and e-commerce applications (Cao et al., 2024; Wang et al., 2024b; Gu et al., 2024b). In specialized domains like healthcare and education, they assist multimodal reasoning while ensuring privacy and accessibility (Cuadra et al., 2024; Arnold and Tilton, 2024; Srinivas et al., 2024).

Despite recent advances, key challenges remain. Agents are still vulnerable to adversarial multimodal inputs, which can trigger unpredictable behavior (Gao et al., 2024; Ma et al., 2024). They also struggle to generalize to unfamiliar interfaces, highlighting the need for stronger robustness (Kim et al., 2024b). Additionally, balancing real-time performance with safety remains an ongoing challenge, necessitating more efficient architectures (Shen et al., 2024; Nguyen et al., 2024b).

As GUI agents are increasingly deployed in real-world scenarios, addressing challenges related to security, reliability, explainability, and ethical alignment becomes essential. The following section examines these dimensions in depth, and we include a brief overview of key dimensions for building trustworthy GUI agents in Figure 2.

## 3 Security and Privacy

Security and privacy are central concerns for GUI agents. This section first outlines significant attacks and vulnerabilities that arise when agents interact with graphical interfaces. It then addresses risks surrounding user data, and finally discusses defense strategies and open problems. Table 1 summarizes key threats and defenses.

### 3.1 Attacks and Vulnerabilities

The interactive nature of GUI agents gives rise to novel exploits, including adversarial image perturbations and malicious prompt injections embedded in webpages (Janowczyk et al., 2024). Attacks such as Imprompter (Fu et al., 2024) manipulate a single product image to hijack an agent's actions with high success rates. Browser-based jailbreak-ing also emerges, whereby refusal-trained LLMs inadvertently execute harmful behaviors in non-chat contexts (Kumar et al., 2024). Some studies reveal that malicious instructions can be hidden in website structures, where compromised webpages consistently mislead the agent (Wu et al., 2024a). AdvWeb demonstrates how black-box adversarial prompts injected into web pages can mislead web agents into executing unintended actions while remaining undetectable to users (Xu et al., 2024a).

On mobile devices, multiple attack paths target both the perception and reasoning modules of multimodal agents, emphasizing the breadth of vulnerabilities (Yang et al., 2024). Similarly, AEIA-MN shows that mobile GUI agents are highly vulnerable to environmental injection attacks, where malicious elements disguised as system features disrupt decision-making with up to ${93}\%$ success rates (Chen et al., 2025).

Altogether, these exploits demonstrate that seemingly benign elements, such as small visual perturbations or hidden HTML code, can manipulate complex agent pipelines. Because GUI agents operate across modalities and maintain hidden internal state, such threats can slip through and spread across components over time (Wu et al., 2025a). Ensuring security thus requires a holistic view of the GUI agent's entire workflow.

### 3.2 Privacy Risks

Beyond security exploits, GUI agents raise pressing privacy concerns by accessing sensitive personal or enterprise data through visual and textual interfaces (Chen et al., 2024a; Zhang et al., 2024b). Screenshot-based perception can be particularly sensitive, potentially exposing private details without user awareness. Furthermore, when agents run in the cloud or on remote servers, the risk of data leakage increases (Gan et al., 2024; Xu et al., 2024b). These risks become especially critical when GUI agents interact with regulated domains such as finance and healthcare, where the exposure of Personally Identifiable Information (PII), including names, addresses, and financial details, can have severe consequences.

Recent studies highlight emerging threats, such as Environmental Injection Attacks (EIA), which covertly manipulate web environments to extract PII from GUI agents with up to 70% success rates (Liao et al., 2024). Similarly, web-enabled LLM agents have been found to enhance cyber-attacks, such as automated PII harvesting, impersonation post generation, and spear-phishing (Kim et al., 2024a), which reveals critical gaps in existing security measures. Beyond direct attacks, users may also inadvertently disclose sensitive information during normal interactions with GUI agents, underscoring the need for contextual privacy protections (Ngong et al., 2025).


<!-- Media -->

<!-- figureText: Build Trustworthy GUI Agents<br>Foundation Ability<br>Multimodal Perception<br>Eval & Metrics<br>Metrics<br>Eval Techniques<br>Hierarchical Planning<br>2, A<br>Benchmarks<br>$\delta$ б г<br>World Models<br>Malicious Attacks<br>Reliability<br>Traceable Steps<br>Social Awareness<br>Guidelines<br>Security Threats<br>Privacy Risks<br>Harmful ness<br>Content Safety<br>Chain-of-Thought<br>0.26<br>Policy Compliance<br>XAI<br>Ethics<br>User Alignment<br>User-Focused Expl.<br>Defenses -->

<img src="https://cdn.noedgeai.com/bo_d41gpibef24c73d3u7rg_3.jpg?x=190&y=186&w=1266&h=405&r=0"/>

Figure 2: Overview of key dimensions for building trustworthy GUI agents, highlighting foundational abilities, evaluation metrics, security threats, reliability, harmfulness, explainability, transparency, and ethical implications.

<!-- Media -->

### 3.3 Defenses and Mitigation Strategies

Researchers have proposed multiple approaches to mitigate security and privacy risks in GUI agents. Input validation and prompt injection detection aim to filter out unsafe content before it reaches the core model, as demonstrated by Sharma et al. (2024). Other work introduces specialized guardrail agents that intercept and inspect commands generated by primary agents, blocking disallowed actions (Xi-ang et al., 2024). Some frameworks leverage adversarial training or visual analytics systems, like AdversaFlow (Deng et al., 2024), to identify vulnerabilities collaboratively. AutoDroid enhances mobile GUI automation by integrating language models with dynamic UI analysis, enabling scalable, hands-free task execution across arbitrary Android apps without manual effort (Wen et al., 2024). Similarly, G-Safeguard applies graph-based anomaly detection to multi-agent systems, mitigating prompt injection attacks and securing agent collaboration (Wang et al., 2025).

To address privacy, solutions like CLEAR (Chen et al., 2024a) analyze user-provided data and privacy policies to highlight potential leakages. Others advocate secure sandboxing, local processing, and advanced authentication to constrain agent permissions (Zhang et al., 2024b; Gu et al., 2024a). PAPILLON proposes a privacy-conscious delegation framework that selectively routes queries between local and proprietary LLMs to minimize sensitive data exposure while maintaining high response quality (Siyan et al., 2024).

Recent commercial implementations of GUI agent frameworks also provide valuable insights into multi-layered defense strategies. For example, OpenAI's CUA employs a comprehensive defense-in-depth approach. This includes preventative measures such as website blocklists and refusal training, interactive safeguards like user confirmations for critical actions, and detection systems for real-time moderation and monitoring of suspicious content (OpenAI, 2025). This strategy recognizes that perfect prevention is unattainable and instead focuses on using complementary systems to gradually reduce risk. On the other hand, Anthropic advises limiting computer use to secure environments, such as virtual machines with minimal privileges, to mitigate ongoing vulnerabilities to jailbreaking and prompt injection (Anthropic, 2025).

### 3.4 Future Directions

Securing GUI agents requires balanced solutions that protect users while maintaining usability. Three promising directions deserve exploration:

Smarter Defense Tools: We need lightweight, real-time mechanisms to detect hidden attacks like those demonstrated in AdvWeb (Xu et al., 2024a) and Imprompter (Fu et al., 2024). Browser extensions could sanitize webpage elements before agents process them, while mobile applications might verify UI elements against device sensors to detect overlay attacks similar to those in AEIA-MN (Chen et al., 2025). Simple visual filters could automatically blur sensitive information on screens before agents capture screenshots, preventing data leakage without complex infrastructure changes, addressing concerns raised by Chen et al. (2024a). In parallel, recent advances such as ZIP (Shi et al., 2023a) demonstrate how zero-shot image purification techniques can effectively defend against visual backdoor attacks, offering a direction for mitigating image-based threats in GUI agents.


<!-- Media -->

<table><tr><td>Approach/Attack</td><td>Key Characteristics</td></tr><tr><td colspan="2">Malicious Attack & Vulnerabilities</td></tr><tr><td>Imprompter (Fu et al., 2024)</td><td>Hijacks agent actions through modified product images</td></tr><tr><td>Browser-based jailbreaking (Kumar et al., 2024)</td><td>Induces harmful behaviors in non-chat environments</td></tr><tr><td>WebPI (Wu et al., 2024a)</td><td>Embeds malicious commands in webpage structures</td></tr><tr><td>AdvWeb (Xu et al., 2024a)</td><td>Injects adversarial prompts with high user undetectability</td></tr><tr><td>AEIA-MN (Chen et al., 2025)</td><td>Disguises malicious elements as system features (93% success)</td></tr><tr><td>ARE (Wu et al., 2025a)</td><td>Propagates threats across agent module boundaries</td></tr><tr><td colspan="2">Privacy Risk</td></tr><tr><td>Screenshot leakage (Chen et al., 2024a)</td><td>Captures sensitive information in interface snapshots</td></tr><tr><td>EIA (Liao et al., 2024)</td><td>Extracts PII with up to 70% success rate</td></tr><tr><td>Agent-enabled cyberattacks (Kim et al., 2024a)</td><td>Facilitates PII harvesting and spear-phishing</td></tr><tr><td>Cloud-based processing (Gan et al., 2024)</td><td>Amplifies exposure risk in distributed architectures</td></tr><tr><td>Protecting Users (Ngong et al., 2025)</td><td>Reveals sensitive information during normal operations</td></tr><tr><td colspan="2">Defense & Mitigation Approaches</td></tr><tr><td>GuardAgent (Xiang et al., 2024)</td><td>Intercepts and blocks disallowed agent actions</td></tr><tr><td>AdversaFlow (Deng et al., 2024)</td><td>Identifies vulnerabilities through collaborative analysis</td></tr><tr><td>AutoDroid (Wen et al., 2024)</td><td>Automates tasks by combining LLMs with dynamic UI analysis</td></tr><tr><td>CLEAR (Chen et al., 2024a)</td><td>Analyzes data against privacy policies</td></tr><tr><td>PAPILLON (Siyan et al., 2024)</td><td>Minimizes exposure while maintaining response quality</td></tr><tr><td>Input validation (Sharma et al., 2024)</td><td>Screens unsafe content before model processing</td></tr></table>

Table 1: Taxonomy of Security and Privacy Considerations for GUI Agents. The table presents three key dimensions: (1) attacks and vulnerabilities exploiting multimodal interfaces, (2) privacy risk mechanisms that can expose sensitive information, and (3) defense and mitigation approaches aimed at protecting agent operations and user data.

<!-- Media -->

User-Controlled Privacy: Drawing inspiration from mobile permissions, future GUI agents should request specific, time-limited access to data (e.g., "view this webpage for 5 minutes"). Combined with local models that automatically redact personal information and clear activity logs, this approach would give users meaningful control while preserving convenience, as suggested by Zhang et al. (2024b). Users should understand what their agent sees and how their data is used.

Connected Defense Layers: Since GUI agents process multiple types of information (images, text, system states), defenses should verify consistency across channels (Wu et al., 2025a). For instance, an agent should cross-check text from a button image with the underlying HTML to catch any possible tampering. In high-stakes scenarios like payments or accessing healthcare data, agents should leverage a hardware-based security checking approach and require explicit user confirmation for any suspicious or sensitive actions, following ideas similar to those in Xiang et al. (2024).

## 4 Reliability and Harmlessness

This section examines how GUI agents handle visual hallucination, inappropriate content, and alignment with human values. They are core challenges for building both robust and safe GUI agents.

### 4.1 Reliability

Ensuring stable and accurate interaction with visual interfaces is essential for GUI agents, especially in tackling the challenge of hallucination, where agents generate actions or interpretations that do not match the visual content (Bai et al., 2024; Chen et al., 2024d). Such errors can include fabricated UI elements, incorrect readings of interface components, or lapses in visual focus, all of which can lead to unreliable behavior (Liu et al., 2023a; Jiang et al., 2024a; Yu et al., 2024). Furthermore, recent work (Ma et al., 2024) also reveals that even in benign, non-malicious environments, multimodal GUI agents are vulnerable to environmental distractions that undermine their reliability.

To mitigate these issues, several strategies have been developed. Opera introduces an over-trust penalty to prevent reliance on misleading summary tokens (Huang et al., 2024a), while Volcano employs self-feedback for natural language correction (Lee et al., 2023). Contrastive learning techniques help distinguish between hallucinative and non-hallucinative text, enhancing model robustness (Jiang et al., 2024a).


Real-time detection frameworks like UNIHD validate outputs against visual evidence, and methods such as Residual Visual Decoding address "hallucination snowballing" by revising outputs with residual visual input (Chen et al., 2024d; Zhong et al., 2024). Specialized datasets like LRV-Instruction and M-HalDetect further aid in reducing hallucination rates by providing targeted training resources (Liu et al., 2023a; Gunjal et al., 2023). Multi-agent systems also offer promising solutions by combining self-correction, external feedback, and agent debate to maintain accurate grounding in complex interactions (Yu et al., 2024).

### 4.2 Content Safety

Because GUI agents can generate and display multimodal content, ensuring safe and appropriate outputs is critical (Gao et al., 2024; Liu et al., 2023b). Image-based manipulations may prompt harmful or toxic responses, undermining the base LLM's alignment. Tailored calibration approaches, such as CoCA (Gao et al., 2024), attempt to restore the model's original safety guardrails under multimodal contexts. Recent work by Zou et al. (2024) introduces "circuit breakers" that can interrupt GUI agents when generating harmful outputs, functioning effectively even against sophisticated adversarial attacks in multimodal settings. Similarly, frameworks like RapGuard (Jiang et al., 2024b) dynamically generate scenario-specific safety prompts to reduce risks in each interaction.

Recent work on self-defense mechanisms offers promising strategies to protect GUI agents from manipulation. Phute et al. (2023) demonstrated that agents can effectively filter their own responses to block harmful content generation while not sacrificing functionality. Moving on, Xie et al. (2023) developed a complementary "system-mode self-reminder" technique, which wraps user queries in prompts that reinforce safe behavior. This method reduced jailbreak attack success rates from 67% to 19%. For agents handling sensitive tasks, Greenblatt et al. (2023) proposed robust safety protocols like "trusted editing" and "untrusted monitoring" that remain effective even when the agent actively tries to bypass them. At the model level, Liu et al. (2024) introduced Selective Knowledge Unlearning (SKU) to remove harmful knowledge from the underlying models powering GUI agents while preserving performance on legitimate tasks.

In practice, content safety depends on the agent's ability to reject unsafe requests, avoid exposing sensitive information, and handle ambiguous inputs responsibly. To catch potential harms before deployment, frameworks like AHA! (Anticipating Harms of AI)(Buçinca et al., 2023) support developers in identifying how different AI behaviors might negatively impact various stakeholders. AHA! creates example scenarios that show different ways agent systems can go wrong, based on responses from both crowd workers and language models. Interactive benchmarks such as ST-WebAgentBench highlight how easily agent alignment can fail when faced with real-world websites (Levy et al., 2024). Overall, adding stronger content filtering throughout the pipeline, along with consistent logging of agent actions, can help limit the impact when alignment breaks down.

### 4.3 Alignment with Human Values

Aligning GUI agents with human values means weighing individual user goals, like efficiency and personalization, against broader concerns such as fairness and inclusivity. One approach is to define these principles up front, as in Hua et al. (2024), where agent constitutions are used to embed safety guidelines during the planning process. Other frameworks, such as ResponsibleTA (Zhang et al., 2023), structure collaboration among multiple agent components to verify each step's feasibility and security. FREYR introduces a modular approach to tool integration in LLMs, improving adaptability to user needs without requiring extensive model fine-tuning (Gallotta et al., 2025).

Moral reasoning and cultural sensitivity are increasingly relevant, particularly when agents operate in diverse contexts or handle sensitive content (Qiu et al., 2024; Piatti et al., 2024). To achieve deeper alignment, agents may need to model nuanced social norms or policy constraints. Visual-Critic demonstrates how LMMs can assess visual content quality from a human perspective, a critical capability for ensuring user-aligned perception in GUI interactions (Huang et al., 2024b).

Commercial GUI agent implementations provide further insights into practical alignment strategies. OpenAI's CUA, for example, implements "watch mode" on sensitive websites (e.g., email) to ensure that critical operations are supervised by users, and it declines higher-risk tasks, such as banking transactions or sensitive decision-making, thereby enforcing clear capability boundaries as a safety mechanism (OpenAI, 2025). Similarly, Anthropic restricts its computer use beta feature from creating accounts or generating content on social platforms to prevent human impersonation (Anthropic, 2025).


Another critical challenge is user intent understanding. As noted by Kim et al. (2024b), GUI agents still struggle to accurately infer user goals across diverse applications, achieving only poor accuracy on unseen websites. Designing models that generalize effectively across varying tasks is crucial, particularly for handling contextual variations in user interactions and predicting user behavior in complex interfaces (Stefanidi et al., 2022; Gao et al., 2023). Recent research on Role-Playing Language Agents (RPLAs) highlights how LLMs can simulate personas and dynamically adapt to user preferences, offering a pathway to more personalized and context-aware GUI interactions (Chen et al., 2024b).

### 4.4 Future Directions

To enhance GUI agent reliability and safety while maintaining practical implementations, research should focus on these promising directions:

Real-Time Hallucination Prevention: Building on work by Chen et al. (2024d) and Zhong et al. (2024), future systems need lightweight verification mechanisms that catch inconsistencies before they cause errors. Browser extensions could cross-verify agent actions against actual webpage structures, flagging discrepancies immediately. Interactive correction interfaces would allow users to adjust an agent's visual attention during errors, creating valuable feedback loops to improve perception models. Additionally, environmental awareness systems could detect real-world distractions that might compromise reliability, addressing concerns raised by Ma et al. (2024).

Adaptive Safety Architecture: Rather than applying uniform safety measures, agents should dynamically adjust protection levels based on context. When financial or medical interfaces are detected, content filters could automatically tighten, similar to the "watch mode" implemented in commercial systems (OpenAI, 2025). Modular safety components, like specialized verifiers for payment dialogs, could be plugged in as needed, extending the "circuit breaker" concept introduced by Zou et al. (2024). For critical operations, requiring physical confirmation (e.g., a device authentication) could provide an additional security layer inspired by multi-agent verification (Yu et al., 2024).

Learning from Failures: Perhaps most promising is the systematic improvement of agents through failure analysis. Community-driven reporting of rare errors could create diverse testing datasets beyond what developers anticipate. Automated post-failure analysis reports would help identify perception or reasoning gaps, extending approaches like those in Gunjal et al. (2023). By prioritizing fixes based on real-world impact rather than theoretical concerns, development resources could target the most critical reliability issues first.

## 5 Explainability and Transparency

Explainability and transparency foster trust in GUI agents by helping users understand how the system perceives, reasons, and acts. This section discusses mechanisms for providing explanations, transparent decision-making, and user-centric presentation.

### 5.1 Techniques for Explaining Agent Behavior

Many methods focus on decomposing the agent's decision pipeline to surface intermediate steps. Explainable Behavior Cloning (EBC-LLMAgent) (Guan et al., 2024) captures demonstrations, generates executable code, and maps the code to UI elements. By documenting these transformations, the agent can clarify how it arrived at a particular action. Similarly, hierarchical designs that separate high-level planning from low-level execution offer more interpretable structures (Liu et al., 2025a; Zhu et al., 2025; Agashe et al., 2024).

Other efforts highlight introspection through multi-agent or chain-of-thought strategies (Nguyen et al., 2024c; Wang and Liu, 2024). Here, LLM-based agents iteratively reflect on previous reasoning steps, generating self-explanations or corrections. These reflective traces not only boost performance but also produce human-readable rationales. However, ensuring that explanations remain truthful rather than post-hoc justifications is still an open research challenge.

### 5.2 Transparency in Decision-Making

Transparency is especially crucial for high-stakes domains like finance or healthcare, where agents may access sensitive data or perform costly actions. Systems such as XMODE (Nooralahzadeh et al., 2024) rely on multimodal decomposition, combining textual and visual analytics to highlight evidence supporting each decision. Providing users with comprehensible summaries, such as color-coded or textual rationales, can help users trace the logic of agents (Houssel et al., 2024; Arnold and Tilton, 2024).


Equally vital is the agent's ability to justify or revise actions when confronted with unexpected outcomes. World models can enhance transparency by simulating multiple paths and explaining why certain actions seem preferable (Chae et al., 2024; Gu et al., 2024a). While this can help build user trust, it also comes with added computational cost; therefore, designs need to strike a balance to keep interactions responsive.

### 5.3 User-Centric Explanations

User-centered design focuses on tailoring explanations based on a person's context, preferences, and familiarity with a given domain (Xu et al., 2024b). For example, a health data entry system designed for older adults might adjust how it highlights input errors or suggests alternatives (Cuadra et al., 2024). In contrast, enterprise software might generate justifications that align with domain-specific workflows and terminology (Srinivas et al., 2024).

Beyond presentation, recent research explores how GUI agents can generate inherently interpretable outputs grounded in user-understandable concepts. For example, in vision-based tasks, synthesizing explanations with human-verifiable visual features has been shown to improve model reasoning and transparency, which could potentially enable GUI agents to justify actions in complex visual environments (Shi et al., 2025). On the language side, interpreting LLM representations using mutual information and sparse activations allows for controllable, semantically meaningful explanations, offering a promising direction for more steerable and trustworthy GUI behaviors (Wu et al., 2025b). These techniques bridge model internals with users, helping GUI agents adapt explanation styles while maintaining transparency and alignment.

### 5.4 Future Directions

Enhancing explainability and transparency in GUI agents requires practical solutions that balance technical depth with user accessibility. Two promising directions emerge:

Interactive Explanation Tools: Real-time visualization of agent reasoning could transform how users understand automated processes. Browser extensions could display decision chains (e.g., "identify search box $\rightarrow$ enter query $\rightarrow$ select result") using interactive flowcharts that visualize the agent's current focus, building on techniques from EBC-LLMAgent (Guan et al., 2024). On mobile devices, lightweight on-device models could highlight interface elements being analyzed without cloud processing delays, extending approaches from hierarchical agent designs (Liu et al., 2025a; Zhu et al., 2025). When operations fail, automated error playback could compare intended versus actual results, incorporating reflective techniques from XA-gent (Nguyen et al., 2024c) to make troubleshooting intuitive.

Context-Adaptive Explanations: Different users require different types of transparency. Future systems should provide role-based explanations, offering technical details for developers while generating simplified summaries for general users, extending the user-centric approaches seen in (Cuadra et al., 2024). Cultural context filters could automatically adjust explanation styles and privacy considerations based on regional norms, addressing localization challenges. Accessibility-focused explanation channels (such as voice explanations for visually impaired users) would ensure transparency benefits reach diverse populations, aligning with inclusive design principles (Xu et al., 2024b).

## 6 Ethical Implications

Developing GUI agents responsibly entails going beyond technical design to incorporate ethical principles, cultural sensitivity, and policy considerations. This section highlights core guidelines, discusses the need for cultural and social awareness, and addresses regulatory and policy implications.

### 6.1 Cultural and Social Awareness

Agents that serve diverse user groups need to account for cultural context and social norms (Qiu et al., 2024). For example, platforms in e-commerce or online discussions often contain content that carries culturally specific meaning, which requires context-aware handling. Benchmarks like CASA measure assess how well agents navigate these cross-cultural settings without overstepping boundaries (Qiu et al., 2024). Similarly, frameworks that embed moral reasoning (Piatti et al., 2024) encourage cooperative behaviors aligned with universalized ethical principles. Recent work argues that cultural NLP often lacks a unified theoretical foundation, emphasizing the need for localization-focused approaches rather than relying on static cultural templates (Zhou et al., 2025).


<!-- Media -->

<table><tr><td>Benchmark</td><td>Focus Area</td><td>Key Metrics</td></tr><tr><td colspan="3">Security & Privacy Evaluation</td></tr><tr><td>InjecAgent (Zhan et al., 2024)</td><td>Tool-integrated agent vulnerability</td><td>Attack success rate across 17 user tools</td></tr><tr><td>BrowserART (Kumar et al., 2024)</td><td>Browser agent jailbreaking</td><td>Harmful behavior attempt rate</td></tr><tr><td>AdvWeb (Xu et al., 2024a)</td><td>Black-box adversarial web attacks</td><td>Stealth effectiveness, Success rate</td></tr><tr><td>EIA (Liao et al., 2024)</td><td>Web agent privacy risks</td><td>PII extraction rate, Attack detection</td></tr><tr><td>PUPA (Siyan et al., 2024)</td><td>Privacy-preserving evaluation</td><td>PII exposure, Response quality</td></tr><tr><td>ARE (Wu et al., 2025a)</td><td>Adversarial robustness</td><td>Flow of adversarial information</td></tr><tr><td colspan="3">Harmfulness & Reliability Assessment</td></tr><tr><td>Agent-SafetyBench (Zhang et al., 2024c)</td><td>Comprehensive agent safety</td><td>Safety scores across 8 risk categories</td></tr><tr><td>AgentHarm (Andriushchenko et al., 2024)</td><td>Harmfulness assessment</td><td>Refusal rate, Task completion</td></tr><tr><td>MobileSafetyBench (Lee et al., 2024)</td><td>Mobile device control safety</td><td>Risk management, Injection resistance</td></tr><tr><td>ST-WebAgentBench (Levy et al., 2024)</td><td>Web safety and trustworthiness</td><td>Completion Under Policy, Risk Ratio</td></tr><tr><td>GTArena (Zhao et al., 2024)</td><td>Automated GUI testing</td><td>Test intention, Defect detection</td></tr><tr><td>MM-SafetyBench (Liu et al., 2023b)</td><td>Image-based manipulations</td><td>Visual attack resilience</td></tr><tr><td colspan="3">Human & Cultural Alignment</td></tr><tr><td>MSSBench (Zhou et al., 2024)</td><td>Multimodal situational safety</td><td>Safety reasoning, Visual understanding</td></tr><tr><td>CASA (Qiu et al., 2024)</td><td>Cultural and social awareness</td><td>Awareness coverage, Violation rate</td></tr></table>

Table 2: Taxonomy of GUI Agent Evaluation Frameworks. The benchmarks are categorized into three dimensions: (1) security and privacy evaluation, focusing on vulnerability assessment and attack resistance; (2) harmfulness and reliability assessment, measuring agent compliance with safety protocols and failure modes; and (3) human and cultural alignment, evaluating agents' ability to handle visual manipulations and conform to social norms.

<!-- Media -->

At the same time, agents also need to address accessibility, meeting the needs of older adults or individuals with sensory impairments (Cuadra et al., 2024). Designing flexible interaction paths, whether through speech, visual cues, or textual descriptions, will allow broader inclusivity. As technology advances, bridging cultural gaps and ensuring accessibility will likely require more elaborate training data and dedicated modules.

### 6.2 Policy Implications

Because GUI agents can execute complex actions with real-world consequences, policy and regulatory considerations are paramount (Gan et al., 2024; Chen et al., 2024a). In regulated sectors such as healthcare or finance, compliance with data protection requirements becomes mandatory. Meanwhile, governments and institutions face difficulties in overseeing technologies that are rapidly evolving and often proprietary. Decentralized governance frameworks, such as those leveraging blockchain, have been proposed to enhance transparency, accountability, and decision rights in foundation-model-based AI systems (Liu et al., 2023d).

Several initiatives encourage open-sourcing benchmarks and best practices (Levy et al., 2024), fostering community-driven standards for agent safety. The collaboration between industry, academia, and policymakers could also help clarify rules around data use and accountability. In the long term, building in responsible practices, through clear guidelines, strong evaluations, and cross-sector oversight, can better align GUI agents with societal values while supporting innovation.

### 6.3 Guidelines and Principles

Some efforts have focused on formalizing design principles through pattern-based architectures that ensure security, accountability, and fairness across the agent's lifecycle (Lu et al., 2023; Wu et al., 2024c). Modular systems make it easier to trace how different components handle data and interact, improving transparency and alignment with user inputs (Zhang et al., 2023; Hua et al., 2024). On the security side, newer authentication schemes aim to tighten control over delegation, making it harder for agents to take unauthorized actions while keeping the chain of responsibility clear (South et al., 2025).

In real-world settings, developers must consider both the power and risks of autonomy. When agents handle critical tasks, such as financial transactions or medical record management, clear guidelines for fallback procedures and user oversight should be essential (Wright, 2024). Recent studies also highlight ethical concerns beyond security, such as how interactions with agents may inadvertently shape user beliefs, with evidence showing that LLM-powered conversational agents can significantly amplify false memories in sensitive contexts like witness interviews (Chan et al., 2024).


## 7 Evaluation Frameworks and Benchmarks

Evaluating GUI agents requires solid frameworks to assess reliability and trust. This section covers current metrics, practical evaluation methods, and trustworthiness-specific benchmarks used to test performance and behavior. Table 2 summarizes key evaluation frameworks across different dimensions.

### 7.1 Metrics for Assessing Trustworthiness

Evaluation often begins with task completion: whether the agent navigates, inputs data, or detects anomalies accurately (Koh et al., 2024a; Chen et al., 2024c). However, success rate alone cannot capture trustworthiness. ST-WebAgentBench, for example, evaluates how well agents follow explicit policy rules, flagging any violations as signs of unsafe behavior (Levy et al., 2024). To detect problems earlier, intermediate metrics like URL or form field matching are also used to pinpoint where agents make mistakes (Zhou et al., 2023; Shi et al., 2017; Yao et al., 2022).

Researchers also propose metrics for robustness under adversarial conditions (Wu et al., 2025a), cultural or social awareness (Qiu et al., 2024), and situational safety (Zhou et al., 2024; Liu et al., 2023b). These approaches emphasize that a reliable GUI agent must not only achieve the user's intended outcome but also demonstrate safe and consistent behavior throughout the process.

### 7.2 Comprehensive Evaluation Techniques

Comprehensive frameworks often adopt a modular approach. ChEF (Comprehensive Evaluation Framework) systematically tests scenario variation, instruction diversity, inference strategies, and flexible metrics (Shi et al., 2023b). GTArena partitions automated GUI testing into intent generation, test execution, and defect detection for mobile apps (Zhao et al., 2024). By capturing multiple facets, including correctness, error handling, and safety, such evaluations reveal more profound insights into agent behavior.

Distinctions between closed-world and open-world tests matter for ecological validity. Closed-world environments, like curated sets of web pages, enable controlled experimentation but lack real-world unpredictability. Open-world evaluations allow dynamic changes and unknown interfaces (Chen et al., 2024c; He et al., 2024), forcing agents to adapt. Balancing reproducibility and realism remains an ongoing challenge.

### 7.3 Case Studies and Benchmarks

Numerous case studies develop domain-specific or specialized benchmarks. Mind2Web measures task completion on live websites, revealing difficulties in grounding instructions (Zheng et al., 2024). MSSBench focuses on "multimodal situational safety," where half of the query-image pairs require context-sensitive reasoning (Zhou et al., 2024). Similarly, VETL (Wang et al., 2024b) and WebCanvas (Koh et al., 2024a) test web GUI interactions and bug detection.

These benchmarks illustrate that trustworthy evaluation is inherently multifaceted. Future work could unify the disparate tasks, data sources, and metrics into more holistic frameworks, enabling meaningful comparisons of agents' safety, robustness, and usability. Such efforts will be critical for driving standardization and progress in this rapidly evolving domain.

## 8 Conclusion

This survey has examined trustworthiness in GUI agents across five critical dimensions: security vulnerabilities, reliability, explainability, ethical alignment, and evaluation methodologies. Our analysis reveals significant challenges at the intersection of these dimensions, where multimodal interactions create novel attack surfaces and failure modes that traditional approaches cannot adequately address. While research has primarily focused on functional performance, the integrated nature of GUI agents demands holistic approaches to trustworthiness that span their entire operational pipeline.

Looking forward, advancing trustworthy GUI agents will require: (1) robust multimodal defense mechanisms that protect against adversarial manipulations, (2) adaptive safety frameworks that balance autonomy with protection, and (3) user-centered transparency systems that make agent reasoning accessible without compromising security. Evaluation benchmarks should assess both capability and safety. With the right safeguards and cross-field collaboration, GUI agents can be made effective, secure, and aligned with human values.


## References

Saaket Agashe, Jiuzhou Han, Shuyu Gan, Jiachen Yang, Ang Li, and Xin Eric Wang. 2024. Agent s: An open agentic framework that uses computers like a human. arXiv preprint arXiv:2410.08164.

Maksym Andriushchenko, Alexandra Souly, Mateusz Dziemian, Derek Duenas, Maxwell Lin, Justin Wang, Dan Hendrycks, Andy Zou, Zico Kolter, Matt Fredrikson, et al. 2024. Agentharm: A benchmark for measuring harmfulness of llm agents. arXiv preprint arXiv:2410.09024.

Anthropic. 2025. Agents and tools: Computer use. Accessed: March 16, 2025.

Taylor B. Arnold and Lauren Tilton. 2024. Explainable search and discovery of visual cultural heritage collections with multimodal large language models. Workshop on Computational Humanities Research.

Zechen Bai, Pichao Wang, Tianjun Xiao, Tong He, Zongbo Han, Zheng Zhang, and Mike Zheng Shou. 2024. Hallucination of Multimodal Large Language Models: A Survey. arXiv.org.

Zana Buçinca, Chau Minh Pham, Maurice Jakesch, Marco Tulio Ribeiro, Alexandra Olteanu, and Saleema Amershi. 2023. Aha!: Facilitating AI Impact Assessment by Generating Examples of Harms. arXiv.org.

Tri Cao, Chengyu Huang, Yuexin Li, Huilin Wang, Amy He, Nay Oo, and Bryan Hooi. 2024. Phishagent: A robust multimodal agent for phishing webpage detection. arXiv preprint arXiv:2408.10738.

Hyungjoo Chae, Namyoung Kim, Kai Tzu-iunn Ong, Minju Gwak, Gwanwoo Song, Jihoon Kim, Sungh-wan Kim, Dongha Lee, and Jinyoung Yeo. 2024. Web agents with world models: Learning and leveraging environment dynamics in web navigation. arXiv preprint arXiv:2410.13232.

Samantha Chan, Pat Pataranutaporn, Aditya Suri, Wazeer Zulfikar, Pattie Maes, and Elizabeth F Loftus. 2024. Conversational ai powered by large language models amplifies false memories in witness interviews. arXiv preprint arXiv:2408.04681.

Chaoran Chen, Daodao Zhou, Yanfang Ye, Toby Li, and Yaxing Yao. 2024a. Clear: Towards contextual llm-empowered privacy policy analysis and risk generation for large language model applications. arXiv preprint arXiv:2410.13387.

Jiangjie Chen, Xintao Wang, Rui Xu, Siyu Yuan, Yikai Zhang, Wei Shi, Jian Xie, Shuang Li, Ruihan Yang, Tinghui Zhu, et al. 2024b. From persona to personalization: A survey on role-playing language agents. arXiv preprint arXiv:2404.18231.

Qi Chen, Dileepa Pitawela, Chongyang Zhao, Gengze Zhou, Hsiang-Ting Chen, and Qi Wu. 2024c. We-bvln: Vision-and-language navigation on websites. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, pages 1165-1173.

Xiang Chen, Chenxi Wang, Yida Xue, Ningyu Zhang, Xiaoyan Yang, Qiang Li, Yue Shen, Lei Liang, Jinjie Gu, and Huajun Chen. 2024d. Unified Hallucination Detection for Multimodal Large Language Models. Annual Meeting of the Association for Computational Linguistics.

Yurun Chen, Xueyu Hu, Keting Yin, Juncheng Li, and Shengyu Zhang. 2025. Aeia-mn: Evaluating the robustness of multimodal llm-powered mobile agents against active environmental injection attacks. arXiv preprint arXiv:2502.13053.

Andrea Cuadra, Justine Breuch, Samantha Estrada, David Ihim, Isabelle Hung, Derek Askaryar, Marwan Hassanien, K. Fessele, and J. Landay. 2024. Digital forms for all: A holistic multimodal large language model agent for health data entry. Proceedings of the ACM on Interactive Mobile Wearable and Ubiquitous Technologies.

Dazhen Deng, Chuhan Zhang, Hongxing Fan, Zhen-fei Yin, Lu Sheng, Yu Qiao, and Jing Shao. 2024. Adver-saflow: Visual red teaming for large language models with multi-level adversarial flow. IEEE Transactions on Visualization and Computer Graphics.

Xiang Deng, Yu Gu, Boyuan Zheng, Shijie Chen, Sam Stevens, Boshi Wang, Huan Sun, and Yu Su. 2023. Mind2web: Towards a generalist agent for the web. Advances in Neural Information Processing Systems, 36:28091-28114.

Xiaohan Fu, Shuheng Li, Zihan Wang, Yihao Liu, Rajesh K. Gupta, Taylor Berg-Kirkpatrick, and Ear-lence Fernandes. 2024. Imprompter: Tricking llm agents into improper tool use. arXiv preprint arXiv:2410.14923.

Roberto Gallotta, Antonios Liapis, and Georgios N Yannakakis. 2025. Freyr: A framework for recognizing and executing your requests. arXiv preprint arXiv:2501.12423.

Yuyou Gan, Yong Yang, Zhen Ma, Ping He, Rui Zeng, Yiming Wang, Qingming Li, Chunyi Zhou, Songze Li, Ting Wang, Yunjun Gao, Yingcai Wu, and Shoul-ing Ji. 2024. Navigating the risks: A survey of security, privacy, and ethics threats in llm-based agents. arXiv preprint arXiv:2411.09523.

Difei Gao, Lei Ji, Zechen Bai, Mingyu Ouyang, Peiran Li, Dongxing Mao, Qinchen Wu, Weichen Zhang, Peiyi Wang, Xiangwu Guo, et al. 2023. Assistgui: Task-oriented desktop graphical user interface automation. arXiv preprint arXiv:2312.13108.

Jiahui Gao, Renjie Pi, Tianyang Han, Han Wu, Chenyang Lyu, Huayang Li, Lanqing Hong, Ling-peng Kong, Xin Jiang, and Zhenguo Li. 2024. Coca: Regaining safety-awareness of multimodal large language models with constitutional calibration. arXiv preprint arXiv:2409.11365.

Ryan Greenblatt, Buck Shlegeris, Kshitij Sachan, and Fabien Roger. 2023. Ai Control: Improving Safety


Despite Intentional Subversion. International Conference on Machine Learning.

Yu Gu, Boyuan Zheng, Boyu Gou, Kai Zhang, Cheng Chang, Sanjari Srivastava, Yanan Xie, Peng Qi, Huan Sun, and Yu Su. 2024a. Is your llm secretly a world model of the internet? model-based planning for web agents. arXiv preprint arXiv:2411.06559.

Yu Gu, Boyuan Zheng, Boyu Gou, Kai Zhang, Cheng Chang, Sanjari Srivastava, Yanan Xie, Peng Qi, Huan Sun, and Yu Su. 2024b. Is your llm secretly a world model of the internet? model-based planning for web agents. arXiv preprint arXiv:2411.06559.

Yanchu Guan, Dong Wang, Yan Wang, Haiqing Wang, Renen Sun, Chenyi Zhuang, Jinjie Gu, and Zhixuan Chu. 2024. Explainable behavior cloning: Teaching large language model agents through learning by demonstration. arXiv preprint arXiv:2410.22916.

Anisha Gunjal, Jihan Yin, and Erhan Bas. 2023. Detecting and Preventing Hallucinations in Large Vision Language Models. AAAI Conference on Artificial Intelligence.

Hongliang He, Wenlin Yao, Kaixin Ma, Wenhao Yu, Yong Dai, Hongming Zhang, Zhenzhong Lan, and Dong Yu. 2024. Webvoyager: Building an end-to-end web agent with large multimodal models. arXiv preprint arXiv:2401.13919.

Paul RB Houssel, Priyanka Singh, Siamak Layeghy, and Marius Portmann. 2024. Towards explainable network intrusion detection using large language models. arXiv preprint arXiv:2408.04342.

Xueyu Hu, Tao Xiong, Biao Yi, Zishu Wei, Ruixuan Xiao, Yurun Chen, Jiasheng Ye, Meiling Tao, Xi-angxin Zhou, Ziyu Zhao, et al. Os agents: A survey on mllm-based agents for computer, phone and browser use.

Wenyue Hua, Xianjun Yang, Zelong Li, Cheng Wei, and Yongfeng Zhang. 2024. Trustagent: Towards safe and trustworthy llm-based agents. Conference on Empirical Methods in Natural Language Processing.

Qidong Huang, Xiaoyi Dong, Pan Zhang, Bin Wang, Conghui He, Jiaqi Wang, Dahua Lin, Weiming Zhang, and Nenghai Yu. 2024a. Opera: Alleviating Hallucination in Multi-Modal Large Language Models via Over-Trust Penalty and Retrospection-Allocation. In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 13418-13427. IEEE.

Zhipeng Huang, Zhizheng Zhang, Yiting Lu, Zheng-Jun Zha, Zhibo Chen, and Baining Guo. 2024b. Visu-alcritic: Making lmms perceive visual quality like humans. arXiv preprint arXiv:2403.12806.

Pete Janowczyk, Linda Laurier, Ave Giulietta, Arlo Octavia, and Meade Cleti. 2024. Seeing is deceiving: Exploitation of visual pathways in multi-modal language models. arXiv preprint arXiv:2411.05056.

Chaoya Jiang, Haiyang Xu, Mengfan Dong, Jiaxing Chen, Wei Ye, Ming Yan, Qinghao Ye, Ji Zhang, Fei Huang, and Shikun Zhang. 2024a. Hallucination Augmented Contrastive Learning for Multimodal Large Language Model. In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 27026-27036. IEEE.

Yilei Jiang, Yingshui Tan, and Xiangyu Yue. 2024b. Rapguard: Safeguarding multimodal large language models via rationale-aware defensive prompting. arXiv preprint arXiv:2412.18826.

Hanna Kim, Minkyoo Song, Seung Ho Na, Seungwon Shin, and Kimin Lee. 2024a. When llms go online: The emerging threat of web-enabled llms. arXiv preprint arXiv:2410.14569.

Jaekyeom Kim, Dong-Ki Kim, Lajanugen Logeswaran, Sungryull Sohn, and Honglak Lee. 2024b. Auto-intent: Automated intent discovery and self-exploration for large language model web agents. arXiv preprint arXiv:2410.22552.

Jing Yu Koh, Robert Lo, Lawrence Jang, Vikram Duvvur, Ming Chong Lim, Po-Yu Huang, Graham Neubig, Shuyan Zhou, Ruslan Salakhutdinov, and Daniel Fried. 2024a. Visualwebarena: Evaluating multimodal agents on realistic visual web tasks. arXiv preprint arXiv:2401.13649.

Jing Yu Koh, Stephen McAleer, Daniel Fried, and Rus-lan Salakhutdinov. 2024b. Tree search for language model agents. arXiv preprint arXiv:2407.01476.

Priyanshu Kumar, Elaine Lau, Saranya Vijayaku-mar, Tu Trinh, Scale Red Team, Elaine Chang, Vaughn Robinson, Sean Hendryx, Shuyan Zhou, Matt Fredrikson, Summer Yue, and Zifan Wang. 2024. Refusal-trained llms are easily jailbroken as browser agents. arXiv preprint arXiv:2410.13886.

Juyong Lee, Dongyoon Hahm, June Suk Choi, W Bradley Knox, and Kimin Lee. 2024. Mo-bilesafetybench: Evaluating safety of autonomous agents in mobile device control. arXiv preprint arXiv:2410.17520.

Seongyun Lee, Sue Hyun Park, Yongrae Jo, and Min-joon Seo. 2023. Volcano: mitigating multimodal hallucination through self-feedback guided revision. arXiv preprint arXiv:2311.07362.

Ido Levy, Ben Wiesel, Sami Marreed, Alon Oved, Avi Yaeli, and Segev Shlomov. 2024. St-webagentbench: A benchmark for evaluating safety and trustworthiness in web agents. arXiv preprint arXiv:2410.06703.

Zeyi Liao, Lingbo Mo, Chejian Xu, Mintong Kang, Jiawei Zhang, Chaowei Xiao, Yuan Tian, Bo Li, and Huan Sun. 2024. Eia: Environmental injection attack on generalist web agents for privacy leakage. arXiv preprint arXiv:2409.11295.


Fuxiao Liu, Kevin Lin, Linjie Li, Jianfeng Wang, Yaser Yacoob, and Lijuan Wang. 2023a. Mitigating Hallucination in Large Multi-Modal Models via Robust Instruction Tuning. arXiv.org.

Haowei Liu, Xi Zhang, Haiyang Xu, Yuyang Wanyan, Junyang Wang, Ming Yan, Ji Zhang, Chunfeng Yuan, Changsheng Xu, Weiming Hu, et al. 2025a. Pc-agent: A hierarchical multi-agent collaboration framework for complex task automation on pc. arXiv preprint arXiv:2502.14282.

William Liu, Liang Liu, Yaxuan Guo, Han Xiao, Weifeng Lin, Yuxiang Chai, Shuai Ren, Xiaoyu Liang, Linghao Li, Wenhao Wang, et al. 2025b. Llm-powered gui agents in phone automation: Surveying progress and prospects.

Xin Liu, Yichen Zhu, Jindong Gu, Yunshi Lan, Chao Yang, and Yu Qiao. 2023b. Mm-safetybench: A benchmark for safety evaluation of multimodal large language models. European Conference on Computer Vision.

Yang Liu, Yuanshun Yao, Jean-Francois Ton, Xiaoying Zhang, Ruocheng Guo, Hao Cheng, Yegor Klochkov, Muhammad Faaiz Taufiq, and Hang Li. 2023c. Trustworthy llms: a survey and guideline for evaluating large language models' alignment. arXiv preprint arXiv:2308.05374.

Yue Liu, Qinghua Lu, Liming Zhu, and Hye-Young Paik. 2023d. Decentralised governance-driven architecture for designing foundation model based systems: Exploring the role of blockchain in responsible ai. arXiv preprint arXiv:2308.05962.

Zheyuan Liu, Guangyao Dou, Zhaoxuan Tan, Yijun Tian, and Meng Jiang. 2024. Towards Safer Large Language Models through Machine Unlearning. Annual Meeting of the Association for Computational Linguistics.

Qinghua Lu, Liming Zhu, Xiwei Xu, Zhenchang Xing, Stefan Harrer, and Jon Whittle. 2023. Towards responsible generative ai: A reference architecture for designing foundation model based agents. In 2024 IEEE 21st International Conference on Software Architecture Companion (ICSA-C). IEEE.

Xinbei Ma, Yiting Wang, Yao Yao, Tongxin Yuan, Aston Zhang, Zhuosheng Zhang, and Hai Zhao. 2024. Caution for the environment: Multimodal agents are susceptible to environmental distractions. arXiv preprint arXiv:2408.02544.

Ivoline Ngong, Swanand Kadhe, Hao Wang, Keerthiram Murugesan, Justin D Weisz, Amit Dhurandhar, and Karthikeyan Natesan Ramamurthy. 2025. Protecting users from themselves: Safeguarding contextual privacy in interactions with conversational agents. arXiv preprint arXiv:2502.18509.

Dang Nguyen, Jian Chen, Yu Wang, Gang Wu, Namy-ong Park, Zhengmian Hu, Hanjia Lyu, Junda Wu, Ryan Aponte, Yu Xia, Xintong Li, Jing Shi, Hongjie Chen, Viet Dac Lai, Zhouhang Xie, Sungchul Kim, Ruiyi Zhang, Tong Yu, Mehrab Tanjim, Nesreen K. Ahmed, Puneet Mathur, Seunghyun Yoon, Lina Yao, Branislav Kveton, Thien Huu Nguyen, Trung Bui, Tianyi Zhou, Ryan A. Rossi, and Franck Dernon-court. 2024a. GUI Agents: A Survey. arXiv preprint. ArXiv:2412.13501 [cs].

Dang Nguyen, Viet Dac Lai, Seunghyun Yoon, Ryan A Rossi, Handong Zhao, Ruiyi Zhang, Puneet Mathur, Nedim Lipka, Yu Wang, Trung Bui, et al. 2024b. Dynasaur: Large language agents beyond predefined actions. arXiv preprint arXiv:2411.01747.

Van Bach Nguyen, Jörg Schlötterer, and Christin Seifert. 2024c. Xagent: A conversational xai agent harnessing the power of large language models. xAI.

Songqin Nong, Jiali Zhu, Rui Wu, Jiongchao Jin, Shuo Shan, Xiutian Huang, and Wenhao Xu. 2024. Mo-bileflow: A multimodal llm for mobile gui agent. arXiv preprint arXiv:2407.04346.

Farhad Nooralahzadeh, Yi Zhang, Jonathan Furst, and Kurt Stockinger. 2024. Explainable multi-modal data exploration in natural language via llm agent. arXiv preprint arXiv:2412.18428.

OpenAI. 2025. Computer-using agent. Accessed: March 16, 2025.

Mansi Phute, Alec Helbling, Matthew Hull, Sheng Yun Peng, Sebastian Szyller, Cory Cornelius, and Duen Horng Chau. 2023. Llm Self Defense: By Self Examination, LLMs Know They Are Being Tricked. Tiny Papers @ ICLR.

Giorgio Piatti, Zhijing Jin, Max Kleiman-Weiner, Bernhard Schölkopf, Mrinmaya Sachan, and Rada Mi-halcea. 2024. Cooperate or collapse: Emergence of sustainable cooperation in a society of llm agents. Advances in Neural Information Processing Systems, 37:111715-111759.

Haoyi Qiu, A. R. Fabbri, Divyansh Agarwal, Kung-Hsiang Huang, Sarah Tan, Nanyun Peng, and Chien-Sheng Wu. 2024. Evaluating cultural and social awareness of llm web agents. arXiv preprint arXiv:2410.23252.

Reshabh K Sharma, Vinayak Gupta, and Dan Grossman. 2024. Defending language models against image-based prompt attacks via user-provided specifications. 2024 IEEE Security and Privacy Workshops (SPW).

Huawen Shen, Chang Liu, Gengluo Li, Xinlong Wang, Yu Zhou, Can Ma, and Xiangyang Ji. 2024. Falcon-ui: Understanding gui before following user instructions. arXiv preprint arXiv:2412.09362.

Tianlin Shi, Andrej Karpathy, Linxi Fan, Jonathan Hernandez, and Percy Liang. 2017. World of bits: An open-domain platform for web-based agents. In Proceedings of the 34th International Conference on Machine Learning, volume 70 of Proceedings of Machine Learning Research, pages 3135-3144. PMLR.


Yucheng Shi, Mengnan Du, Xuansheng Wu, Zihan Guan, Jin Sun, and Ninghao Liu. 2023a. Black-box backdoor defense via zero-shot image purification. Advances in Neural Information Processing Systems, 36:57336-57366.

Yucheng Shi, Quanzheng Li, Jin Sun, Xiang Li, and Ninghao Liu. 2025. Enhancing cognition and explainability of multimodal foundation models with self-synthesized data. In The Thirteenth International Conference on Learning Representations.

Zhelun Shi, Zhipin Wang, Hongxing Fan, Zhen-fei Yin, Lu Sheng, Yu Qiao, and Jing Shao. 2023b. Chef: A comprehensive evaluation framework for standardized assessment of multimodal large language models. arXiv preprint arXiv:2311.02692.

Li Siyan, Vethavikashini Chithrra Raghuram, Omar Khattab, Julia Hirschberg, and Zhou Yu. 2024. Papillon: Privacy preservation from internet-based and local language model ensembles. arXiv preprint arXiv:2410.17127.

Tobin South, Samuele Marro, Thomas Hardjono, Robert Mahari, Cedric Deslandes Whitney, Dazza Greenwood, Alan Chan, and Alex Pentland. 2025. Authenticated delegation and authorized ai agents. arXiv preprint arXiv:2501.09674.

Sakhinana Sagar Srinivas, Geethan Sannidhi, and Venkataramana Runkana. 2024. Towards human-level understanding of complex process engineering schematics: A pedagogical, introspective multi-agent framework for open-domain question answering. arXiv preprint arXiv:2409.00082.

Zinovia Stefanidi, George Margetis, Stavroula Ntoa, and George Papagiannakis. 2022. Real-time adaptation of context-aware intelligent user interfaces, for enhanced situational awareness. IEEE Access, 10:23367-23393.

Shilong Wang, Guibin Zhang, Miao Yu, Guancheng Wan, Fanci Meng, Chongye Guo, Kun Wang, and Yang Wang. 2025. G-safeguard: A topology-guided security lens and treatment on llm-based multi-agent systems. arXiv preprint arXiv:2502.11127.

Shuai Wang, Weiwen Liu, Jingxuan Chen, Yuqi Zhou, Weinan Gan, Xingshan Zeng, Yuhan Che, Shuai Yu, Xinlong Hao, Kun Shao, et al. 2024a. Gui agents with foundation models: A comprehensive survey. arXiv preprint arXiv:2411.04890.

Siyi Wang, Sinan Wang, Yujia Fan, Xiaolei Li, and Yepang Liu. 2024b. Leveraging large vision-language model for better automatic web gui testing. IEEE International Conference on Software Maintenance and Evolution.

Xiaoqiang Wang and Bang Liu. 2024. Oscar: Operating system control via state-aware reasoning and re-planning. arXiv preprint arXiv:2410.18963.

Yuntao Wang, Yanghe Pan, Quan Zhao, Yi Deng, Zhou Su, Linkang Du, and Tom H Luan. 2024c. Large model agents: State-of-the-art, cooperation paradigms, security and privacy, and future trends. arXiv preprint arXiv:2409.14457.

Laura Weidinger, Jonathan Uesato, Maribeth Rauh, Conor Griffin, Po-Sen Huang, John Mellor, Amelia Glaese, Myra Cheng, Borja Balle, Atoosa Kasirzadeh, et al. 2022. Taxonomy of risks posed by language models. In Proceedings of the 2022 ACM conference on fairness, accountability, and transparency, pages 214-229.

Hao Wen, Yuanchun Li, Guohong Liu, Shanhui Zhao, Tao Yu, Toby Jia-Jun Li, Shiqi Jiang, Yunhao Liu, Yaqin Zhang, and Yunxin Liu. 2024. Autodroid: Llm-powered task automation in android. In Proceedings of the 30th Annual International Conference on Mobile Computing and Networking, pages 543-557.

Jesse Wright. 2024. Here's charlie! realising the semantic web vision of agents in the age of llms. International Workshop on the Semantic Web.

Chen Henry Wu, Rishi Rajesh Shah, Jing Yu Koh, Russ Salakhutdinov, Daniel Fried, and Aditi Raghunathan. 2025a. Dissecting adversarial robustness of multimodal Im agents. In The Thirteenth International Conference on Learning Representations.

Fangzhou Wu, Shutong Wu, Yulong Cao, and Chaowei Xiao. 2024a. Wipi: A new web threat for llm-driven web agents. arXiv preprint arXiv:2402.16965.

Xuansheng Wu, Jiayi Yuan, Wenlin Yao, Xiaoming Zhai, and Ninghao Liu. 2025b. Interpreting and steering llms with mutual information-based explanations on sparse autoencoders. arXiv preprint arXiv:2502.15576.

Xuansheng Wu, Haiyan Zhao, Yaochen Zhu, Yucheng Shi, Fan Yang, Tianming Liu, Xiaoming Zhai, Wenlin Yao, Jundong Li, Mengnan Du, et al. 2024b. Usable xai: 10 strategies towards exploiting explainability in the llm era. arXiv preprint arXiv:2403.08946.

Zhiyong Wu, Zhenyu Wu, Fangzhi Xu, Yian Wang, Qiushi Sun, Chengyou Jia, Kanzhi Cheng, Zichen Ding, Liheng Chen, Paul Pu Liang, et al. 2024c. Os-atlas: A foundation action model for generalist gui agents. arXiv preprint arXiv:2410.23218.

Zhen Xiang, Linzhi Zheng, Yanjie Li, Junyuan Hong, Qinbin Li, Han Xie, Jiawei Zhang, Zidi Xiong, Chulin Xie, Carl Yang, Dawn Song, and Bo Li. 2024. Guardagent: Safeguard llm agents by a guard agent via knowledge-enabled reasoning. arXiv preprint arXiv:2406.09187.

Junlin Xie, Zhihong Chen, Ruifei Zhang, Xiang Wan, and Guanbin Li. 2024. Large multimodal agents: A survey. arXiv preprint arXiv:2402.15116.


Yueqi Xie, Jingwei Yi, Jiawei Shao, Justin Curl, Lingjuan Lyu, Qifeng Chen, Xing Xie, and Fangzhao Wu. 2023. Defending ChatGPT against jailbreak attack via self-reminders. Nature Machine Intelligence, 5(12):1486-1496.

Chejian Xu, Mintong Kang, Jiawei Zhang, Zeyi Liao, Lingbo Mo, Mengqi Yuan, Huan Sun, and Bo Li. 2024a. Advweb: Controllable black-box attacks on vlm-powered web agents. arXiv preprint arXiv:2410.17401.

Yuanyuan Xu, Weiting Gao, Yining Wang, Xinyang Shan, and Yin-Shan Lin. 2024b. Enhancing user experience and trust in advanced llm-based conversational agents. Computing and Artificial Intelligence, 2(2).

Yulong Yang, Xinshan Yang, Shuaidong Li, Chenhao Lin, Zhengyu Zhao, Chao Shen, and Tianwei Zhang. 2024. Security matrix for multimodal agents on mobile devices: A systematic and proof of concept study. arXiv preprint arXiv:2407.09295.

Shunyu Yao, Howard Chen, John Yang, and Karthik Narasimhan. 2022. Webshop: Towards scalable real-world web interaction with grounded language agents. Advances in Neural Information Processing Systems, 35:20744-20757.

Chung-En (Johnny) Yu, Brian Jalaian, and Nathaniel D. Bastian. 2024. Mitigating Large Vision-Language Model Hallucination at Post-hoc via Multi-agent System. Proceedings of the AAAI Symposium Series, 4(1):110-113.

Qiusi Zhan, Zhixiang Liang, Zifan Ying, and Daniel Kang. 2024. Injecagent: Benchmarking indirect prompt injections in tool-integrated large language model agents. arXiv preprint arXiv:2403.02691.

Chaoyun Zhang, Shilin He, Jiaxu Qian, Bowen Li, Liqun Li, Si Qin, Yu Kang, Minghua Ma, Qingwei Lin, Saravan Rajmohan, et al. 2024a. Large language model-brained gui agents: A survey. arXiv preprint arXiv:2411.18279.

Xinyu Zhang, Huiyu Xu, Zhongjie Ba, Zhibo Wang, Yuan Hong, Jian Liu, Zhan Qin, and Kui Ren. 2024b. Privacyasst: Safeguarding user privacy in tool-using large language model agents. IEEE Transactions on Dependable and Secure Computing.

Zhexin Zhang, Shiyao Cui, Yida Lu, Jingzhuo Zhou, Junxiao Yang, Hongning Wang, and Minlie Huang. 2024c. Agent-safetybench: Evaluating the safety of llm agents. arXiv preprint arXiv:2412.14470.

Zhizheng Zhang, Xiaoyi Zhang, Wenxuan Xie, and Yan Lu. 2023. Responsible task automation: Empowering large language models as responsible task au-tomators. arXiv preprint arXiv:2306.01242.

Kangjia Zhao, Jiahui Song, Leigang Sha, HaoZhan Shen, Zhi Chen, Tiancheng Zhao, Xiubo Liang, and Jianwei Yin. 2024. Gui testing arena: A unified benchmark for advancing autonomous gui testing agent. arXiv preprint arXiv:2412.18426.

Boyuan Zheng, Boyu Gou, Jihyung Kil, Huan Sun, and Yu Su. 2024. Gpt-4v(ision) is a generalist web agent, if grounded. International Conference on Machine Learning.

Weihong Zhong, Xiaocheng Feng, Liang Zhao, Qiming Li, Lei Huang, Yuxuan Gu, Weitao Ma, Yuan Xu, and Bing Qin. 2024. Investigating and Mitigating the Multimodal Hallucination Snowballing in Large Vision-Language Models. Annual Meeting of the Association for Computational Linguistics.

KAI-QING Zhou, Chengzhi Liu, Xuandong Zhao, Anderson Compalas, Dawn Song, and Xin Eric Wang. 2024. Multimodal situational safety. arXiv preprint arXiv:2410.06172.

Naitian Zhou, David Bamman, and Isaac L Bleaman. 2025. Culture is not trivia: Sociocultural theory for cultural nlp. arXiv preprint arXiv:2502.12057.

Shuyan Zhou, Frank F Xu, Hao Zhu, Xuhui Zhou, Robert Lo, Abishek Sridhar, Xianyi Cheng, Tianyue Ou, Yonatan Bisk, Daniel Fried, et al. 2023. We-barena: A realistic web environment for building autonomous agents. arXiv preprint arXiv:2307.13854.

Zichen Zhu, Hao Tang, Yansi Li, Dingye Liu, Hongshen Xu, Kunyao Lan, Danyang Zhang, Yixuan Jiang, Hao Zhou, Chenrun Wang, Situo Zhang, Liangtai Sun, Yixiao Wang, Yuheng Sun, Lu Chen, and Kai Yu. 2025. Moba: Multifaceted memory-enhanced adaptive planning for efficient mobile task automation. Preprint, arXiv:2410.13757.

Andy Zou, Long Phan, Justin Wang, Derek Duenas, Maxwell Lin, Maksym Andriushchenko, Rowan Wang, Zico Kolter, Matt Fredrikson, and Dan Hendrycks. 2024. Improving Alignment and Robustness with Circuit Breakers. arXiv.org.