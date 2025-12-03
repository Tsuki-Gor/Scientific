# Long-Horizon Planning for Multi-Agent Robots in Partially Observable Environments

Siddharth Nayak ${}^{1 * }$ Adelmo Morrison Orozco ${}^{1 * }$ Marina Ten Have ${}^{1}$ Vittal Thirumalai ${}^{1}$ Jackson Zhang ${}^{1}$ Darren Chen ${}^{1}$ Aditya Kapoor ${}^{2}$ Eric Robinson ${}^{3}$ Karthik Gopalakrishnan ${}^{4}$ James Harrison ${}^{5} \; {\text{ on }}^{5}$ Brian Ichter ${}^{ \dagger  }{}^{5}$ Anuj Mahajan ${}^{ \ddagger  }{}^{6}$ Hamsa Balakrishnan ${}^{1}$

${}^{1}$ MIT ${}^{2}$ TCS ${}^{3}$ USAF-MIT AI Accelerator

${}^{4}$ Stanford ${}^{5}$ Google DeepMind ${}^{6}$ Apple

## Abstract

The ability of Language Models (LMs) to understand natural language makes them a powerful tool for parsing human instructions into task plans for autonomous robots. Unlike traditional planning methods that rely on domain-specific knowledge and handcrafted rules, LMs generalize from diverse data and adapt to various tasks with minimal tuning, acting as a compressed knowledge base. However, LMs in their standard form face challenges with long-horizon tasks, particularly in partially observable multi-agent settings. We propose an LM-based Long-Horizon Planner for Multi-Agent Robotics (LLaMAR), a cognitive architecture for planning that achieves state-of-the-art results in long-horizon tasks within partially observable environments. LLaMAR employs a plan-act-correct-verify framework, allowing self-correction from action execution feedback without relying on oracles or simulators. Additionally, we present MAP-THOR, a comprehensive test suite encompassing household tasks of varying complexity within the AI2-THOR environment. Experiments show that LLaMAR achieves a 30% higher success rate than other state-of-the-art LM-based multi-agent planners in MAP-THOR and Search & Rescue tasks. Code can be found at https://github.com/nsidn98/LLaMAR

## 1 Introduction

Creating embodied agents that assist humans in real-life scenarios is a significant challenge when humans communicate their intentions in natural language. Certain tasks like moving furniture [1, 2], search-and-rescue [3], environmental monitoring [4], etc., require coordination among multiple agents to solve the tasks efficiently as compared to single-agent scenarios. This challenge of understanding these natural language inputs and effectively coordinating these agents to solve the task is exacerbated in the multi-agent scenario. Recent works [5, 6, 7, 8, 9, 10, 11, 12, 13] have shown that Language Models (LMs) ${}^{1}$ can effectively use language instructions to develop plans for robots. However,most studies focus on single-agent long-horizon task planning. Naïve extensions of single-agent planning algorithms to multi-agent settings often fail due to environment non-stationarity, where the policies of other agents—modeled as a part of the environment—are continuously changing [14, 15]. Such failures lead to suboptimal performance as agents struggle to anticipate and adapt to the actions of others. We therefore formulate a centralized process in which decisions are made simultaneously for all agents based on their (partial) observations, similar to the centralized multi-agent system framework (CMAS) proposed in [16]. Leveraging the ability of pre-trained LMs to generalize across diverse tasks, we aim to use LMs for long-horizon embodied multi-agent task planning.

The key insight of our work is that integrating a plan-act-correct-verify framework with LMs enables a robust and adaptive approach to multi-agent task planning in dynamic, partially observable environments that allows agents to: (1) plan subtasks required to complete the task, (2) select high-level actions for each agent to complete the proposed subtasks, (3) identify and correct failures after high-level action execution, and (4) self-verify subtask completion based on high-level action execution. Unlike existing methods, our approach uses real-time execution feedback, observations, and agent histories to iteratively refine action planning and execution. This allows agents to adjust strategies based on reasoned insights on action execution, effectively addressing failures without relying on perfect environmental knowledge or oracle feedback. The correction and verification process in our cognitive architecture [17] is grounded in the environment's reality, which sets it apart from LM self-verification methods that lack such grounding [18]. This framework enhances agents' ability to complete complex, long-horizon tasks, yielding substantial improvement over current state-of-the-art methods.

---

* Equal Contribution. ${}^{ \dagger  }$ Now at Physical Intelligence. ${}^{ \ddagger  }$ Work done outside Apple. ${}^{1}$ We denote Large Language Models as LLMs, Vision Language Models as VLMs

---

Similar to our approach, recent works [19, 20, 21, 22, 23, 24, 16] utilize LMs for multi-agent planning, often adopting a hierarchical decision-making structure. The LMs are used for high-level planning to determine subtasks, sometimes in conjunction with planning domain definition language (PDDL) that together with the LM planner, functions as a feasibility solver. Specific actions are executed using low-level policies pre-trained through reinforcement learning, behavior cloning, or heuristic approaches. While these methods effectively use LMs as high-level planners, they assume perfect low-level primitive action policies and simulator or oracle-provided environmental information. By contrast, LLaMAR does not assume perfect knowledge of the environment, does not rely on oracle feedback, and does not assume perfect execution of low-level primitive policies. This approach moves us closer to enabling real-world robots that operate independently of privileged knowledge.

To avoid ambiguity, we use the following conventions. We refer to the objectives within the environments as "tasks" or "goals" and "subtasks" to describe the breakdown of tasks or goals. "High-level actions" are defined as skills the agent can perform, while "low-level actions" or "primitive actions" refer to existing policies—either learned or predefined using heuristics—that execute a sequence of actions to accomplish a high-level action. More details and examples can be found in Appendix A

The main contributions of this paper are:

- LLaMAR: An LM-based Long-Horizon Planner for Multi-Agent Robotics, designed for iterative planning of long-horizon, multi-objective tasks in partially observable environments, with the following key features:

- It operates without prior knowledge of the environment, allowing agents to explore and make decisions based on new observations.

- It evaluates outcomes through direct observation of images, rather than relying on oracles for feedback, enabling independent identification and correction of action failures.

- MAP-THOR (Multi-Agent Planning in THOR): a benchmark suite of tasks within the AI2- THOR simulator under partial observability to standardize methodologies and metrics for evaluating multi-agent planning effectiveness and robustness.

## 2 Related Work

Reinforcement Learning (RL) for Long-Horizon Planning: While RL algorithms have shown promise in many applications, they still struggle with long-horizon tasks. Hierarchical reinforcement learning (HRL) has been used to address these challenges in both single-agent [25, 26, 27, 28] and multi-agent settings [29, 30, 31]. However, these approaches are typically applied to single-task, stationary environments, such as games, where agents solve for one goal in a fixed environment. Consequently, these methods do not generalize well across multiple environments or tasks. Multi-task RL has been explored as a potential solution, requiring sophisticated task planning to handle diverse objectives [32, 33]. This often involves decomposing tasks into manageable subtasks, a process well-suited for hierarchical frameworks. However, subtasks are known apriori in multi-task RL formulations. Real-world long-horizon RL necessitates robust task planning, and LMs have emerged as a promising approach for this purpose.

LMs for Embodied Single-Agent Planning: Recent studies have demonstrated the effectiveness of LMs in generating and executing plans in embodied single-agent environments [34, 35, 36, 37, 38, 39 40] and creating plans in single-agent embodied robotic environments [41, 42, 43, 44, 45, 46, 47] 48 49 50, 51 52]. Works like SayCan [5] and Grounded Decoding [6] use a combination of value functions and LLM predictions for long-horizon tasks. ProgPrompt [8] and Zero-Shot Language Planner [13] generate static plans executed in the environment, which may fail in partially observable and dynamic settings. To mitigate this, LLM-planner [53] updates plans based on new observations, similar to our approach.

LMs for Multi-Agent Planning: Xu et al. [54] use LLMs in multi-agent games, while CoNavGPT [23] creates global plans for two robots in an embodied environment. RoCo [24] and CoELA [22] assign separate LMs to each agent for decentralized action prediction, allowing natural language communication between agents. However, RoCo and CoNavGPT require detailed environment information for planning, and CoELA's action space is filtered by an oracle. Relying on privileged information from an oracle is impractical in real-world applications. By contrast, our work focuses on free-form action generation and handles tasks with more ambiguous descriptions. Prior work [16] compare centralized (CMAS) and decentralized (DMAS) planning frameworks, showing that centralized planners perform better, though their experiments are in simple, known environments with limited number of agents. Two-Step [21] decomposes goals for main and helper agents, using PDDL planners for high-level actions. SmartLLM [19] uses multiple LLM modules for subtask decomposition, multi-robot group formation and task allocation but assumes robots have complete knowledge of the environment, making plans prone to errors in unknown settings. S-ATLAS [20] use LLMs with conformal prediction for safe multi-agent planning, but the action choices are limited to a small set of objects. Table 1 presents a comparison of the characteristics of different LM-based approaches to multi-agent planning with our work.

LMs can interpret high-level instructions and break them down into feasible subtasks, making them ideal for long-horizon, multi-task scenarios. Our work leverages LMs to enable long-horizon planning across a variety of tasks and environments, building on these advances to address the limitations of traditional RL and HRL methods. By integrating LMs into our planning framework, we enhance the ability to generalize across diverse tasks and scenarios, making significant strides toward practical, real-world applications of RL in dynamic, multi-agent settings.

<table><tr><td>Method</td><td>Dynamic Planning</td><td>Local Information</td><td>Failure Correction</td><td>Self Verification</td></tr><tr><td>Two-Step [21]</td><td>✘</td><td>✘</td><td>✘</td><td>✘</td></tr><tr><td>Smart LLM [19]</td><td>✘</td><td>✘</td><td>✘</td><td>✘</td></tr><tr><td>S-ATLAS [20]</td><td>✓</td><td>✘</td><td>✘</td><td>✘</td></tr><tr><td>CoELA [22]</td><td>✓</td><td>✓</td><td>✘</td><td>✘</td></tr><tr><td>LLaMAR (this paper)</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td></tr></table>

Table 1: The proposed model, LLaMAR: 1) performs dynamic planning, avoiding the open-loop plan-and-execute paradigm; 2) operates without privileged simulator information (e.g., access to all objects in the environment); 3) re-plans when low-level actions fail, not assuming perfect execution; and 4) self-verifies subtask completion without relying on the simulator.

## 3 Background

Problem Setting: We consider a setting where multiple robots perform a series of tasks (a) such as cleaning a room or putting groceries in the fridge, in a home-like environment, and (b) rescuing missing personnel and putting out forest fires in a search & rescue environment (SAR). These tasks typically require long-horizon planning, involving around 100 low-level actions to reach the goal. Our objective is to compute plans for a team of robots to execute high-level language instructions, $I$ . We formalize these tasks as partially observable Markov decision processes (POMDP) [55, 56], denoted as $\left\langle  {N,\mathcal{I},\mathcal{S},\left\{  {\mathcal{O}}_{i}\right\}  ,\left\{  {\mathcal{A}}_{i}\right\}  ,\mathcal{P},\mathcal{G},T}\right\rangle  .N$ is the number of agents and $\mathcal{I}$ is the high-level language instruction set. Here, $s \in  \mathcal{S}$ represents the joint state of all agents,and $o \in  \mathcal{O}$ denotes the observation set for all agents. Particularly, ${o}_{i} \in  {\mathcal{O}}_{i}$ is the observation set of agent $i$ ,that captures incomplete environment state information. $a \in  \mathcal{A} = {\mathcal{A}}_{1} \times  {\mathcal{A}}_{2}\cdots {\mathcal{A}}_{N}$ represents the joint action space. The joint action space comprises of different categories of high-level actions $\mathcal{A} = {\mathcal{A}}_{NAV} \cup  {\mathcal{A}}_{INT} \cup  {\mathcal{A}}_{EXP}$ , where ${\mathcal{A}}_{NAV}$ is the joint navigation action set, ${\mathcal{A}}_{INT}$ is the joint interaction actions which allow the agents to interact with objects,and ${\mathcal{A}}_{EXP}$ are the joint exploration actions which allow the agents to explore the environment. Examples of the high-level actions include PickUp(object) $\in  {\mathcal{A}}_{INT}$ and NavigateTo(location) $\in  {\mathcal{A}}_{NAV}$ . Each high-level action is associated with a low-level primitive action (pre-trained RL, behavior cloned, or heuristic-based policy). These actions are executed synchronously by all agents at every high-level decision step. $\mathcal{P}\left( {{s}^{\prime } \mid  s,a}\right)$ is the joint transition probability function that defines the probability of arriving at ${s}^{\prime } \in  \mathcal{S}$ after taking joint action $a \in  \mathcal{A}$ in $s \in  \mathcal{S}.\mathcal{G} = \left\{  {{g}_{1},\cdots ,{g}_{k}}\right\}$ defines the subtasks that the agents need to perform to accomplish the language instruction task. $T$ is the length of the planning horizon.

<img src="https://cdn.noedgeai.com/bo_d4nfsebef24c73bbfgrg_3.jpg?x=313&y=211&w=1178&h=711&r=0"/>

Figure 1: An overview of LLaMAR's modular cognitive architecture. LLaMAR leverages LMs within four key modules: Planner, Actor, Corrector, and Verifier, each with specific roles. The Planner breaks down the high-level language instruction into feasible subtasks to achieve the environment goal. The Actor determines the high-level actions each agent should perform. These actions trigger low-level policies that generate and execute a sequence of primitive actions in sync across all agents. Based on execution feedback, the Corrector suggests corrections for high-level actions and the Verifier Module validates completion of subtasks.

Environment: To simulate open-ended, long-horizon tasks that resemble everyday activities in a home-like environment, we use the AI2Thor simulator [41], which supports a diverse set of interactions and photorealistic rendering. Since our approach does not require any parametric training, it can potentially translate to other similar embodied environments like VirtualHome [57], Habitat [42, 58, 59], and ThreeDWorld [60], possibly extending beyond household domains. Similarly, to simulate the search & rescue scenario, we create a custom Search & Rescue environment (SAR). More information about the MAP-THOR and SAR environments can be found in Appendix B and D respectively. We instantiate the problem with $N$ agents cooperating to accomplish a long-horizon rearrangement task [61] in an indoor environment. The agents do not know the objects present in the environment a priori and are encouraged to explore the environment to gather more information to complete the task. Unlike previous solutions in similar settings [22, 19, 20], we do not rely on an oracle/simulator to verify subtask completion. In prior works, a predefined conditional satisfiability check is used as subtask completion feedback.

## 4 Approach

We describe our approach in this section. Figure 1 illustrates LLaMAR's architecture comprising four modules: Planner, Actor, Corrector, and Verifier, each an LM with a distinct role. Prior work [62] shows that splitting roles across different LMs improves performance in sequential decision-making. Our initial experiments confirm that LMs tasked with reasoning about multiple inputs and providing long outputs perform poorly. We iterate through these four modules at every high-level decision step. The pseudocode for our approach is in Appendix E We define some key notation below:

- Memory $\mathcal{M}$ : A textual description of the joint memory of all agents,summarizing past observations, high-level actions, plausible reasons for action failures, and specific subtasks that each agent is attempting to solve.

- Open Subtasks ${\mathcal{G}}_{O} \subset  \mathcal{G}$ : Feasible subtasks proposed by the Planner LM to achieve the environment task that are yet to be accomplished by the agents.

- Completed Subtasks ${\mathcal{G}}_{S} \subset  \mathcal{G}$ : Subtasks completed by the agents.

- Corrective Actions ${a}_{c}$ : Corrective actions for each agent based on failure information from the previous step.

At the start of each episode,Memory $\mathcal{M}$ ,Open Subtasks ${\mathcal{G}}_{O}$ ,Completed Subtasks ${\mathcal{G}}_{S}$ ,Actions $a$ , Corrective Actions ${a}_{c}$ ,and Failure Information $\mathcal{F}$ are initialized as empty sets.

Consider an example of a kitchen with groceries, a fridge, and a counter. Two agents are tasked with "Fetch the groceries and place them in the fridge". This example will help illustrate the utility of each module. All LMs receive a language task instruction $\mathcal{I}$ ,joint observations from all agents, and information about open and completed subtasks and memory unless stated otherwise. We next discuss the various components in our architecture in detail:

Planner Module The Planner LM module suggests feasible subtasks to ensure the completion of the environment task. This method, similar to SmartLLM's [19] zero-shot planning, uses only observations in the agents' field of view. The Planner suggests subtasks related to objects seen in the current observation or memory of all the agents. For the example considered, it decomposes the task into subtasks like "transport the tomato to the fridge" and "transport the lettuce to the fridge", which are added to ${\mathcal{G}}_{O}$ .

Actor Module The Actor LM additionally uses corrective actions suggested by the Corrector module in the previous time step to predict high-level actions for the current step. These actions are then executed in the environment to progress a subset of subtasks in the open subtask set ${\mathcal{G}}_{O}$ and accordingly updates the joint memory. For instance, the Actor module might suggest actions such as $a =$ [Pickup(Tomato),NavigateTo(Lettuce)],updating memory with "We saw a tomato on the counter-top, Alice is picking up the tomato, and Bob is navigating to the lettuce".

Corrector Module The Corrector LM self-corrects high-level actions suggested by the Actor LM after controller failures in the previous step's execution ${}^{2}$ It suggests corrective high-level actions and provides reasons for failures and chosen corrections. For example, it might suggest "The action of picking up the tomato failed because it is too far away. Alice first needs to navigate closer to the tomato."; ${a}_{c} =$ [NavigateTo(Tomato),None].

Verifier Module After executing high-level actions, the Verifier LM assesses whether these actions have completed any subtasks in the open subtask set. Successful subtasks are moved to the completed subtask set. Without the Verifier LM, the method would need to rely on the simulator/oracle for success or failure information. The Verifier LM along with other information uses the successfully executed high-level actions proposed by the Actor LM to predict subtask completion. For example, after transporting the lettuce to the fridge, the Verifier updates the completed subtasks with "transport lettuce to the fridge".

Admissible Action parsing with Semantic Translation When LMs generate action plans, natural language outputs often fail to translate to executable high-level actions. This happens when the output does not match the predefined format or refers to unrecognized contextually similar objects. We use a cosine similarity method from [13], fine-tuning a pre-trained sentence-BERT [63] to transform the free-form text into admissible high-level actions. Hyperparameters and additional details of the sentence transformer fine-tuning are provided in Appendix I

Exploration Strategy In unexplored environments, agents need to search for task-relevant objects. If agents cannot find the required objects, the language model can choose an 'exploration' action ${a}_{exp}$ . We use a semantically-guided heuristic to determine the choice of region to be explored. The agent rotates to four cardinal directions $d \in$ North,South,East,West,capturing image observations ${o}_{n,d}$ . These images are processed through a pre-trained CLIP image encoder [64] to obtain embeddings ${I}_{d}$ . The list of open subtasks ${\mathcal{G}}_{O}$ is processed through the corresponding CLIP text encoder to get text embeddings ${g}_{O,i}$ . The exploration score ${\mathcal{E}}_{d}$ in direction $d$ is defined as ${\mathcal{E}}_{d} = \mathop{\sum }\limits_{{i = 1}}^{\left| {\mathcal{G}}_{O}\right| }\frac{{g}_{O,i} \cdot  {I}_{d}}{\begin{Vmatrix}{g}_{O,i}\end{Vmatrix}\begin{Vmatrix}{I}_{d}\end{Vmatrix}}$ . The direction with the highest score ${d}^{ * } = \arg \mathop{\max }\limits_{d}{\mathcal{E}}_{d}$ is chosen. Summing the scores helps select the best direction to explore in expectation. The agent rotates towards ${d}^{ * }$ and moves $J = 2$ steps,repeating this process $K = 3$ times in one explore action. This approach ensures that images relevant to identifying potential subtasks are prioritized. For example,if ${\mathcal{G}}_{O}$ includes "locate a computer", it is more likely to find a computer on a table than on a sofa, resulting in a higher cosine similarity score between the subtask CLIP text embedding and table CLIP image embedding. Refer to Appendix B. 2 for more details about the exploration heuristic.

---

${}^{2}$ We use the simulator just to provide a boolean value about the success of high-level action execution.

---

Motivation for Proposed Framework The specific order of the modules in LLaMAR is due to natural causal relationships in which environment feedback is received. We use the Planner as the first module because it allows LLaMAR to come up with an initial list of open subtasks that could be completed based on the current observation and past memory to satisfy the task. This list serves as a rough high-level plan. The actor then uses this information to suggest the necessary actions. The Corrector is used after the Actor module to identify reasons for failures in the execution of the actions suggested by the Actor . Note that the failure module is inert and only suggests corrective actions. Only the Actor module decides the final actions to be executed. This role distinction allows for clear reasoning on failures when they occur and lets the actor module focus on choosing actions. The Verifier is used after the action is executed to update the list of closed subtasks so that LLaMAR can be current with the progress toward the completion of the environment task. This allows the planner to update the list of open subtasks in the next step. In essence, the Planner and the Verifier ensure that the progress of the agents is tracked and the actor and the corrector ensure that the actions are executed successfully to advance towards completion of the task.

Multi-Agent features in LLaMAR While our method can be easily adapted to a single-agent setting, our design choice for the architecture was motivated to include the following multi-agent features:

- Coordination through communication: Agents share their state information with the centralized LLaMAR modules to predict actions, enabling them to coordinate and avoid conflicts. This information sharing allows for the agents to cooperate and achieve the collective goal.

- Dynamic Role Assignment: Agents are dynamically assigned roles based on the current task requirements and their capabilities. This flexibility allows LLaMAR to adapt to changing environments and task demands.

- Hierarchical Task Decomposition: To handle the complexity of multi-agent planning, LLaMAR decomposes the action space by creating specific subgoals/subtasks available for any agent to assign itself (done by the actor module) based on the observation and current context. This decomposition reduces the overall search space and improves planning efficiency.

## 5 Experiments

MAP-THOR: To evaluate the performance of LLaMAR and benchmark other baseline methods, we create a benchmark dataset of tasks which we call MAP-THOR (Multi-Agent Planning tasks in AI2-THOR). While Smart-LLM [19] introduces a dataset of 36 tasks within AI2-Thor [41] classified by complexity, their tasks are limited to single floor plans. This limitation hinders testing the robustness of planners across different room layouts. Additionally, some tasks in their dataset cannot be performed by multiple agents, regardless of task division, such as Pick up the pillow, Open the laptop to turn it on, and Turn off the lamp.

By contrast, MAP-THOR includes tasks solvable by both single and multiple agents. We classify the tasks into four categories based on the ambiguity of the language instructions. To test the planner robustness, we provide five different floor plans for each task. We also include automatic checker modules to verify subtask completion and evaluate plan quality. Our dataset comprises 45 tasks, each defined for five distinct floor plans, ensuring comprehensive testing and evaluation.

We conduct experiments with tasks of varying difficulty levels, where an increase in difficulty of the tasks corresponds to an increased ambiguity in the language instructions. The complete task list of each category can be found in the Appendix C

- Explicit item type, quantity, and target location: Agents are explicitly instructed to transport specific items to specific target locations. For example, put bread, lettuce, and a tomato in the fridge clearly defines the objects (tomato, lettuce, bread) and the target (fridge).

- Explicit item type and target location but implicit item quantity: The object type is explicitly described, but its quantity is not disclosed. For example, Put all the apples in the fridge. Agents must explore the environment to locate all specified items and also predict when to stop.

- Explicit target location but implicit item types and quantity: The target location is explicitly defined but the item types and their quantities are concealed. For example, Put all groceries in the fridge.

- Implicit target location, item type, and quantity: Item types and their quantities along with the target location are implicitly defined. For example, Clear the floor by placing the items at their appropriate positions. The agent is expected to place items like pens, books, and laptops on the study table, and litter in the trash can.

Search & Rescue Environment (SAR): To showcase the effectiveness of LLaMAR with respect to explicit coordination in multi-agent settings, we evaluate LLaMAR in a partially observable search & rescue and fire relief environment in a grid world. Depending on the scene, there is a mix of missing people to be found, and wildfires to be stopped before they spread geographically. More details about the environment can be found in Appendix D.

- Fire Extinguishing: Fires consist of expansive flammable regions with a fixed set of sources that propagate over time. The rate of fire spread is proportional to its intensity; higher intensities result in faster spread. Fires are categorized as either Class A or Class B, which are extinguished using water or sand, respectively. These extinguishing resources are sourced from reservoirs distributed across the environment.

- Human Rescue: Each individual is initially located at an unknown position within the environment. The objective is to locate, rescue, and transport the individuals to a designated drop-off location, which is known beforehand. Transporting a person requires the coordinated effort of two agents simultaneously, who must carry them to the specified drop-off point.

## Metrics

We evaluate the algorithms using the following metrics to compare their performances on the tasks:

- Success Rate (SR): The fraction of episodes in which all subtasks are completed. Success equals 1 if all subtasks are successfully executed in an episode, otherwise it is 0.

- Transport Rate (TR): The fraction of subtasks completed within an episode, provides a finer granularity of task completion.

- Coverage (C): The fraction of successful interactions with target objects. It is useful to verify if the LMs can infer the objects to interact with, in scenarios where the tasks have objects that are specified implicitly.

- Balance (B): The ratio between the minimum and maximum number of successful high-level actions executed by any agent that contributed towards task completion. We only check for a subset of high-level actions that must be executed to accomplish critical subtasks that lead to the successful completion of the language instruction task. If each agent $i$ out of $n$ agents completes ${s}_{i}$ successful tasks,the balance is defined as: $B \mathrel{\text{ := }} \frac{\min \left\{  {{s}_{1},\cdots ,{s}_{n}}\right\}  }{\max \left\{  {{s}_{1},\cdots ,{s}_{n}}\right\}   + \epsilon }$ . This measures how evenly the work is distributed among agents. A balance of zero indicates at least one agent performed no successful high-level actions, while a balance of one indicates all agents performed the same number of successful high-level actions. Here $\epsilon  = {1e} - 4$ is a small number to avoid division by zero.

- Average steps (L): The number of high-level actions taken by the team to complete the task, capped at $L = {30}$ in our experiments. If the task is not completed within $L$ steps,the episode is deemed a failure. Note that the metric $L$ is presented in the table providing the complete results, located in Appendix F.

For all the metrics, we report the means along with the 95% confidence interval across all the tasks (refer Appendix F for complete results). Since SR is a binomial metric, we report the Clopper-Pearson Interval as the confidence interval.

## Baselines

For a fair comparison with our method, we make modifications to the baselines to make them work in partially observable settings with limited reliance on the simulator. More details about implementations can be found in Appendix H

- Act: We query the LLM with the task and the observations to suggest a high-level action.

- Chain-of-Thought [65]: We modify the Act prompt with a chain-of-thought style addendum to let the LM reason about the possible implications while selecting a high-level action.

- ReAct [66]: We use a ReAct-style prompting to let the LMs reason after suggesting high-level actions and possibly suggest ways to correct any failures.

- SmartLLM [19]: We modify the official codebase to only include information from the local observations of the agents instead of assuming full observability.

- CoELA [22]: We modify the list of available high-level actions to include all possible valid combinations of actions with interactable objects in the agent's local observation. As the scene becomes more cluttered, this list and the prompt becomes combinatorially longer. In the original implementation, the list of available actions is filtered based on the feasibility of the actions as suggested by a conditional checker.

It should be noted that Act, Chain-of-Thought, ReAct, and SmartLLM are all CMAS frameworks where CoELA follows the DMAS framework.

## 6 Results and Discussion

Choice of the underlying LM: To understand the impact of the underlying LM's quality on decision-making, we initially experimented with different LMs on MAP-THOR. Specifically, we utilize both the language-only and vision-language models of GPT-4 [67], IDEFICS-2 [68, 69], LLaVA [70, 71], and CoGVLM [72]. Among these, GPT-4, when used solely with text inputs, exhibits the poorest performance. This is attributed to the agents' inability to reason about visual observations, which is particularly detrimental for the Corrector module. Substituting GPT-4V with other vision-language models results in a decline in performance (refer Table 2) and hence we use GPT-4V as the underlying VLM while comparing to the baselines.

Baseline Comparisons: Table 2 compares our method, LLaMAR, with other baselines in a 2-agent scenario using GPT-4V as the underlying VLM. Act and ReAct show similar performance, with Act struggling due to its lack of strategic planning or correction, and ReAct performing slightly better by dynamically adjusting actions based on reasoning on immediate feedback. CoT's performance declines with longer planning horizons due to its inability to maintain coherence over extended planning sequences, consistent with findings in [73], showing its effectiveness only with highly specific prompts. SmartLLM, operating in a plan-and-execute paradigm, generates impractical plans with issues like infinite loops and failure to handle low-level action failures, leading to lower success rates and poor transport metrics. It also tends to hallucinate objects. CoELA, using a decentralized multi-agent system (DMAS), performs poorly due to large input prompts and struggles to select the correct action from numerous choices. Its decentralized decision-making is less efficient than the centralized multi-agent system (CMAS) used by LLaMAR. Previous research [16] confirms CMAS frameworks are more effective than DMAS frameworks. Overall, our method, LLaMAR, benefits from its modular cognitive architecture, which integrates planning, acting, correcting, and verifying through distinct LLM roles, resulting in superior performance across various evaluation metrics. By avoiding reliance on privileged information and incorporating a robust exploration strategy that allows it to scout for objects that are not initially visible, LLaMAR ensures higher success rates and balanced task execution among agents.

Roles of different modules in LLaMAR: To demonstrate the effectiveness of the various modules in our cognitive architecture, we performed ablation studies by evaluating performance metrics with each module removed individually. The results are summarized in Table 3 Using only the Actor module corresponds to the "Act" baseline, which demonstrates its fundamental capabilities in isolation but shows limited effectiveness without planning and correction due to relatively lower success and transport rates. Adding the Planner and Verifier modules improves performance, benefiting from better task planning and validation, increasing the overall SR and TR, and ensuring more effective task completion and even work distribution, as indicated by the increase in balance (B). However, in scenarios where the suggested action fails, the actor suggests the same action in the next decision step since it is not able to reason why the action failed until the end of the planning horizon. Incorporating the Corrector module, with access to privileged information from an environment oracle, significantly boosts performance, enhancing the SR, TR, C, and further improving B, consistent with the findings in [17]. This highlights the Corrector module's importance in adjusting actions based on controller feedback, resulting in higher task success and more efficient task completion, albeit with reliance on oracle knowledge. Finally, the complete LLaMAR system, without privileged information, achieves SR, TR, C, and B values close to those of the oracle setup. This demonstrates the system's robustness and effectiveness in a realistic setting. The Corrector module plays a crucial role in enabling agents to learn from past failures and avoid repeating actions, preventing task failures due to timeout. Despite lacking oracle knowledge, LLaMAR performs nearly as well as the oracle-enhanced setup. These results highlight the importance of each module in our cognitive architecture. Removing any module diminishes effectiveness.

Increasing the number of agents Increasing the number of agents in the environment shows distinct trends in our method's performance metrics for both MAP-THOR and SAR environments (refer Table 4). In MAP-THOR, with two agents, we establish a solid baseline for success rate (SR) and transport rate (TR), which is similarly reflected in the SAR environment. Adding a third agent improves both SR and TR in both environments, indicating enhanced task completion and transportation efficiency. Coverage (C) also increases, suggesting better exploration and interaction with objects across both environments. There is a slight decrease in SR and TR when the number of agents increases from 3 to 5 in the MAP-THOR environment. The decrease in these metrics can be attributed to the rooms in MAP-THOR becoming crowded with 4 and 5 agents hence blocking the agents from navigating without colliding with other agents. But this phenomenon is not seen in the SAR environment which is comparatively more spacious and navigable. However, balance (B), which measures the even distribution of tasks among agents, decreases with more agents. This drop highlights the challenge of ensuring equal contributions from all agents in a larger multi-agent system. While SR remains high, the balance metric drops significantly from 2 to 5 agents, indicating some agents do more work than others. In summary, adding more agents improves task performance and efficiency but introduces challenges in maintaining balanced contributions. Addressing this imbalance is crucial for refining multi-agent planning algorithms.

<table><tr><td>Algorithm</td><td>LM</td><td>SR↑</td><td>TR↑</td><td>C↑</td><td>B↑</td></tr><tr><td>Act</td><td>GPT-4V</td><td>0.33</td><td>0.67</td><td>0.91</td><td>0.59</td></tr><tr><td>ReAct</td><td>GPT-4V</td><td>0.34</td><td>0.72</td><td>0.92</td><td>0.67</td></tr><tr><td>CoT</td><td>GPT-4V</td><td>0.14</td><td>0.59</td><td>0.87</td><td>0.62</td></tr><tr><td>SmartLLM</td><td>GPT-4V</td><td>0.11</td><td>0.23</td><td>0.91</td><td>0.45</td></tr><tr><td>CoELA</td><td>GPT-4V</td><td>0.25</td><td>0.46</td><td>0.76</td><td>0.73</td></tr><tr><td>LLaMAR</td><td>GPT-4</td><td>0.51</td><td>0.85</td><td>0.95</td><td>0.83</td></tr><tr><td>LLaMAR</td><td>LLaVA</td><td>0.54</td><td>0.84</td><td>0.91</td><td>0.75</td></tr><tr><td>LLaMAR</td><td>IDEFICS-2</td><td>0.57</td><td>0.86</td><td>0.94</td><td>0.78</td></tr><tr><td>LLaMAR</td><td>CogVLM</td><td>0.61</td><td>0.89</td><td>0.95</td><td>0.80</td></tr><tr><td>LLaMAR (w/o expl)</td><td>GPT-4V</td><td>0.62</td><td>0.87</td><td>0.95</td><td>0.82</td></tr><tr><td>LLaMAR (w/ expl)</td><td>GPT-4V</td><td>0.66</td><td>0.91</td><td>0.97</td><td>0.82</td></tr></table>

Table 2: Comparison of evaluation metrics against baselines averaged across all tasks for the 2-agent MAP-THOR scenarios. The complete table with confidence intervals can be found in Appendix F More details about peculiar behaviors for the baselines can be found in Appendix H

<table><tr><td>Modules Used</td><td>SR $\uparrow$</td><td>TR↑</td><td>C↑</td><td>B↑</td></tr><tr><td>Actor</td><td>0.33</td><td>0.67</td><td>0.91</td><td>0.59</td></tr><tr><td>Planner + Actor + Verifier</td><td>0.45</td><td>0.78</td><td>0.92</td><td>0.69</td></tr><tr><td>Planner + Actor + Corrector‡</td><td>0.67</td><td>0.91</td><td>0.97</td><td>0.84</td></tr><tr><td>Planner + Actor + Corrector + Verifier +</td><td>0.66</td><td>0.91</td><td>0.97</td><td>0.82</td></tr></table>

Table 3: Performance in the 2-agent scenarios in MAP-THOR obtained by ablating different modules in LLaMAR with GPT-4V as the underlying LM

<table><tr><td rowspan="2">#of agents</td><td colspan="4">MAP-THOR</td><td colspan="4">SAR</td></tr><tr><td>SR↑</td><td>TR↑</td><td>C↑</td><td>B↑</td><td>SR↑</td><td>TR↑</td><td>C↑</td><td>B↑</td></tr><tr><td>1</td><td>0.37</td><td>0.67</td><td>0.87</td><td>1.00</td><td>0.28</td><td>0.75</td><td>0.86</td><td>1.00</td></tr><tr><td>2</td><td>0.62</td><td>0.87</td><td>0.95</td><td>0.82</td><td>0.44</td><td>0.86</td><td>0.94</td><td>0.91</td></tr><tr><td>3</td><td>0.70</td><td>0.91</td><td>0.98</td><td>0.66</td><td>0.68</td><td>0.92</td><td>0.96</td><td>0.80</td></tr><tr><td>4</td><td>0.68</td><td>0.90</td><td>0.99</td><td>0.62</td><td>0.72</td><td>0.94</td><td>0.98</td><td>0.78</td></tr><tr><td>5</td><td>0.62</td><td>0.90</td><td>0.99</td><td>0.54</td><td>0.74</td><td>0.96</td><td>1.00</td><td>0.73</td></tr></table>

Table 4: LLaMAR with various number of agents in the scenario in both MAP-THOR and SAR environments

Correcting Failures: In numerous instances, the actions proposed by the Actor module, such as pick up <object>, are unsuccessful due to the agent's insufficient proximity to the target object. In such situations, the Corrector module uses visual feedback to learn from these failures and recommends appropriate corrective actions, such as navigate to <object> to facilitate closer proximity. Figure 2 shows examples where the Corrector module interprets low-level action failures and suggests remedies, highlighting its importance.

## 7 Limitations and Future Work

Higher number queries to the LM: Since each high-level decision step requires querying 4 different LM-based modules, the cost and the compute times are higher than other baselines, especially compared to the plan-and-execute baselines like SmartLLM. An interesting future direction to

<img src="https://cdn.noedgeai.com/bo_d4nfsebef24c73bbfgrg_9.jpg?x=308&y=200&w=1183&h=341&r=0"/>

Figure 2: A few examples of the Corrector module mitigate failures in predicted actions by the Actor module. (a) the Corrector suggests getting closer to the agent before attempting to pick it up, (b) the Corrector recommends opening the fridge because the previous action of placing the plate failed, (c) the Corrector advises rotating right so that it can access the table to place the tissue box on it when the low-level navigation policy failed to find a path to the table

improve this would be to fine-tune smaller LMs with trajectories collected in the simulator (eg: ALFRED [46]) as done in [22]. Another potential direction worth exploring is using different sizes of LMs for each module based on their specific utility.

Limited spatial reasoning: Although we use both textual descriptions and visual features to guide the language model's actions, it still lacks the ability to reason about the spatial features of the environment. Spatial reasoning is crucial in scenarios such as navigating around obstacles to reach an object, or determining the shortest path to collect multiple items scattered across different locations. One way to address this limitation is to inject information about the 3D world into the LM, as done in [74], which is an interesting direction for future work.

Performance limited by the underlying VLM: Although LMs make correct reasoning most of the time, they still occasionally make mistakes, including misunderstanding the environment rules specified in the prompt. For example, the agent assumes that the cleaning task requires putting soap, drying, and putting it in the sink when all it needs is the action "CleanObject", and can't figure out the appropriate level of abstraction. The performance of the algorithm is limited by the instruction following and reasoning capability of the underlying LM [75, 76]; this calls for developing LMs that are fine-tuned to instruction-image pairs relevant to the environment (as done in [22]).

## 8 Conclusion

We address long-horizon planning in dynamic, partially observable multi-agent environments with LLaMAR, an LM-based planner using four specialized modules: Planner, Actor, Corrector, and Verifier. This framework iteratively refines action planning, adapts to failures, and verifies subtask completion using real-time observations and action feedback, without privileged information. We also introduce a heuristic-based exploration strategy to guide agents to semantically relevant regions. Additionally, we present MAP-THOR, a benchmark dataset for multi-agent tasks in the AI2Thor simulator. Empirical results show LLaMAR outperforms existing LM-based approaches, achieving a 30% higher success rate on MAP-THOR.

## Acknowledgements

We would like to thank Keerthana Gopalakrishnan, Sydney Dolan, Jasmine Aloor, and Victor Qin for helpful discussions and feedback. OpenAI credits for GPT-4 access was provided through OpenAI's Researcher Access Program. The research was sponsored by the Department of the Air Force Artificial Intelligence Accelerator and was accomplished under Cooperative Agreement Number FA8750-19-2-1000. The views and conclusions contained in this document are those of the authors and should not be interpreted as representing the official policies, either expressed or implied, of the Department of the Air Force or the U.S. Government. The U.S. Government is authorized to reproduce and distribute reprints for Government purposes notwithstanding any copyright notation herein.

## References

[1] Unnat Jain, Luca Weihs, Eric Kolve, Ali Farhadi, Svetlana Lazebnik, Aniruddha Kembhavi, and Alexander G. Schwing. A Cordial Sync: Going Beyond Marginal Policies For Multi-Agent Embodied Tasks. In ECCV, 2020. first two authors contributed equally. 1

[2] Unnat Jain, Luca Weihs, Eric Kolve, Mohammad Rastegari, Svetlana Lazebnik, Ali Farhadi, Alexander G. Schwing, and Aniruddha Kembhavi. Two Body Problem: Collaborative Visual Task Completion. In CVPR, 2019. first two authors contributed equally. 1

[3] Vijay Kumar, D. Rus, and Sanjiv Singh. Robot and sensor networks for first responders. IEEE Pervasive Computing, 3(4):24-33, 2004. 1

[4] Matthew Dunbabin and Lino Marques. Robots for Environmental Monitoring: Significant Advancements and Applications. IEEE Robotics & Automation Magazine, 19(1):24-39, 2012. 1

[5] Brian Ichter, Anthony Brohan, Yevgen Chebotar, Chelsea Finn, Karol Hausman, Alexander Herzog, Daniel Ho, Julian Ibarz, Alex Irpan, Eric Jang, Ryan Julian, Dmitry Kalashnikov, Sergey Levine, Yao Lu, Carolina Parada, Kanishka Rao, Pierre Sermanet, Alexander T Toshev, Vincent Vanhoucke, Fei Xia, Ted Xiao, Peng Xu, Mengyuan Yan, Noah Brown, Michael Ahn, Omar Cortes, Nicolas Sievers, Clayton Tan, Sichun Xu, Diego Reyes, Jarek Rettinghouse, Jornell Quiambao, Peter Pastor, Linda Luu, Kuang-Huei Lee, Yuheng Kuang, Sally Jesmonth, Nikhil J. Joshi, Kyle Jeffrey, Rosario Jauano, Jasmine Hsu, Keerthana Gopalakrishnan, Byron David, Andy Zeng, and Chuyuan Kelly Fu. Do As I Can, Not As I Say: Grounding Language in Robotic Affordances. In Karen Liu, Dana Kulic, and Jeff Ichnowski, editors, Proceedings of The 6th Conference on Robot Learning, volume 205 of Proceedings of Machine Learning Research, pages 287-318. PMLR, 14-18 Dec 2023. 1.2

[6] Wenlong Huang, Fei Xia, Dhruv Shah, Danny Driess, Andy Zeng, Yao Lu, Pete Florence, Igor Mordatch, Sergey Levine, Karol Hausman, and Brian Ichter. Grounded Decoding: Guiding Text Generation with Grounded Models for Embodied Agents. In A. Oh, T. Naumann, A. Globerson, K. Saenko, M. Hardt, and S. Levine, editors, Advances in Neural Information Processing Systems, volume 36, pages 59636-59661. Curran Associates, Inc., 2023. 12

[7] Wenlong Huang, Fei Xia, Ted Xiao, Harris Chan, Jacky Liang, Pete Florence, Andy Zeng, Jonathan Tompson, Igor Mordatch, Yevgen Chebotar, Pierre Sermanet, Tomas Jackson, Noah Brown, Linda Luu, Sergey Levine, Karol Hausman, and Brian Ichter. Inner Monologue: Embodied Reasoning through Planning with Language Models. In Karen Liu, Dana Kulic, and Jeff Ichnowski, editors, Proceedings of The 6th Conference on Robot Learning, volume 205 of Proceedings of Machine Learning Research, pages 1769-1782. PMLR, 14-18 Dec 2023. 1

[8] Ishika Singh, Valts Blukis, Arsalan Mousavian, Ankit Goyal, Danfei Xu, Jonathan Tremblay, Dieter Fox, Jesse Thomason, and Animesh Garg. ProgPrompt: Generating Situated Robot Task Plans using Large Language Models. In 2023 IEEE International Conference on Robotics and Automation (ICRA), pages 11523-11530, 2023. 12

[9] Jacky Liang, Wenlong Huang, Fei Xia, Peng Xu, Karol Hausman, Brian Ichter, Pete Florence, and Andy Zeng. Code as Policies: Language Model Programs for Embodied Control. In arXiv preprint arXiv:2209.07753, 2022. [1]

[10] Kevin Lin, Christopher Agia, Toki Migimatsu, Marco Pavone, and Jeannette Bohg. Text2Motion: from natural language instructions to feasible plans. Autonomous Robots, 47(8):1345-1365, November 2023. 1

[11] Dhruv Shah, Blazej Osinski, Brian Ichter, and Sergey Levine. LM-Nav: Robotic Navigation with Large Pre-Trained Models of Language, Vision, and Action. In 6th Annual Conference on Robot Learning, 2022. 1

[12] Chenguang Huang, Oier Mees, Andy Zeng, and Wolfram Burgard. Visual Language Maps for Robot Navigation. In Proceedings of the IEEE International Conference on Robotics and Automation (ICRA), London, UK, 2023. 1

[13] Wenlong Huang, Pieter Abbeel, Deepak Pathak, and Igor Mordatch. Language Models as Zero-Shot Planners: Extracting Actionable Knowledge for Embodied Agents. In Kamalika Chaudhuri, Stefanie Jegelka, Le Song, Csaba Szepesvari, Gang Niu, and Sivan Sabato, editors, Proceedings of the 39th International Conference on Machine Learning, volume 162 of Proceedings of Machine Learning Research, pages 9118-9147. PMLR, 17-23 Jul 2022. 12 25

[14] Ming Tan. Multi-Agent Reinforcement Learning: Independent vs. Cooperative Agents. In In Proceedings of the Tenth International Conference on Machine Learning, pages 330-337. Morgan Kaufmann, 1993. 1

[15] Ardi Tampuu, Tambet Matiisen, Dorian Kodelja, Ilya Kuzovkin, Kristjan Korjus, Juhan Aru, Jaan Aru, and Raul Vicente. Multiagent Cooperation and Competition with Deep Reinforcement Learning. CoRR, abs/1511.08779, 2015. 1

[16] Yongchao Chen, Jacob Arkin, Yang Zhang, Nicholas Roy, and Chuchu Fan. Scalable Multi-Robot Collaboration with Large Language Models: Centralized or Decentralized Systems? arXiv preprint arXiv:2309.15943, 2023. 1233.

[17] Daman Arora and Subbarao Kambhampati. Learning and Leveraging Verifiers to Improve Planning Capabilities of Pre-trained Language Models. arXiv preprint arXiv:2305.17077, 2023. 2. B

[18] Karthik Valmeekam, Matthew Marquez, and Subbarao Kambhampati. Can Large Language Models Really Improve by Self-critiquing Their Own Plans? arXiv preprint arXiv:2310.08118, 2023. 2

[19] Shyam Sundar Kannan, Vishnunandan LN Venkatesh, and Byung-Cheol Min. SMART-LLM: Smart Multi-Agent Robot Task Planning using Large Language Models. arXiv preprint arXiv:2309.10062, 2023. 234567

[20] Jun Wang, Guocheng He, and Yiannis Kantaros. Safe Task Planning for Language-Instructed Multi-Robot Systems using Conformal Prediction. arXiv preprint arXiv:2402.15368, 2024. 2. 3. 4. 24

[21] Ishika Singh, David Traum, and Jesse Thomason. TwoStep: Multi-agent Task Planning using Classical Planners and Large Language Models. arXiv preprint arXiv:2403.17246, 2024. 2 3 24

[22] Hongxin Zhang, Weihua Du, Jiaming Shan, Qinhong Zhou, Yilun Du, Joshua B Tenenbaum, Tianmin Shu, and Chuang Gan. Building Cooperative Embodied Agents Modularly with Large Language Models. ICLR, 2024.204810

[23] Bangguo Yu, Hamidreza Kasaei, and Ming Cao. Co-NavGPT: Multi-Robot Cooperative Visual Semantic Navigation using Large Language Models. ArXiv, abs/2310.07937, 2023. 2 3

[24] Zhao Mandi, Shreeya Jain, and Shuran Song. RoCo: Dialectic multi-robot collaboration with large language models. arXiv preprint arXiv:2307.04738, 2023. 2 3

[25] Andrew G. Barto and Sridhar Mahadevan. Recent Advances in Hierarchical Reinforcement Learning. Discrete Event Dynamic Systems, 13:41-77, 2003. 2

[26] Ofir Nachum, Shixiang (Shane) Gu, Honglak Lee, and Sergey Levine. Data-Efficient Hierarchical Reinforcement Learning. In S. Bengio, H. Wallach, H. Larochelle, K. Grauman, N. Cesa-Bianchi, and R. Garnett, editors, Advances in Neural Information Processing Systems, volume 31. Curran Associates, Inc., 2018. 2

[27] Shubham Pateria, Budhitama Subagdja, Ah-hwee Tan, and Chai Quek. Hierarchical Reinforcement Learning: A Comprehensive Survey. ACM Comput. Surv., 54(5), jun 2021. 2

[28] Jan Wohlke, Felix Schmitt, and Herke van Hoof. Hierarchies of Planning and Reinforcement Learning for Robot Navigation. In 2021 IEEE International Conference on Robotics and Automation (ICRA). IEEE, May 2021. 2

[29] Hongyao Tang, Jianye Hao, Tangjie Lv, Yingfeng Chen, Zongzhang Zhang, Hangtian Jia, Chunxu Ren, Yan Zheng, Zhaopeng Meng, Changjie Fan, and Li Wang. Hierarchical Deep Multiagent Reinforcement Learning with Temporal Abstraction, 2019. 2

[30] Jiachen Yang, Igor Borovikov, and Hongyuan Zha. Hierarchical Cooperative Multi-Agent Reinforcement Learning with Skill Discovery. CoRR, abs/1912.03558, 2019. 2

[31] Jesse Meyer, Ryan Praeuner, and Luke Vanderbeek. Hierarchical Multi-Agent Reinforcement Learning. https://cse.unl.edu/~lksoh/Classes/CSCE475_875_Fall11/seminars/Seminar_JRL pdf 2020.2

[32] Adam Stooke, Kimin Lee, Pieter Abbeel, and Michael Laskin. Decoupling representation learning from reinforcement learning. In International Conference on Machine Learning, pages 9870-9879. PMLR, 2021. 2

[33] Ruihan Yang, Huazhe Xu, Yi Wu, and Xiaolong Wang. Multi-task reinforcement learning with soft modularization. Advances in Neural Information Processing Systems, 33:4767-4777, 2020. 2

[34] Sherry Yang, Ofir Nachum, Yilun Du, Jason Wei, Pieter Abbeel, and Dale Schuurmans. Foundation models for decision making: Problems, methods, and opportunities. arXiv preprint arXiv:2303.04129, 2023. 2

[35] Lei Wang, Chen Ma, Xueyang Feng, Zeyu Zhang, Hao Yang, Jingsen Zhang, Zhiyuan Chen, Jiakai Tang, Xu Chen, Yankai Lin, Wayne Xin Zhao, Zhewei Wei, and Ji-Rong Wen. A Survey on Large Language Model based Autonomous Agents, 2023. 2

[36] Zhiheng Xi, Wenxiang Chen, Xin Guo, Wei He, Yiwen Ding, Boyang Hong, Ming Zhang, Junzhe Wang, Senjie Jin, Enyu Zhou, Rui Zheng, Xiaoran Fan, Xiao Wang, Limao Xiong, Yuhao Zhou, Weiran Wang, Changhao Jiang, Yicheng Zou, Xiangyang Liu, Zhangyue Yin, Shihan Dou, Rongxiang Weng, Wensen Cheng, Qi Zhang, Wenjuan Qin, Yongyan Zheng, Xipeng Qiu, Xuanjing Huang, and Tao Gui. The Rise and Potential of Large Language Model Based Agents: A Survey, 2023. 2

[37] Theodore R. Sumers, Shunyu Yao, Karthik Narasimhan, and Thomas L. Griffiths. Cognitive Architectures for Language Agents, 2023. 2

[38] Pratyusha Sharma, Antonio Torralba, and Jacob Andreas. Skill Induction and Planning with Latent Language. CoRR, abs/2110.01517, 2021. 2

[39] Shreyas Sundara Raman, Vanya Cohen, Eric Rosen, Ifrah Idrees, David Paulius, and Stefanie Tellex. Planning with Large Language Models via Corrective Re-prompting. January 2022. 2

[40] Maitrey Gramopadhye and Daniel Szafir. Generating Executable Action Plans with Environmentally-Aware Language Models, 2023. 2

[41] Eric Kolve, Roozbeh Mottaghi, Winson Han, Eli VanderBilt, Luca Weihs, Alvaro Herrasti, Daniel Gordon, Yuke Zhu, Abhinav Gupta, and Ali Farhadi. AI2-THOR: An Interactive 3D Environment for Visual AI. arXiv, 2017. 246

[42] Manolis Savva, Abhishek Kadian, Oleksandr Maksymets, Yili Zhao, Erik Wijmans, Bhavana Jain, Julian Straub, Jia Liu, Vladlen Koltun, Jitendra Malik, Devi Parikh, and Dhruv Batra. Habitat: A Platform for Embodied AI Research. CoRR, abs/1904.01201, 2019. 2 4

[43] Fei Xia, Amir R. Zamir, Zhi-Yang He, Alexander Sax, Jitendra Malik, and Silvio Savarese. Gibson Env: Real-World Perception for Embodied Agents. CoRR, abs/1808.10654, 2018. 2

[44] Chengshu Li, Ruohan Zhang, Josiah Wong, Cem Gokmen, Sanjana Srivastava, Roberto Martín-Martín, Chen Wang, Gabrael Levine, Michael Lingelbach, Jiankai Sun, Mona Anvari, Minjune Hwang, Manasi Sharma, Arman Aydin, Dhruva Bansal, Samuel Hunter, Kyu-Young Kim, Alan Lou, Caleb R Matthews, Ivan Villa-Renteria, Jerry Huayang Tang, Claire Tang, Fei Xia, Silvio Savarese, Hyowon Gweon, Karen Liu, Jiajun Wu, and Li Fei-Fei. BEHAVIOR-1K: A Benchmark for Embodied AI with 1,000 Everyday Activities and Realistic Simulation. In Karen Liu, Dana Kulic, and Jeff Ichnowski, editors, Proceedings of The 6th Conference on Robot Learning, volume 205 of Proceedings of Machine Learning Research, pages 80-93. PMLR, 14-18 Dec 2023. 2

[45] Aishwarya Padmakumar, Jesse Thomason, Ayush Shrivastava, Patrick Lange, Anjali Narayan-Chen, Spandana Gella, Robinson Piramuthu, Gokhan Tur, and Dilek Hakkani-Tur. TEACh: Task-driven Embodied Agents that Chat, 2021. 2

[46] Mohit Shridhar, Jesse Thomason, Daniel Gordon, Yonatan Bisk, Winson Han, Roozbeh Mottaghi, Luke Zettlemoyer, and Dieter Fox. ALFRED: A Benchmark for Interpreting Grounded Instructions for Everyday Tasks. CoRR, abs/1912.01734, 2019. 2 10

[47] Dipendra Kumar Misra, Andrew Bennett, Valts Blukis, Eyvind Niklasson, Max Shatkhin, and Yoav Artzi. Mapping Instructions to Actions in 3D Environments with Visual Goal Prediction. CoRR, abs/1809.00786, 2018.2

[48] Yuke Zhu, Daniel Gordon, Eric Kolve, Dieter Fox, Li Fei-Fei, Abhinav Gupta, Roozbeh Mottaghi, and Ali Farhadi. Visual semantic planning using deep successor representations. In Proceedings of the IEEE international conference on computer vision, pages 483-492, 2017. 2

[49] Simon Brodeur, Ethan Perez, Ankesh Anand, Florian Golemo, Luca Celotti, Florian Strub, Jean Rouat, Hugo Larochelle, and Aaron C. Courville. HoME: a Household Multimodal Environment. CoRR, abs/1711.11017, 2017. 2

[50] Fanbo Xiang, Yuzhe Qin, Kaichun Mo, Yikuan Xia, Hao Zhu, Fangchen Liu, Minghua Liu, Hanxiao Jiang, Yifu Yuan, He Wang, Li Yi, Angel X. Chang, Leonidas J. Guibas, and Hao Su. SAPIEN: A Simulated Part-based Interactive ENvironment. CoRR, abs/2003.08515, 2020. 2

[51] Unnat Jain, Luca Weihs, Eric Kolve, Mohammad Rastegari, Svetlana Lazebnik, Ali Farhadi, Alexander G. Schwing, and Aniruddha Kembhavi. Two Body Problem: Collaborative Visual Task Completion. CoRR, abs/1904.05879, 2019. 2

[52] Unnat Jain, Luca Weihs, Eric Kolve, Ali Farhadi, Svetlana Lazebnik, Aniruddha Kembhavi, and Alexander Schwing. A cordial sync: Going beyond marginal policies for multi-agent embodied tasks. In Computer Vision-ECCV 2020: 16th European Conference, Glasgow, UK, August 23-28, 2020, Proceedings, Part V 16, pages 471-490. Springer, 2020. 2

[53] Chan Hee Song, Jiaman Wu, Clayton Washington, Brian M. Sadler, Wei-Lun Chao, and Yu Su. LLM-Planner: Few-Shot Grounded Planning for Embodied Agents with Large Language Models, 2023. 3

[54] Lin Xu, Zhiyuan Hu, Daquan Zhou, Hongyu Ren, Zhen Dong, Kurt Keutzer, See Kiong Ng, and Jiashi Feng. MAgIC: Benchmarking Large Language Model Powered Multi-Agent in Cognition, Adaptability, Rationality and Collaboration. arXiv preprint arXiv:2311.08562, 2023. 3

[55] Martin L. Puterman. Markov Decision Processes. Wiley, 1994. 3

[56] Leslie Pack Kaelbling, Michael L. Littman, and Anthony R. Cassandra. Planning and acting in partially observable stochastic domains. Artificial Intelligence, 101(1):99-134, 1998. 3

[57] Xavier Puig, Kevin Ra, Marko Boben, Jiaman Li, Tingwu Wang, Sanja Fidler, and Antonio Torralba. VirtualHome: Simulating Household Activities via Programs, 2018. 4

[58] Andrew Szot, Alex Clegg, Eric Undersander, Erik Wijmans, Yili Zhao, John Turner, Noah Maestre, Mustafa Mukadam, Devendra Chaplot, Oleksandr Maksymets, Aaron Gokaslan, Vladimir Vondruš, Sameer Dharur, Franziska Meier, Wojciech Galuba, Angel Chang, Zsolt Kira, Vladlen Koltun, Jitendra Malik, Manolis Savva, and Dhruv Batra. Habitat 2.0: Training Home Assistants to Rearrange their Habitat. In Advances in Neural Information Processing Systems (NeurIPS), 2021. 4

[59] Xavi Puig, Eric Undersander, Andrew Szot, Mikael Dallaire Cote, Ruslan Partsey, Jimmy Yang, Ruta Desai, Alexander William Clegg, Michal Hlavac, Tiffany Min, Theo Gervet, Vladimír Vondruš, Vincent-Pierre Berges, John Turner, Oleksandr Maksymets, Zsolt Kira, Mrinal Kalakrishnan, Jitendra Malik, Devendra Singh Chaplot, Unnat Jain, Dhruv Batra, Akshara Rai, and Roozbeh Mottaghi. Habitat 3.0: A Co-Habitat for Humans, Avatars and Robots, 2023. 4

[60] Chuang Gan, Siyuan Zhou, Jeremy Schwartz, Seth Alter, Abhishek Bhandwaldar, Dan Gutfreund, Daniel L. K. Yamins, James J DiCarlo, Josh McDermott, Antonio Torralba, and Joshua B. Tenenbaum. The Three-DWorld Transport Challenge: A Visually Guided Task-and-Motion Planning Benchmark for Physically Realistic Embodied AI, 2021. 4

[61] Dhruv Batra, Angel X. Chang, Sonia Chernova, Andrew J. Davison, Jia Deng, Vladlen Koltun, Sergey Levine, Jitendra Malik, Igor Mordatch, Roozbeh Mottaghi, Manolis Savva, and Hao Su. Rearrangement: A Challenge for Embodied AI. CoRR, abs/2011.01975, 2020. 4

[62] Archiki Prasad, Alexander Koller, Mareike Hartmann, Peter Clark, Ashish Sabharwal, Mohit Bansal, and Tushar Khot. ADaPT: As-Needed Decomposition and Planning with Language Models, 2023. 4

[63] Nils Reimers and Iryna Gurevych. Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks, 2019. 5

[64] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning Transferable Visual Models From Natural Language Supervision. CoRR, abs/2103.00020, 2021. 5

[65] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Ed H. Chi, Quoc Le, and Denny Zhou. Chain of Thought Prompting Elicits Reasoning in Large Language Models. CoRR, abs/2201.11903, 2022. 7

[66] Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. ReAct: Synergizing Reasoning and Acting in Language Models, 2023. 7

[67] OpenAI, :, Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, Red Avila, Igor Babuschkin, Suchir Balaji, Valerie Balcom, Paul Baltescu, Haiming Bao, Mo Bavarian, Jeff Belgum, Irwan Bello, Jake Berdine, Gabriel Bernadett-Shapiro, Christopher Berner, Lenny Bogdonoff, Oleg Boiko, Madelaine Boyd, Anna-Luisa Brakman, Greg Brockman, Tim Brooks, Miles Brundage, Kevin Button, Trevor Cai, Rosie Campbell, Andrew Cann, Brittany Carey, Chelsea Carlson, Rory Carmichael, Brooke Chan, Che Chang, Fotis Chantzis, Derek Chen, Sully Chen, Ruby Chen, Jason Chen, Mark Chen, Ben Chess, Chester Cho, Casey Chu, Hyung Won Chung, Dave Cummings, Jeremiah Currier, Yunxing Dai, Cory Decareaux, Thomas Degry, Noah Deutsch, Damien Deville, Arka Dhar, David Dohan, Steve Dowling, Sheila Dunning, Adrien Ecoffet, Atty Eleti, Tyna Eloundou, David Farhi, Liam Fedus, Niko Felix, Simón Posada Fishman, Juston Forte, Isabella Fulford, Leo Gao, Elie Georges, Christian Gibson, Vik Goel, Tarun Gogineni, Gabriel Goh, Rapha Gontijo-Lopes, Jonathan Gordon, Morgan Grafstein, Scott Gray, Ryan Greene, Joshua Gross, Shixiang Shane Gu, Yufei Guo, Chris Hallacy, Jesse Han, Jeff Harris, Yuchen He, Mike Heaton, Johannes Heidecke, Chris Hesse, Alan Hickey, Wade Hickey, Peter Hoeschele, Brandon Houghton, Kenny Hsu, Shengli Hu, Xin Hu, Joost Huizinga, Shantanu Jain, Shawn Jain, Joanne Jang, Angela Jiang, Roger Jiang, Haozhun Jin, Denny Jin, Shino Jomoto, Billie Jonn, Heewoo Jun, Tomer Kaftan, Łukasz Kaiser, Ali Kamali, Ingmar Kanitscheider, Nitish Shirish Keskar, Tabarak Khan, Logan Kilpatrick, Jong Wook Kim, Christina Kim, Yongjik Kim, Hendrik Kirchner, Jamie Kiros, Matt Knight, Daniel Kokotajlo, Łukasz Kondraciuk, Andrew Kondrich, Aris Konstantinidis, Kyle Kosic, Gretchen Krueger, Vishal Kuo, Michael Lampe, Ikai Lan, Teddy Lee, Jan Leike, Jade Leung, Daniel Levy, Chak Ming Li, Rachel Lim, Molly Lin, Stephanie Lin, Mateusz Litwin, Theresa Lopez, Ryan Lowe, Patricia Lue, Anna Makanju, Kim Malfacini, Sam Manning, Todor Markov, Yaniv Markovski, Bianca Martin, Katie Mayer, Andrew Mayne, Bob McGrew, Scott Mayer McKinney, Christine McLeavey, Paul McMillan, Jake McNeil, David Medina, Aalok Mehta, Jacob Menick, Luke Metz, Andrey Mishchenko, Pamela Mishkin, Vinnie Monaco, Evan Morikawa, Daniel Mossing, Tong Mu, Mira Murati, Oleg Murk, David Mély, Ashvin Nair, Reiichiro Nakano, Rajeev Nayak, Arvind Neelakantan, Richard Ngo, Hyeonwoo Noh, Long Ouyang, Cullen O'Keefe, Jakub Pachocki, Alex Paino, Joe Palermo, Ashley Pantuliano, Giambattista Parascandolo, Joel Parish, Emy Parparita, Alex Passos, Mikhail Pavlov, Andrew Peng, Adam Perelman, Filipe de Avila Belbute Peres, Michael Petrov, Henrique Ponde de Oliveira Pinto, Michael, Pokorny, Michelle Pokrass, Vitchyr Pong, Tolly Powell, Alethea Power, Boris Power, Elizabeth Proehl, Raul Puri, Alec Radford, Jack Rae, Aditya Ramesh, Cameron Raymond, Francis Real, Kendra Rimbach, Carl Ross, Bob Rotsted, Henri Roussez, Nick Ryder, Mario Saltarelli, Ted Sanders, Shibani Santurkar, Girish Sastry, Heather Schmidt, David Schnurr, John Schulman, Daniel Selsam, Kyla Sheppard, Toki Sherbakov, Jessica Shieh, Sarah Shoker, Pranav Shyam, Szymon Sidor, Eric Sigler, Maddie Simens, Jordan Sitkin, Katarina Slama, Ian Sohl, Benjamin Sokolowsky, Yang Song, Natalie Staudacher, Felipe Petroski Such, Natalie Summers, Ilya Sutskever, Jie Tang, Nikolas Tezak, Madeleine Thompson, Phil Tillet, Amin Tootoonchian, Elizabeth Tseng, Preston Tuggle, Nick Turley, Jerry Tworek, Juan Felipe Cerón Uribe, Andrea Vallone, Arun Vijayvergiya, Chelsea Voss, Carroll Wainwright, Justin Jay Wang, Alvin Wang, Ben Wang, Jonathan Ward, Jason Wei, CJ Weinmann, Akila Welihinda, Peter Welinder, Jiayi Weng, Lilian Weng, Matt Wiethoff, Dave Willner, Clemens Winter, Samuel Wolrich, Hannah Wong, Lauren Workman, Sherwin Wu, Jeff Wu, Michael Wu, Kai Xiao, Tao Xu, Sarah Yoo, Kevin Yu, Qiming Yuan, Wojciech Zaremba, Rowan Zellers, Chong Zhang, Marvin Zhang, Shengjia Zhao, Tianhao Zheng, Juntang Zhuang, William Zhuk, and Barret Zoph. GPT-4 Technical Report, 2023. 8

[68] Hugo Laurençon, Lucile Saulnier, Léo Tronchon, Stas Bekman, Amanpreet Singh, Anton Lozhkov, Thomas Wang, Siddharth Karamcheti, Alexander M. Rush, Douwe Kiela, Matthieu Cord, and Victor Sanh. OBELICS: An Open Web-Scale Filtered Dataset of Interleaved Image-Text Documents, 2023. 8 32

[69] Hugo Laurençon, Léo Tronchon, Matthieu Cord, and Victor Sanh. What matters when building vision-language models?, 2024. 8 32

[70] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual Instruction Tuning. In NeurIPS, 2023. 8 32

[71] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved Baselines with Visual Instruction Tuning, 2023. 8

[72] Wenyi Hong, Weihan Wang, Qingsong Lv, Jiazheng Xu, Wenmeng Yu, Junhui Ji, Yan Wang, Zihan Wang, Yuxuan Zhang, Juanzi Li, Bin Xu, Yuxiao Dong, Ming Ding, and Jie Tang. CogAgent: A Visual Language Model for GUI Agents, 2023. 8 32

[73] Kaya Stechly, Karthik Valmeekan, and Subbarao Kambhampati. Chain of Thoughtlessness: An Analysis of CoT in Planning. arXiv preprint arXiv:2405.04776, 2024. 8

[74] Yining Hong, Haoyu Zhen, Peihao Chen, Shuhong Zheng, Yilun Du, Zhenfang Chen, and Chuang Gan. 3D-LLM: Injecting the 3D World into Large Language Models, 2023. 10

[75] Subbarao Kambhampati. Can large language models reason and plan? Annals of the New York Academy of Sciences, 1534(1):15-18, 2024. 10

[76] Karthik Valmeekam, Matthew Marquez, Sarath Sreedharan, and Subbarao Kambhampati. On the planning abilities of large language models-a critical investigation. Advances in Neural Information Processing Systems, 36:75993-76005, 2023. 10

## A Terminology

We differentiate between the terms subtasks and high-level actions in this section. In essence, multiple high-level actions are needed to be carried out in a sequence to complete a subtask. Multiple subtasks need to be satisfied to complete the high-level language instruction.

- Subtasks: A task is split up into multiple subtasks. For example, if a task is "Fetch all the groceries and put them in the fridge", then the initial subtasks could include: "Locate the groceries", "transport the groceries", "Locate the fridge". These subtasks could get updated with new observations. For example, while locating the groceries, the agents come across a tomato and a lettuce. Then the subtasks "transport the tomato to the fridge" and "transport the lettuce to the fridge" gets updated in the subtasks list. This basically splits up the high-level instruction $\mathcal{I}$ into multiple mid-level subtasks

- High-level actions: These are the set of actions required to complete the subtasks. For example, to complete the "transport the lettuce in the fridge", we would require: the following set of actions:

- Navigate to lettuce

- Pickup lettuce

- Navigate to the fridge

- Open fridge

- Put lettuce in the fridge

- Close fridge

Note that different agents have to complete different high-level actions that progress the subtasks efficiently whilst avoiding conflicts.

- Conflicts can arise in the following ways:

- Same actions: Agents performing the same action at the same time. Example: "Open the fridge".

- Blocking: Agent 1 is blocking Agent 2 and not allowing it to complete its high-level action. Example: Agent 1 is attempting to execute "PlaceObject(Tomato)" in front of the fridge to place the tomato in its hand in the fridge and Agent 2 is attempting to execute "OpenFreezer()" needs to interact with the fridge. Would require some form of conflict resolution in the state cell domain. Agent 1 should move away to allow fridge access to Agent 2. In LLaMAR, the Corrector module helps in figuring out these conflicts and suggest different corrective high-level actions.

## B MAP-THOR Environment

The environment is based on the AI2Thor simulator with a multi-agent setup. All the experiments were performed in the single-room floor plans. When more than 3 agents are added to some of the floor plans (especially the kitchen floor plans), the environment gets crowded and does not allow for a lot of free space to navigate to different objects (the number of reachable paths reduces).

<img src="https://cdn.noedgeai.com/bo_d4nfsebef24c73bbfgrg_15.jpg?x=326&y=1575&w=1141&h=202&r=0"/>

Figure 3: Photorealistic rendering of household scenarios in the AI2Thor simulator enables the usage of multiple autonomous robots to carry out daily tasks.

### B.1 Observation Space

The observations for each robot include an image of size resolution ${1000} \times  {1000} \times  3$ . The textual observation for each agent in the prompt is the list of objects visible in this image and the agents' current location and rotation. The field of view is 90 degrees. The agents can interact with the objects only if it is within its visibility range of ${1.5}\mathrm{\;m}$ .

### B.2 Action Space

The actions space $\mathcal{A}$ consists of navigation actions ${\mathcal{A}}_{NAV}$ ,interaction actions ${\mathcal{A}}_{INT}$ ,exploration action ${\mathcal{A}}_{EXP}$ .

Navigation actions ${\mathcal{A}}_{NAV}$ consists of the following actions:

- Move(<direction>): Moves the robot by ${0.25}\mathrm{\;m}$ towards the specified direction where <direction> can be one of (Ahead, Back, Right, Left)

- Rotate(<direction>): Rotates the robot by 90 degrees towards the specified direction where, <direction> can be one of (Right, Left)

- LookUp(<angle>) rotates the pitch of the robot camera upwards by the specified angle.

- LookDown<angle> rotates the pitch of the robot camera downwards by the specified angle.

- NavigateTo(<object_id>) makes the robot navigate to the specified object. The path is found using the ${A}^{ * }$ -shortest path algorithm. Note that the robot is only able to find a path to the specified object in the environment only if it has encountered that object previously during the episode. Otherwise, the NavigateTo(.) action will be unsuccessful and the agent will have to explore.

Interaction actions ${\mathcal{A}}_{INT}$ consists of the following actions:

---

- Pickup(<object_id>): Picks up the object

- Put(<receptacle_id>): Puts the object in the robots hand on the receptacle

	- Open(<object_id>): Opens the object

	- Close(<object_id>): Closes the open object

	- Slice(<object_id>): Slices the object

	- Clean(<object_id>): Cleans the object

- ToggleOn(<object_id>): Toggles the object on

- ToggleOff (<object_id>): Toggles the object off

---

Explore action ${\mathcal{A}}_{EXP}$ : The explore action is carried out by the heuristic explained in Algorithm 1 and Figure 4 We use the clip-vit-large-patch14-336 model for the CLIP weights which we download from https://huggingface.co/openai/clip-vit-large-patch14-336

The action space consists of several executable actions, including Move(<direction>), Rotate(<direction>), LookUp(<angle>), LookDown(<angle>), NavigateTo(<object_id>), Pickup(<object_id>), Put(<receptacle_id>), and others. These actions can be combined in numerous ways, as the <object_id>, <directions>, and <angles> can vary significantly, resulting in a combinatorially large action space. To address this complexity, we do not impose constraints on the language model (LM) modules to select actions in a predefined format from this set. Instead, we allow for free-form language outputs, offering greater expressivity. The SentenceBERT module then maps these free-form natural language outputs to executable actions within the environment.

<img src="https://cdn.noedgeai.com/bo_d4nfsebef24c73bbfgrg_16.jpg?x=331&y=1680&w=1143&h=347&r=0"/>

Figure 4: Choice of direction for the exploration heuristic: The agent (Alice) rotates towards 4 cardinal directions to get observations. The cosine similarity between the CLIP embeddings ${I}_{d}$ for these 4 images are calculated with the CLIP embeddings for each subtask in the open subtasks set ${\mathcal{G}}_{O}$ to get the exploration score ${\mathcal{E}}_{d}$ for each direction. The direction with the highest ${\mathcal{E}}_{d}$ is chosen to explore and the agent moves $J = 2$ steps in that direction.

Algorithm 1 AI2Thor Exploration Heuristic

---

Input: Agent ID $n$ ,Environment env,number of exploration steps $K$ ,number of move steps $J$

	${g}_{O} = {\operatorname{CLIP}}_{\text{ text }}\left( {\mathcal{G}}_{O}\right)$

	while $k < K$ do

		Exploration Score $\mathcal{E} \in  {\mathbb{R}}^{4} \leftarrow  \mathbf{0}$

		for $d \in  \{$ North,South,East,West $\}$ do

			${o}_{n,d} =$ env.step(Rotate(Right,n))

			${I}_{d} = {\operatorname{CLIP}}_{\text{ img }}\left( {o}_{n,d}\right)$

			${\mathcal{E}}_{d} = \frac{{I}_{d} \cdot  {g}_{O}}{\begin{Vmatrix}{I}_{d}\end{Vmatrix}\begin{Vmatrix}{g}_{O}\end{Vmatrix}}$

		end for

		${d}^{ * } = \arg \mathop{\max }\limits_{d}\mathcal{E}$

		while $j < J$ do

			${o}_{i} = \operatorname{env.step}\left( {\operatorname{Move}\left( {{d}^{ * },n}\right) }\right)$

			$j \leftarrow  j + 1$

		end while

		$k \leftarrow  k + 1$

	end while

---

## C MAP-THOR Task Types

The complete list of task for each task type:

- Explicit object type, explicit quantity and target:

- put bread, lettuce, and a tomato in the fridge

- Put the pots and pans on the stove burners

- Slice the bread and tomato and crack the egg

- Put the butter knife, bowl, and mug in the sink

- Turn off the faucet and light if either is on

- Put the tissue box, keys, and plate in the box

- Put the computer, book, and pen on the couch

- Put the bowl and tissue box on the table

- Put apple in fridge and switch off the light

- Put the watch and Keychain inside the drawer

- Wash the bowl, mug, pot, and pan

- Put the Box on the sofa and the bowl in the box

- Explicit object type and explicit target: Here we explicitly describe the object type but keep the quantity of the objects ambiguous. E.g. Put all the apples in the fridge. For this, the agents have to explore the environment to ensure that they find all of them.

- Open all the drawers

- Open all the cabinets

- Turn on all the stove knobs

- Put all the vases on the table

- Put all the potatoes in the bowl

- Put all pencils and pens in the box

- Move all lamps next to the door

- Turn off all light switches

- Turn on all light switches

- Explicit target, implicit object types: The object types are implicitly defined whereas the target is explicitly defined. E.g. Put all groceries in the fridge. This tests whether the model can identify objects of certain categories.

- Put all groceries in the fridge (should identify the tomato, bread, apple, potato, and lettuce)

- Put all shakers in the closest drawer (should identify the salt shaker and pepper shaker)

- Put all tableware on the countertop (should identify the bowl, plate, mug)

- Put all food on the countertop (should identify the tomato, bread, apple, potato, and lettuce)

- Put all school supplies on the couch (should identify the pencil, computer, and book)

- Put all kitchenware in the cardboard box (should move the bowl and plate)

- Put all silverware in the sink

- Move everything on the table to the desk (should move the laptop, pencil, pen, plate, credit card, book, and newspaper)

- Slice the lettuce, trash the mug and switch off the light

- Put all electronics on the couch

- Make a dish by microwaving eggs and tomato

- Put all readable objects on the sofa

- Wash all fruits

- Implicit target and object types: Here both the object type and the target are implicitly defined. E.g. Clear the floor by placing the items at their appropriate positions. Here the model is expected to keep items like pens, book, laptop on the study table, litter in the trash can, etc.

- Clear the floor by placing items at their appropriate positions (depending on what's on the floor)

- Clear the table by placing the items in their appropriate positions (depends on the floorplan, e.g. bread, apple, tomato, knife, bowl, book)

- Clear the countertop by placing items in their appropriate positions (should move the lettuce, mug, and paper towel roll)

- Clear the desk by placing the items in other appropriate positions (should move the statue, watch, and remote control)

- Clear the table by placing the items in other appropriate positions (should move the book, credit card, laptop, plate, newspaper, pen, and pencil)

- Clear the couch by placing the items in other appropriate positions (should move the pillow) - Make the living room dark

- Make a mug of coffee and toast the bread

- Trash all groceries

- Slice all sliceable objects

## D Search & Rescue Environment (SAR)

The Search & Rescue environment consists of multiple agents in an unknown environment that has multiple wildfires and missing personnel in the environment. The agents are tasked to extinguish all the fires before they spread and rescue the missing humans. Here, each fire is composed of a large flammable region with a fixed set of sources that spread through time. The higher the intensity, the faster the fire will spread. The fires can be of class A or B, extinguished through the use of water and sand respectively. Both these resources can be collected through resource reservoirs spread geographically. Each person is initially stranded in an unknown location. The goal is to rescue and transport each person to a drop-off location (known apriori). The person must have two agents simultaneously carrying them to be transported to the drop-off location.

<img src="https://cdn.noedgeai.com/bo_d4nfsebef24c73bbfgrg_18.jpg?x=564&y=1541&w=658&h=390&r=0"/>

Figure 5: The search & rescue environment consists of multiple drones in an unknown environment with missing people, fires of different types, and water and sand reservoirs.

We provide a justification to why this task is relevant to long-term planning and multi-agent collaboration below.

### D.1 Multi-agent Nature

- Scalability - Unlike the AI2THOR environment, the search & rescue environment is more spacious and hence prevents congestion amongst agents. With the addition of multiple parallel emergency disaster scenarios at once, the environment scales comfortably to more agents. Additionally, the complexity slowly increases as the model has to coordinate between more actors and competing emergencies.

- Tasks requires explicit cooperation -

- To move a group of humans, at least 2 agents are required. Thus, explicit multi-agent collaboration is required.

- Due to the time pressure, in order to successfully stop the fire, agents must all effectively collaborate in collecting and strategically using resources in high-intensity regions.

- Exploration & Task Assignment - There is an explicit tradeoff between allocating agents towards exploring to find lost people and fighting the current fires.

### D.2 Long-term Planning

- Dependency Chain: To resolve a fire it requires that the source is identified, the type of fire is identified, and the appropriate resources to stop it are acquired and then used.

- Uncertainty: Inherent uncertainty as to where the lost people can be found, and at what point in the timeline they will.

- Irreversibility and forced consequences - With a fire that spreads, present actions have irreversible future consequences. A balance needs to be struck between fighting the fire's source (to stop it from continuing) versus a periphery (to prevent geographic spread).

### D.3 Description of Scenes

We evaluate LLaMAR in five different scenarios in the search & rescue environment. Here each scene is evaluated on 5 different random seeds.

- Scene 1: This consists of 1 Type A fire with 2 initial sources, 1 Type B fire with 1 initial source, and 1 lost person at a random location in the environment.

- Scene 2: This consists of 1 Type A fire with 1 initial source, 1 Type B fire with 2 initial sources, and 1 lost person at a random location in the environment.

- Scene 3: This consists of 2 Type A fires each with 1 initial source, 1 Type B fire with 1 initial source, and 1 lost person at a random location in the environment.

- Scene 4: This consists of 1 Type A fire with 3 initial sources, and 1 lost person at a random location in the environment.

- Scene 5: This consists of 1 Type A fire with 1 initial source, 1 Type B fire with 1 initial source, and 2 lost persons at random locations in the environment.

### D.4 Observation Space

The agent’s observation $\mathcal{O} = {\mathcal{O}}_{G} \cup  {\mathcal{O}}_{L} \cup  {\mathcal{O}}_{L}$ is a union of the following observations.

Global Observations ${\mathcal{O}}_{G}$ consists of all the objects that are globally visible by the agent:

- Fires: If visible, fire name, type, and average intensity.

- Fire Regions: If fire visible, fire region name, type, and average intensity.

- Reservoir: If visible, reservoir name, and reservoir resource type.

- Deposit: If visible, deposit name, inventory of all the resources (water, sand) and persons in deposit.

- Person: If visible, person name, carried or not status, dropped-off or not status.

- Agent inventory: List of all the resources (water, sand) and person (being carried) in the agent's inventory.

Local Observations ${\mathcal{O}}_{L}$ consists of all the objects that are visible in grid cells adjacent to the agent:

- For all the directions <direction> in (Up, Down, Left, Right, and Center), output of the following.

- <direction>: At direction <direction>, either 'Empty' if there is no object, 'Flammable' along with corresponding intensity & fire name if object is a part of a fire, or 'Obstacle' for any other object.

Names ${\mathcal{O}}_{N}$ consists of a list of the name of all visible and interactable objects.

### D.5 Action Space

The action space $\mathcal{A}$ consists of navigation actions ${\mathcal{A}}_{NAV}$ ,interaction actions ${\mathcal{A}}_{INT}$ ,exploration action ${\mathcal{A}}_{EXP}$ .

Navigation actions ${\mathcal{A}}_{NAV}$ consists of the following actions:

- Move (<direction>): Moves the agent to the neighboring grid cell in the specified direction where <direction> can be one of (Up, Down, Left, Right, and Center)

- NavigateTo(<targetID>): Moves the agent next to the location of the object <targetID> if <targetID> is visible.

Interaction actions ${\mathcal{A}}_{INT}$ consists of the following actions:

- Carry (<person>): Makes agent carry a person <person> if <person> is visible and inter-actable. The person is successfully 'group' carried, if at least the required number of agents successfully does the 'Carry (<person>)' action. If carry action is successful, all other resources in agent's inventory are dropped.

- DropOff (<person>, <deposit>): Drops off person <person> at location <deposit>. This action is only successful if the person has been 'group' carried, all the agents carrying the person have the deposit be visible and interactable, and after all the agents do this action.

- StoreSupply(<deposit>): Stores all the resources from the agent's current inventory in the deposit <deposit>.

- UseSupply(<fire>, <supply-type>): Uses all the supplies of type <supply-type> on the fire <fire> at the location the agent is at.

- GetSupply(<deposit>, <supply-type>): Fills the remaining of the agent's inventory space with the available supply of type <supply-type> in <deposit>.

- GetSupply(<reservoir>): Collects 1 unit of supply from reservoir <reservoir> and stores it in the agent's inventory.

Exploration action ${\mathcal{A}}_{EXP}$ consists of the following actions:

- Explore(): Takes the agent in a direction of exploration as described by the heuristic exploration function described in algorithm 2.

Algorithm 2 SAR Exploration Heuristic

---

Input: Agent A, Environment env, Previous Direction D, Current Position P, Max steps M

Initialize: New direction ${ND} \leftarrow  \varnothing$

	${ND} \leftarrow$ randomly choose an angle from $\lbrack 0,{2\pi })$ at $\pi /4$ intervals

	while ${ND} = D$ or ${ND} \equiv  D + \pi {\;\operatorname{mod}\;2}\pi$

			or a barrier exists at most $\frac{3}{4}M$ steps in direction ${ND}$ from $P$ do

		${ND} \leftarrow$ randomly choose an angle from $\lbrack 0,{2\pi })$ at $\pi /4$ intervals

	end while

	Move agent $A,M$ steps in direction ${ND}$ in environment env

---

## E Pseudocode for LLaMAR

Algorithm 3 LLaMAR

---

Input: $N$ agents,Task instruction $\mathcal{I}$ ,Environment env

Initialize: Memory $\mathcal{M} \leftarrow  \varnothing$ ; Open Subtasks ${\mathcal{G}}_{O} \leftarrow  \varnothing$ ;

Completed Subtasks ${\mathcal{G}}_{C} \leftarrow  \varnothing$ ; Actions $a \leftarrow  \varnothing$ ;

Corrective Actions ${a}_{c} \leftarrow  \varnothing$

Actions Executed $d \leftarrow  \varnothing$

	$o = \left( {{o}_{1},\cdots ,{o}_{N}}\right)  =$ env.reset $\left( \right)$

	while $t < T$ do

		${\mathcal{G}}_{O} \leftarrow  \operatorname{Planner}\left( {\mathcal{I},o,{\mathcal{G}}_{O},{\mathcal{G}}_{C},\mathcal{M}}\right)$

		$a,\mathcal{M} \leftarrow  \operatorname{Actor}\left( {\mathcal{I},o,{a}_{c},{\mathcal{G}}_{O},{\mathcal{G}}_{C},\mathcal{M}}\right)$

		$o = \left( {{o}_{1},\cdots ,{o}_{N}}\right) ,d = \left( {{d}_{1},\cdots ,{d}_{N}}\right)  =$ env.step $\left( a\right)$

		${a}_{c} \leftarrow  \operatorname{Corrector}\left( {\mathcal{I},o,a,d,{\mathcal{G}}_{O},{\mathcal{G}}_{C},\mathcal{M}}\right)$

		${\mathcal{G}}_{C} \leftarrow  \operatorname{Verifier}\left( {\mathcal{I},o,a,d,{\mathcal{G}}_{O},{\mathcal{G}}_{C},\mathcal{M}}\right)$

		if ${\mathcal{G}}_{O} = \varnothing$ then

			break

		end if

		$t \leftarrow  t + 1$

	end while

---

## F Full Results

<table><tr><td>Algorithm</td><td>LM</td><td>Success Rate</td><td>Transport Rate</td><td>Coverage</td><td>Balance</td><td>Steps</td></tr><tr><td>Act</td><td>GPT-4V</td><td>0.33 (0.19, 0.49)</td><td>0.67 (0.59, 0.76)</td><td>0.91 (0.86,0.95)</td><td>0.59 (0.52, 0.66)</td><td>24.92 (22.12,27.73)</td></tr><tr><td>ReAct</td><td>GPT-4V</td><td>0.34 (0.20, 0.49)</td><td>0.72 (0.63,0.80)</td><td>0.92 (0.86, 0.97)</td><td>0.67 (0.61, 0.73)</td><td>24.08 (21.27, 26.89)</td></tr><tr><td>CoT</td><td>GPT-4V</td><td>0.14 (0.06, 0.28)</td><td>0.59 (0.51, 0.67)</td><td>0.87 (0.81, 0.92)</td><td>0.62 (0.56,0.69)</td><td>28.40 (26.91, 29.97)</td></tr><tr><td>SmartLLM</td><td>GPT-4V</td><td>0.11 (0.05, 0.23)</td><td>0.23 (0.13, 0.31)</td><td>0.91 (0.80, 0.96)</td><td>0.45 (0.37, 0.52)</td><td>29.87 (26.20, 30.00)</td></tr><tr><td>CoELA</td><td>GPT-4V</td><td>0.25 (0.10, 0.36)</td><td>0.46 (0.35, 0.56)</td><td>0.76 (0.67, 0.85)</td><td>0.73 (0.67, 0.80)</td><td>28.93 (27.77,30.00)</td></tr><tr><td>LLaMAR</td><td>GPT-4</td><td>0.51 (0.36, 0.66)</td><td>0.85 (0.80, 0.91)</td><td>0.95 (0.91, 0.98)</td><td>0.83 (0.78, 0.86)</td><td>25.80 (23.72,27.88)</td></tr><tr><td>LLaMAR</td><td>LLaVA</td><td>0.54 (0.41, 0.65)</td><td>0.84 (0.71, 0.90)</td><td>0.91 (0.87, 0.98)</td><td>0.75 (0.64, 0.83)</td><td>26.21 (21.56, 28.97)</td></tr><tr><td>LLaMAR</td><td>IDEFICS-2</td><td>0.57 (0.43, 0.67)</td><td>0.86 (0.74, 0.91)</td><td>0.94 $\left( {{0.89},{0.98}}\right)$</td><td>0.78 (0.65, 0.84)</td><td>25.27 (20.14, 28.37)</td></tr><tr><td>LLaMAR</td><td>CogVLM</td><td>0.61 (0.47, 0.68)</td><td>0.89 (0.73, 0.95)</td><td>0.95 (0.89, 0.99)</td><td>0.80 (0.73, 0.86)</td><td>23.21 (20.57, 26.82)</td></tr><tr><td>LLaMAR (w/o exploration)</td><td>GPT-4V</td><td>0.62 (0.46, 0.76)</td><td>0.87 (0.80, 0.93)</td><td>0.95 (0.91, 0.98)</td><td>0.82 (0.77, 0.87)</td><td>23.44 (20.88, 26.00)</td></tr><tr><td>LLaMAR (w/ exploration)</td><td>GPT-4V</td><td>0.66 (0.50, 0.78)</td><td>0.91 (0.81,0.96)</td><td>0.97 (0.93,0.99)</td><td>0.82 (0.75,0.87)</td><td>21.87 (18.76,24.23)</td></tr></table>

Table 5: Comparison of evaluation metrics against baselines averaged across all tasks for the 2-agent MAP-THOR scenarios.

<table><tr><td>Modules Used</td><td>Success Rate</td><td>Transport Rate</td><td>Coverage</td><td>Balance</td><td>Steps</td></tr><tr><td>Actor</td><td>0.33 (0.19, 0.49)</td><td>0.67 (0.59, 0.76)</td><td>0.91 (0.86,0.95)</td><td>0.59 (0.52,0.66)</td><td>24.92 <br> (22.12,27.73)</td></tr><tr><td>Planner + <br> Actor + Verifier</td><td>0.45 <br> (0.29, 0.57)</td><td>0.78 <br> (0.67, 0.84)</td><td>0.92 <br> (0.84, 0.95)</td><td>0.69 <br> (0.61, 0.75)</td><td>24.87 <br> (20.48, 27.95)</td></tr><tr><td>Planner + Actor + Corrector‡</td><td>0.67 <br> (0.51, 0.80)</td><td>0.91 <br> (0.83, 0.96)</td><td>0.97 <br> (0.94, 0.99)</td><td>0.84 <br> (0.79, 0.89)</td><td>22.81 <br> (19.95, 25.76)</td></tr><tr><td>LLaMAR</td><td>0.66 <br> (0.50, 0.76)</td><td>0.91 <br> (0.81, 0.96)</td><td>0.97 <br> (0.93,0.99)</td><td>0.82 <br> (0.75, 0.87)</td><td>21.87 <br> (18.76, 26.43)</td></tr></table>

Table 6: Ablating different modules LLaMAR with GPT-4V as the underlying VLM, 2-agents scenarios.

<table><tr><td rowspan="2">#of agents</td><td colspan="5">MAP-THOR</td><td colspan="5">SAR</td></tr><tr><td>Success Rate</td><td>Transport Rate</td><td>Coverage</td><td>Balance</td><td>Steps</td><td>Success Rate</td><td>Transport Rate</td><td>Coverage</td><td>Balance</td><td>Steps</td></tr><tr><td>1</td><td>0.37 <br> (0.21, 0.51)</td><td>0.67 <br> (0.58, 0.74)</td><td>0.87 <br> (0.81, 0.90)</td><td>1.00 (1.00,1.00)</td><td>28.44 <br> (25.23, 30.00)</td><td>0.28</td><td>0.75</td><td>0.86</td><td>1.00</td><td>28</td></tr><tr><td>2</td><td>0.62 <br> (0.46, 0.76)</td><td>0.87 <br> (0.80, 0.93)</td><td>0.95 <br> (0.91, 0.98)</td><td>0.82 <br> (0.77,0.87)</td><td>23.44 <br> (20.88, 26.00)</td><td>0.44 <br> (0.24,0.65)</td><td>0.86 <br> (0.79, 0.94)</td><td>0.94 <br> (0.88, 0.99)</td><td>0.91 <br> (0.88, 0.95)</td><td>27.76 <br> (24.15, 30)</td></tr><tr><td>3</td><td>0.70 <br> (0.55, 0.82)</td><td>0.91 <br> (0.85, 0.95)</td><td>0.98 <br> (0.95, 0.99)</td><td>0.66 <br> (0.61, 0.71)</td><td>21.30 <br> (18.60, 23.99)</td><td>0.68 <br> (0.46, 0.85)</td><td>0.92 <br> (0.86, 0.98)</td><td>0.96 <br> (0.91,1.0)</td><td>0.80 <br> (0.73, 0.86)</td><td>21.88 <br> (17.83, 25.92)</td></tr><tr><td>4</td><td>0.68 <br> (0.52, 0.79)</td><td>0.90 <br> (0.84, 0.94)</td><td>0.99 <br> (0.95, 0.99)</td><td>0.62 <br> (0.57,0.68)</td><td>22.83 <br> (19.63, 25.69)</td><td>0.72 <br> (0.50, 0.85)</td><td>0.94 <br> (0.88,0.98)</td><td>0.98 <br> (0.93,1.00)</td><td>0.78 <br> (0.74, 0.83)</td><td>22.00 <br> (17.96, 26.03)</td></tr><tr><td>5</td><td>0.62 <br> (0.46, 0.75)</td><td>0.90 <br> (0.85, 0.94)</td><td>0.99 <br> (0.97,1.00)</td><td>0.54 <br> (0.48, 0.59)</td><td>22.91 <br> (20.26, 25.57)</td><td>0.74 <br> (0.52, 0.86)</td><td>0.96 <br> (0.94, 0.99)</td><td>1.00 <br> (1.0,1.0)</td><td>0.73 <br> (0.67, 0.79)</td><td>24.52 <br> (20.24,28.79)</td></tr></table>

Table 7: LLaMAR with more agents

## G Failure Cases

One of the main motivations of our paper was to achieve better success rates in planning for a variety of tasks compared to previous approaches. While our method outperforms other baselines, we acknowledge that the success rate is still under the expectation for real-world deployment. We believe that our approach LLaMAR and the MAP-THOR benchmark would serve as a starting point for future research to improve the success rates in multi-agent embodied robotics tasks. Here we describe a few major types of failures in our experiments:

- Mis-generalization: The agent can sometime fail to properly infer the level of abstraction to perform a task at, even if it should be clear from the action space. For example, for the washing tasks, the LM assumes that it must put the objects in the sink and add some, rather than just using the "CleanObject" action. We observe this error in the direction of performing actions more low-level than necessary, and not the other way around (more high-level than is feasible).

- Mutual Interference (limited spatial reasoning): The agents sometimes block each other and thus fail to carry out actions such as placing an object on a receptacle, opening an object, etc. In particular, we see this behavior in the "put objects on the sofa" and the "put objects in a box" tasks, where the LM does not prevent the agents from blocking each other.

- Improper Object Visibility (un-observability bias): The LM often fails to prioritize exploration for tasks that require objects not yet seen, simply due to it no being able to see it. This bias causes it to improperly assume that further exploration is not necessary, so it fails to find relevant objects.

- SentenceBERT mismatch: The free-form natural language output from the Actor is incorrectly mapped to the executable action. Based on our experimental results, 96.7% of the time the free-form natural language was correctly mapped to the feasible actions. The 2 main reasons for failures in the few cases it failed were:

- Incorrect mapping of objects: When the agents have not explored the environment enough and the Actor module suggests the agents interact with objects that are not yet available in the agents' combined memory. Example: The free-form output "navigate to the table" was mapped to the action "NavigateTo(ArmChair)".

- Wrong object numeration: When there are multiple objects of the same type in the memory, the sentenceBERT maps "Cabinet_3" to "Cabinet_1" which sometimes leads to incorrect actions being executed for the completion of the plan.

- Limited horizon: While each task is solvable by a single-agent human-controlled agent within our chosen episode length of $L = {30}$ ,this is under the assumption of no failures during control and maximum visibility. For the autonomous agents the visibility distance is restricted to 1.5 meters, so while a human could identify objects from afar (from the rendered images) the agents have to explore the environment under a stricter partial observability constraint.

We hypothesize that a higher episode length would lead to higher metrics, but the cutoff was chosen for computational budget considerations. As can be observed in figure 6 , the coverage and transportation rate metrics in particular seem to plateau around our chosen episode length $L$ ,so $L = {30}$ is appropriate.

<img src="https://cdn.noedgeai.com/bo_d4nfsebef24c73bbfgrg_23.jpg?x=354&y=215&w=1088&h=692&r=0"/>

Figure 6: Plots of the average transport rate (a),(d); coverage (b),(e); and success rate (c),(f) metrics vs. maximum horizon steps for the GPT-4V, LLaMAR algorithm on MAP-THOR. Shown for both the 2 agent (a),(b),(c); and 5 agent cases (d),(e),(f), along with an error region of one standard deviation.

### G.1 Failure Cases

We have observed the following categories of failures for the LM on the SAR environment.

- Incorrect Causal Ordering: The LM sometimes fails to execute the steps in the proper causal ordering. For instance, we have observed the agent trying to drop off a lost person before picking them up, or properly getting supplies to stop the fire but not navigating to the fire location before using them.

- Sub-Optimal Action Sequence: Due to the time-constraints of the fire spread, it is not sufficient for the LM to do the sequence of tasks in order. They must also be done in a non-redundant, timely fashion. Otherwise, we have observed that the LM takes the right action sequence, but does not do it efficiently enough to stop the fire spread. For instance, if the agent prioritizes exploration to find the person, then the fire might spread until beyond a critical point where it cannot be extinguished within the horizon.

- Catastrophic Failure Misunderstanding: Even after the LM has completed the task, it can misunderstand a later failure as a cause for continuing the task. For instance, we have observed that if the LM has successfully dropped off a person, but then fails to pick them up again (which is expected since they're already dropped), it will think the task is incomplete.

## H Baselines

While there are a lot of impressive LLM-based multi-agent planners as mentioned in Table 1 they vary in the assumptions about access to information about the environment. We were not able to find the official codebase for the Safe Multi-Agent Planning with Conformal Prediction [20] and TwoStep [21]. We describe the prompts used for our model as well as every baseline. Note that we show the prompt for the 2-agent case,but it is easily modified to generalize to the $n$ -agent case. The italics and bolding added for emphasis.

### H.1 LLaMAR

We describe the prompts used for each of the modules used in LLaMAR:

## Prompt for Planner Module in LLaMAR

You are an excellent planner who is tasked with helping 2 embodied robots named Alice and Bob carry out a task. Both robots have a partially observable view of the environment. Hence they have to explore around in the environment to do the task.

You will get a description of the task robots are supposed to do. You will get an image of the environment from Alice's perspective and Bob's perspective as the observation input. To help you with detecting objects in the image, you will also get a list objects each agent is able to see in the environment. Here the objects are named as "<object_name>_<object_id>".

So, along with the image inputs you will get the following information:

## ###INPUT FORMAT ###

\{Task: description of the task the robots are supposed to do,

Alice's observation: list of objects the Alice is observing,

Bob's observation: list of objects the Bob is observing,

Robots' open subtasks: list of subtasks the robots are supposed to carry out to finish the task. If no plan has been already created, this will be None.

Robots' completed subtasks: list of subtasks the robots have already completed. If no subtasks have been completed, this will be None.

Robots' combined memory: description of robots' combined memory\}

Reason over the robots' task, image inputs, observations, open subtasks, completed subtasks and memory, and then output the following:

* Reason: The reason for why new subtasks need to be added.

* Subtasks: A list of open subtasks the robots are supposed to take to complete the task. Remember, as you get new information about the environment, you can modify this list. You can keep the same plan if you think it is still valid. Do not include the subtasks that have already been completed.

The "Plan" should be in a list format where the subtask are listed sequentially.

For example:

["locate the apple", "transport the apple to the fridge", "transport the book to the table"]

["locate the cup", "go to cup", "clean cup"]

When possible do not perform additional steps when one is sufficient (e.g. CleanObject is sufficient to clean an object, no other actions need to be done) Your output should be in the form of a python dictionary as shown below.

## Example output:

\{"reason": "Since the subtask list is empty, the robots need to transport the apple to the fridge and transport the book to the table.",

"plan": ["transport the apple to the fridge", "transport the book to the table"]\}

Ensure that the subtasks are not generic statements like "do the task". They should be specific to the task at hand.

Do not assign subtasks to any particular robot. Try not to modify the subtasks that already exist in the open subtasks list. Rather add new subtasks to the list.

* NOTE: DO NOT OUTPUT ANYTHING EXTRA OTHER THAN WHAT HAS BEEN SPECIFIED

Let's work this out in a step by step way to be sure we have the right answer.

## Prompt for Verifier Module in LLaMAR

You are an excellent task verifier who is tasked with helping 2 embodied robots named Alice and Bob carry out a task. Both robots have a partially observable view of the environment. Hence they have to explore around in the environment to do the task.

You will get a description of the task robots are supposed to do. You will get an image of the environment from Alice's perspective and Bob's perspective as the observation input. To help you with detecting objects in the image, you will also get a list objects each agent is able to see in the environment. Here the objects are named as "<object_name>_<object_id>".

So, along with the image inputs you will get the following information:

## ###INPUT FORMAT ###

\{Task: description of the task the robots are supposed to do,

Alice's observation: list of objects the Alice is observing,

Alice's state: description of Alice's state,

Alice's previous action: the action Alice took in the previous step and if it was successful,

Bob's observation: list of objects the Bob is observing,

Bob's state: description of Bob's state, Bob's previous action: the action Bob took in the previous step, Robots' open subtasks: list of open subtasks the robots in the previous step. If no plan has been already created, this will be None.

Robots' completed subtasks: list of subtasks the robots have already completed. If no subtasks have been completed, this will be None.

Robots' combined memory: description of robots' combined memory\}

Reason over the robots' task, image inputs, observations, previous actions, open subtasks, completed subtasks and memory, and then output the following:

* Reason: The reason for why you think a particular subtask should be moved from the open subtasks list to the completed subtasks list.

* Completed Subtasks: The list of subtasks that have been completed by the robots. Note that you can add subtasks to this list only if they have been successfully completed and were in the open subtask list. If no subtasks have been completed at the current step, return an empty list.

The "Completed Subtasks" should be in a list format where the completed subtasks are listed.

For example: ["locate the apple", "transport the apple to the fridge", "transport the book to the table"]

Your output should be in the form of a python dictionary as shown below.

## Example output:

\{

"reason": "Alice placed the apple in the fridge in the previous step and was successful and Bob picked up the the book from the table. Hence Alice has completed the subtask of transporting the apple to the fridge, Bob has picked up the book, but Bob has still not completed the subtask of transporting the book to the table",

"completed subtasks": ["picked up book from the table", "transport the apple to the fridge"]

\}

* NOTE: DO NOT OUTPUT ANYTHING EXTRA OTHER THAN WHAT HAS BEEN SPECIFIED

When you output the completed subtasks, make sure to not forget to include the previous ones in addition to the new ones.

Let's work this out in a step by step way to be sure we have the right answer.

## Prompt for the Actor Module in LLaMAR

You are an excellent planner and robot controller who is tasked with helping 2 embodied robots named Alice, and Bob carry out a task. All 2 robots have a partially observable view of the environment. Hence they have to explore around in the environment to do the task.

They can perform the following actions:

["navigate to object <object_id>", "rotate in <rotation> direction", "pick up object <object_id>", "put object on <receptacle_id>", "open object <object_id>", "close object <object_id>", "slice object <object_id>", "toggle object <object_id> on", "toggle object <object_id> off", "clean object <object_id>", "look up by angle <angle>", "look down by angle <angle>", "move in <translation> direction", "stay idle", "Done"]

Here "Done" is used when all the robots have completed the main task. Only use it when you think all the subtasks are complete.

"stay idle" is used when you want the robot to stay idle for a one-time step. This could be used to wait for the other robot to complete its subtask. Use it only when you think it is necessary.

Here <rotation> can be one of ["Right", "Left"].

Here <angle> is the angle in degrees and can only be one of $\left\lbrack  {{30},{60},{90},{120},{150},{180}}\right\rbrack$ .

Here <translation> can be one of ["Ahead", "Back", "Left", "Right"].

So, along with the image inputs you will get the following information:

## ###INPUT FORMAT ###

\{Task: description of the task the robots are supposed to do,

Alice's observation: list of objects the Alice is observing,

Alice's state: description of Alice's state,

Alice's previous action: description of what Alice did in the previous time step and whether it was successful,

Alice's previous failures: if Alice's few previous actions failed,

description of what failed,

Bob's observation: list of objects the Bob is observing,

Bob's state: description of Bob's state,

Bob's previous action: description of what Bob did in the previous time step and whether it was successful,

Bob's previous failures: if Bob's few previous actions failed, description of what failed,

Robots' open subtasks: list of subtasks supposed to carry out to finish the task. If no plan has been already created, this will be None.

Robots' completed subtasks: list of subtasks the robots have already completed. If no subtasks have been completed, this will be None.

Robots' subtask: description of the subtasks the robots were trying to complete in the previous step, Robots' combined memory: description of robot's combined memory\}

## ###OUTPUT FORMAT ###

First of all you are supposed to reason over the image inputs, the robots' observations, previous actions, previous failures, previous memory, subtasks and the available actions the robots can perform, and think step by step and then output the following things:

* Failure reason: If any robot's previous action failed, use the previous history, your current knowledge of the room (i.e. what things are where), and your understanding of causality to think and rationalize about why the previous action failed. Output the reason for failure and how to fix this in the next timestep. If the previous action was successful, output "None".

Common failure reasons to lookout for include: one agent blocking another so must move out of the way, agent can't see an object or its destination and must explore (such as move, rotate, or look in a different direction) to find it, agent doing extraneous actions (such as drying objects when cleaning), etc. If the previous action was successful, output "None".

* Memory: Whatever important information about the scene you think you should remember for the future as a memory. Remember that this memory will be used in future steps to carry out the task. So, you should not include information that is not relevant to the task. You can also include information that is already present in its memory if you think it might be useful in the future.

## (CONTINUED) Prompt for the Actor Module in LLaMAR

* Reason: The reasoning for what each robot is supposed to do next

* Subtask: The subtask each robot should currently try to solve, choose this from the list of open subtasks.

* Action: The actions the robots are supposed to take just in the next step such that they make progress towards completing the task. Make sure that these suggested actions make these robots more efficient in completing the task as compared to only one agent solving the task.

Your output should just be in the form of a python dictionary as shown below.

## Examples of output:

## Example 1:

\{ "failure reason": "Bob failed to put the mug in the cabinet earlier because Alice was blocking it when she was putting the knife. To fix this, Alice should close the cabinet and move away, Charlie should move away to a different open area than Alice to avoid congestion, and Bob should wait until the next timestep until Alice can move aside.",

"memory": "Alice finished putting the knife in the cabinet when Alice was at co-ordinates (1, .5) and was facing north. Bob wanted to put the mug in the cabinet when Bob was at co-ordinates (1, 0.25) and was facing north.",

"reason": "Alice can close the cabinet door and then later back out in order help Bob with completing the task. Bob can be idle until the next timestep when Alice moves aside, by then Bob can navigate to the cabinet.",

"subtask": "Alice is currently closing the cabinet door, Bob is currently waiting to get to navigate to the cabinet",

"Alice's action" : "close the Cabinet_1",

"Bob's action": "stay idle"

\}

## Example 2: \{

"failure reason": "Bob failed to clean the cup earlier because Bob had not navigated to it, Bob assumed the cup to be in the sink which was erroneous. To fix this, Bob should navigate to the cup and in the next step clean cup.",

"memory": "Alice finished navigating to the dish when Alice was at co-ordinates (-.5, .5) and was facing east. Bob was not able to clean the cup in the cabinet when Bob was at co-ordinates (1, .25) and was facing north.",

"reason": "Alice can now clean the dish since Alice has navigated to it. Bob should navigate to the cup in order to be close enough to clean the cup.",

"subtask": "Alice is currently trying to clean the dish, Bob is currently trying to navigate to the cup", "Alice's action" : "clean the dish object",

"Bob's action": "navigate to the cup"

Note that the output should just be a dictionary similar to the example outputs.

## ###Important Notes ###

* The robots can hold only one object at a time.

For example: If Alice is holding an apple, she cannot pick up another object until she puts the apple down.

* Even if the robot can see objects, it might not be able to interact with them if they are too far away. Hence you will need to make the robot navigate closer to the objects they want to interact with.

For example: An action like "pick up <object_id>" is feasible only if robot can see the object and is close enough to it. So you will have to navigate closer to it before you can pick it up.

* In some scenarios, the agents might not see the objects that they want to interact with. In such cases, you will have to make the robot explore the environment to find the object. In such scenarios you can use actions to rotate in place or look up / down or navigate to explore the environment.

* If you open an object, please ensure that you close it before you navigate to a different place.

* Opening object like drawers, cabinets, fridge can block the path of the robot. So open objects only when you think it is necessary.

* NOTE: DO NOT OUTPUT ANYTHING EXTRA OTHER THAN WHAT HAS BEEN SPECIFIED

### H.2 Act

We describe the prompt used for the Act baseline:

## Prompt for the Act Baseline

You are an excellent planner and robot controller who is tasked with helping 2 embodied robots named Alice and Bob carry out a task. Both robots have a partially observable view of the environment. Hence they have to explore around in the environment to do the task.

They can perform the following actions:

["navigate to object <object_id>", "rotate in <rotation> direction", "pick up object <object_id>", "put object on <receptacle_id>", "open object <object_id>", "close object <object_id>", "slice object <object_id>", "toggle object <object_id> on", "toggle object <object_id> off", "clean object <object_id>", "look up by angle <angle>", "look down by angle <angle>", "move in <translation> direction", "stay idle", "Done"]

Here "Done" is used when all the robots have completed the main task. Only use it when you think all the subtasks are complete.

"stay idle" is used when you want the robot to stay idle for a one-time step. This could be used to wait for the other robot to complete its subtask. Use it only when you think it is necessary.

Here <rotation> can be one of ["Right", "Left"].

Here <angle> is the angle in degrees and can only be one of $\left\lbrack  {{30},{60},{90},{120},{150},{180}}\right\rbrack$ .

Here <translation> can be one of ["Ahead", "Back", "Left", "Right"].

You need to suggest the action that each robot should take at the current time step.

## ###Important Notes ###

* The robots can hold only one object at a time.

For example: If Alice is holding an apple, she cannot pick up another object until she puts the apple down.

* Even if the robot can see objects, it might not be able to interact with them if they are too far away. Hence you will need to make the robot navigate closer to the objects they want to interact with.

For example: An action like "pick up <object_id>" is feasible only if robot can see the object and is close enough to it. So you will have to navigate closer to it before you can pick it up.

* In some scenarios, the agents might not see the objects that they want to interact with. In such cases, you will have to make the robot explore the environment to find the object. In such scenarios you can use actions to rotate in place or look up / down or navigate to explore the environment.

* If you open an object, please ensure that you close it before you navigate to a different place.

* Opening object like drawers, cabinets, fridge can block the path of the robot. So open objects only when you think it is necessary.

## ###INPUT FORMAT ###

* You will get a description of the task robots are supposed to do.

* You will get an image of the environment at the current time step from Alice's perspective and Bob's perspective as the observation input. Here the objects are named as "<object_name>_<object_id>".

* You will get a trace of the steps taken by the robots and the actions they took at each time step and whether it was successful or not.

## ###OUTPUT FORMAT ###

In your output, do not have any extra text or content outside of the python dictionary as below. Do NOT put any text, spaces, or enter keys (i.e. "/n") outside of it.

Your output should ONLY be in the form of a python dictionary, without any reasoning or extra text, as shown below:

\{"Alice": "action to be taken by Alice",

"Bob": "action to be taken by Bob\}

For example: If you think Alice should pick up an apple and Bob should navigate to the fridge, you will have to give the output as:

\{"Alice": "pick up apple",

"Bob": "navigate to fridge"\}

* NOTE: DO NOT OUTPUT ANYTHING EXTRA OTHER THAN WHAT HAS BEEN SPECIFIED

### H.3 ReAct

We describe the prompt used for the ReAct baseline:

## Prompt for ReAct Baseline

You are an excellent planner who is tasked with helping 2 embodied robots named Alice and Bob carry out a task. Both robots have a partially observable view of the environment. Hence they have to explore around in the environment to do the task.

They can perform the following actions: ["navigate to object <object_id>", "rotate in <rotation> direction", "pick up object <object_id>", "put object on <receptacle_id>", "open object <object_id>", "close object <object_id>", "slice object <object_id>", "toggle object <object_id> on", "toggle object <object_id> off", "clean object <object_id>", "look up by angle <angle>", "look down by angle <angle>", "move in <translation> direction", "stay idle", "Done"

Here "Done" is used when all the robots have completed the main task. Only use it when you think all the subtasks are complete.

"stay idle" is used when you want the robot to stay idle for a one-time step. This could be used to wait for the other robot to complete its subtask. Use it only when you think it is necessary.

Here <rotation> can be one of ["Right,"

Here <angle> is the angle in degrees and can only be one of $\left\lbrack  {{30},{60},{90},{120},{150},{180}}\right\rbrack$ .

Here <translation> can be one of ["Ahead", "Back", "Left", "Right"].

You need to suggest the action that each robot should take at the current time step.

## ###Important Notes ###

* The robots can hold only one object at a time.

For example: If Alice is holding an apple, she cannot pick up another object until she puts the apple down.

* Even if the robot can see objects, it might not be able to interact with them if they are too far away. Hence you will need to make the robot navigate closer to the objects they want to interact with.

For example: An action like "pick up <object_id>" is feasible only if robot can see the object and is close enough to it. So you will have to navigate closer to it before you can pick it up.

* In some scenarios, the agents might not see the objects that they want to interact with. In such cases, you will have to make the robot explore the environment to find the object. In such scenarios you can use actions to rotate in place or look up / down or navigate to explore the environment.

* If you open an object, please ensure that you close it before you navigate to a different place.

* Opening object like drawers, cabinets, fridge can block the path of the robot. So open objects only when you think it is necessary.

## ###INPUT FORMAT ###

* You will get a description of the task robots are supposed to do.

* You will get an image of the environment at the current time step from Alice's perspective and Bob's perspective as the observation input. Here the objects are named as "<object_name>_<object_id>".

* You will get a trace of the steps taken by the robots and the actions they took at each time step and whether it was successful or not.

## ###OUTPUT FORMAT ###

You are supposed to think and suggest the action each robot is supposed to take at the current time step. Before suggesting an action you need to think, which requires that you reason over the inputs and logically reflect on the task, observation and course of actions needed to complete the task.

Output Requirements: At each time step you must ONLY output a PYTHON DICTIONARY of the following two elements:

*First Element: Key = "Think" | Value:(Type: String): A logical reflection of the best action to be taken given the inputs: task at hand, observations, and trace.

*Second Element: Key = "Action" | Value:(Type: Python Dictionary):

The value should be in the form of a python dictionary as shown below.

\{"Alice": "action to be taken by Alice", "Bob": "action to be taken by Bob"\}

For example: If you think Alice should pick up an apple and Bob should navigate to the fridge, you will have to give the output as: \{"Alice": "pick up apple", "Bob": "navigate to fridge"\} Here is an example output:

\{"Think": "To solve the task, I need to find and put the apple. The apple is likely to be on the countertop or table. Then find the fridge.", "Action": \{"Alice": "pick up apple", "Bob": "navigate to fridge"\}\} * NOTE: DO NOT OUTPUT ANYTHING EXTRA OTHER THAN WHAT HAS BEEN SPECIFIED

### H.4 Chain of Thought

We describe the prompt used for the Chain-of-Thought baseline:

## Prompt for Chain of Thought Baseline

You are an excellent planner who is tasked with helping 2 embodied robots named Alice and Bob carry out a task. Both robots have a partially observable view of the environment. Hence they have to explore around in the environment to do the task.

They can perform the following actions: ["navigate to object <object_id>", "rotate in <rotation> direction", "pick up object <object_id>", "put object on <receptacle_id>", "open object <object_id>", "close object <object_id>", "slice object <object_id>", "toggle object <object_id> on", "toggle object <object_id> off", "clean object <object_id>", "look up by angle <angle>", "look down by angle <angle>", "move in <translation> direction", "stay idle", "Done"] Here "Done" is used when all the robots have completed the main task. Only use it when you think all the subtasks are complete. "stay idle" is used when you want the robot to stay idle for a one-time step. This could be used to wait for the other robot to complete its subtask. Use it only when you think it is necessary. Here <rotation> can be one of ["Right", "Left"].

Here <angle> is the angle in degrees and can only be one of $\left\lbrack  {{30},{60},{90},{120},{150},{180}}\right\rbrack$ .

Here <translation> can be one of ["Ahead", "Back", "Left", "Right"].

You need to suggest the action that each robot should take at the current time step.

## ###Important Notes ###

* The robots can hold only one object at a time. For example: If Alice is holding an apple, she cannot pick up another object until she puts the apple down.

* Even if the robot can see objects, it might not be able to interact with them if they are too far away. Hence you will need to make the robot navigate closer to the objects they want to interact with. For example: An action like "pick up $<$ object_id>" is feasible only if robot can see the object and is close enough to it. So you will have to navigate closer to it before you can pick it up.

* In some scenarios, the agents might not see the objects that they want to interact with. In such cases, you will have to make the robot explore the environment to find the object. In such scenarios you can use actions to rotate in place or look up / down or navigate to explore the environment.

* If you open an object, please ensure that you close it before you navigate to a different place.

* Opening object like drawers, cabinets, fridge can block the path of the robot. So open objects only when you think it is necessary.

## ###INPUT FORMAT ###

* You will get a description of the task robots are supposed to do.

* You will get an image of the environment at the current time step from Alice's perspective and Bob's perspective as the observation input. Here the objects are named as "<object_name>_<object_id>".

* You will get a trace of the steps taken by the robots and the actions they took at each time step and whether it was successful or not.

## ###OUTPUT FORMAT ###

You are supposed to FIRST reason through the situation logically and step by step, then suggest the action each robot is supposed to take at the current time step.

In your output, do not have any extra text or content outside of the python dictionary as below.

Your output should ONLY be in the form of a python dictionary as shown below:

\{"reason": "Reasoning for action plan....", "Alice": "action to be taken by Alice", "Bob": "action to be taken by Bob"\}

Put all of your reasoning inside of the "reason" key of the dictionary. Do NOT put any text, spaces, or enter keys (i.e. "/n") outside of it.

For example: If you think Alice should pick up an apple and Bob should navigate to the fridge, you will have to give the output as:

\{"reason": "since the subtask list is empty, the robots need to transport the apple to the fridge", "Alice": "pick up apple", "Bob": "navigate to fridge"\}

Let's think step by step, but make sure to put all of your reasoning inside of the "reason" key of the dictionary!

* NOTE: DO NOT OUTPUT ANYTHING EXTRA OTHER THAN WHAT HAS BEEN SPECIFIED

### H.5 SmartLLM

We adapt the prompt from the official codebase of SmartLLM (master branch; commit #be42930050f7d4d8f2fad027aff14a699c3300aa) as given here: https://github.com/SMARTlab-Purdue/SMART-LLM/blob/master/scripts/run_llm.py with a slight modification. Instead of letting the agents access all the objects in the environment through the simulator metadata, we just give the list of objects visible from the agents' point-of-view.

### H.6 CoELA

We adapt the prompt from the official codebase of CoELA (master branch: commit #3d34de46dc77f9aaabe438cd2b92ea6c5c04973a) as given here: https://github.com/UMass Foundation-Model/Co-LLM-Agents/tree/master/tdw_mat/LLM. We modify some aspects of the prompt as described: Instead of relying on the simulator/pre-defined conditional logic for generating the list of available action options, we give a list of all possible actions based on the observation. This includes the option to send the communication message, all navigation actions, and all combinations of valid actions with the interactable objects in the current observation.

## I Open Source VLMs

We list the source of the weights we used for the open-source VLMs:

- Idefics 2 [68, 69]: We use the 8B base model fine-tuned on a mixture of supervised and instruction datasets (text-only and multimodal datasets) from HuggingFace. The weights were downloaded from https://huggingface.co/HuggingFaceM4/idefics2-8b with the commit #2c031da2dc71f3ac989f9efa9b8ff476df3842c0. We chose Idefics because it is able to take multiple images as input similar to GPT-4V and reason on them.

- LLaVA [70]: We use the 7B model t trained by fine-tuning LLaMA/Vicuna on GPT-generated multimodal instruction-following data. The weights were downloaded from https://huggingface.co/llava-hf/lava-1.5-7b-hf with the commit # 05ae2434cbb430be33edcba0c5203e7023f785b7.

- CogVLM [72]: We use the 18B model. The weights were downloaded from https://huggingface.co/THUDM/cogagent-chat-hf with the commit # d519da3b191401234f4bd86ce1c287c61bc276a3.

## J SentenceBERT fine-tuning

We finetuned a pre-trained BERT model to function as a semantic mapper between free-form natural language output and the robot's admissible actions in the environment. The pre-trained weights were obtained from https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2 The model was trained on a dataset consisting of 2800 free-form input, valid action output pairs. It ran on one (1) Apple M1 core for a wall clock time of 5 minutes. Table 8 shows the hyper-parameters used for the pre-training of the BERT model.

<table><tr><td>Epochs</td><td>10</td></tr><tr><td>Max gradient norm</td><td>1</td></tr><tr><td>Learning rate</td><td>$2 \times  {10}^{-5}$</td></tr><tr><td>Batch size</td><td>64</td></tr><tr><td>Encoding dimension</td><td>384</td></tr><tr><td>Optimizer</td><td>AdamW</td></tr><tr><td>Scheduler</td><td>Warm-up linear</td></tr><tr><td>Warm-up steps</td><td>45</td></tr><tr><td>Weight decay</td><td>0.01</td></tr><tr><td>Loss scale</td><td>20</td></tr><tr><td>Loss type</td><td>Multiple negatives ranking loss</td></tr><tr><td>Similarity function</td><td>Cosine similarity</td></tr></table>

Table 8: Hyper-parameters for the model fine-tuning including the loss.

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

## Answer: [Yes]

Justification: Our claims in the abstract are reflected in the experimental results shown in the paper.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.

- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.

- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.

- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

## Answer: [Yes]

Justification: we mention the limitations of our paper in a separate section in the paper.

Guidelines:

- The answer NA means that the paper has no limitation while the answer No means that the paper has limitations, but those are not discussed in the paper.

- The authors are encouraged to create a separate "Limitations" section in their paper.

- The paper should point out any strong assumptions and how robust the results are to violations of these assumptions (e.g., independence assumptions, noiseless settings, model well-specification, asymptotic approximations only holding locally). The authors should reflect on how these assumptions might be violated in practice and what the implications would be.

- The authors should reflect on the scope of the claims made, e.g., if the approach was only tested on a few datasets or with a few runs. In general, empirical results often depend on implicit assumptions, which should be articulated.

- The authors should reflect on the factors that influence the performance of the approach. For example, a facial recognition algorithm may perform poorly when image resolution is low or images are taken in low lighting. Or a speech-to-text system might not be used reliably to provide closed captions for online lectures because it fails to handle technical jargon.

- The authors should discuss the computational efficiency of the proposed algorithms and how they scale with dataset size.

- If applicable, the authors should discuss possible limitations of their approach to address problems of privacy and fairness.

- While the authors might fear that complete honesty about limitations might be used by reviewers as grounds for rejection, a worse outcome might be that reviewers discover limitations that aren't acknowledged in the paper. The authors should use their best judgment and recognize that individual actions in favor of transparency play an important role in developing norms that preserve the integrity of the community. Reviewers will be specifically instructed to not penalize honesty concerning limitations.

## 3. Theory Assumptions and Proofs

Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof?

Answer: [NA]

Justification: We do not include any theoretical results in the paper and hence we do not report the theoretical assumptions. Although we note the assumptions in the AI2Thor environment in which we run all our experiments.

Guidelines:

- The answer NA means that the paper does not include theoretical results.

- All the theorems, formulas, and proofs in the paper should be numbered and cross-referenced.

- All assumptions should be clearly stated or referenced in the statement of any theorems.

- The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.

- Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material.

- Theorems and Lemmas that the proof relies upon should be properly referenced.

## 4. Experimental Result Reproducibility

Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)?

## Answer: [Yes]

Justification: We provide as much detail as we can for the reproducibility of the experiments. Although there will be some variations to the underlying LLMs/VLMs we note the specific LLM/VLM model we used for our experiments in the Appendix.

Guidelines:

- The answer NA means that the paper does not include experiments.

- If the paper includes experiments, a No answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not.

- If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable.

- Depending on the contribution, reproducibility can be accomplished in various ways. For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed.

- While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example

(a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm.

(b) If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully.

(c) If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset).

(d) We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility. In the case of closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results.

## 5. Open access to data and code

Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material?

## Answer: [Yes]

Justification: We provide a link to the GitHub repository in the abstract.

Guidelines:

- The answer NA means that paper does not include experiments requiring code.

- Please see the NeurIPS code and data submission guidelines (https://nips.cc/ public/guides/CodeSubmissionPolicy) for more details.

- While we encourage the release of code and data, we understand that this might not be possible, so "No" is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).

- The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines (https: //nips.cc/public/guides/CodeSubmissionPolicy) for more details.

- The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.

- The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.

- At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).

- Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted.

## 6. Experimental Setting/Details

Question: Does the paper specify all the training and test details (e.g., data splits, hyper-parameters, how they were chosen, type of optimizer, etc.) necessary to understand the results?

## Answer: [Yes]

Justification: We specify as many details of our experiments as we can in the appendix.

Guidelines:

- The answer NA means that the paper does not include experiments.

- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.

- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment Statistical Significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

## Answer: [Yes]

Justification: Yes, we report means and the 95% confidence intervals.

Guidelines:

- The answer NA means that the paper does not include experiments.

- The authors should answer "Yes" if the results are accompanied by error bars, confidence intervals, or statistical significance tests, at least for the experiments that support the main claims of the paper.

- The factors of variability that the error bars are capturing should be clearly stated (for example, train/test split, initialization, random drawing of some parameter, or overall run with given experimental conditions).

- The method for calculating the error bars should be explained (closed form formula, call to a library function, bootstrap, etc.)

- The assumptions made should be given (e.g., Normally distributed errors).

- It should be clear whether the error bar is the standard deviation or the standard error of the mean.

- It is OK to report 1-sigma error bars, but one should state it. The authors should preferably report a 2-sigma error bar than state that they have a ${96}\% \mathrm{{CI}}$ ,if the hypothesis of Normality of errors is not verified.

- For asymmetric distributions, the authors should be careful not to show in tables or figures symmetric error bars that would yield results that are out of range (e.g. negative error rates).

- If error bars are reported in tables or plots, The authors should explain in the text how they were calculated and reference the corresponding figures or tables in the text.

## 8. Experiments Compute Resources

Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments?

## Answer: [Yes]

Justification: Yes, we denote all these details in the appendix.

Guidelines:

- The answer NA means that the paper does not include experiments.

- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.

- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.

- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code Of Ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines?

## Answer: [Yes]

Justification: We have read the code of ethics on the website.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.

- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.

- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader Impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

## Answer: [NA]

Justification: In our opinion, our work does not have any significant societal impact other than being able to progress towards enabling embodied multi-agent robots for household planning.

Guidelines:

- The answer NA means that there is no societal impact of the work performed.

- If the authors answer NA or No, they should explain why their work has no societal impact or why the paper does not address societal impact.

- Examples of negative societal impacts include potential malicious or unintended uses (e.g., disinformation, generating fake profiles, surveillance), fairness considerations (e.g., deployment of technologies that could make decisions that unfairly impact specific groups), privacy considerations, and security considerations.

- The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to generate deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster.

- The authors should consider possible harms that could arise when the technology is being used as intended and functioning correctly, harms that could arise when the technology is being used as intended but gives incorrect results, and harms following from (intentional or unintentional) misuse of the technology.

- If there are negative societal impacts, the authors could also discuss possible mitigation strategies (e.g., gated release of models, providing defenses in addition to attacks, mechanisms for monitoring misuse, mechanisms to monitor how a system learns from feedback over time, improving the efficiency and accessibility of ML).

## 11. Safeguards

Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pretrained language models, image generators, or scraped datasets)?

Answer: [NA]

Justification: No high risks in the paper.

Guidelines:

- The answer NA means that the paper poses no such risks.

- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.

- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.

- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

## Answer: [Yes]

Justification: We have cited all the authors of the works we base our work on especially the AI2Thor simulator. We have also provided links to the weights we used for the open-source VLMs in the appendix with the commit # number on HuggingFace.

Guidelines:

- The answer NA means that the paper does not use existing assets.

- The authors should cite the original paper that produced the code package or dataset.

- The authors should state which version of the asset is used and, if possible, include a URL.

- The name of the license (e.g., CC-BY 4.0) should be included for each asset.

- For scraped data from a particular source (e.g., website), the copyright and terms of service of that source should be provided.

- If assets are released, the license, copyright information, and terms of use in the package should be provided. For popular datasets, paperswithcode.com/datasets has curated licenses for some datasets. Their licensing guide can help determine the license of a dataset.

- For existing datasets that are re-packaged, both the original license and the license of the derived asset (if it has changed) should be provided.

- If this information is not available online, the authors are encouraged to reach out to the asset's creators.

### 13.New Assets

Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets?

## Answer: [Yes]

Justification: We provide documentation for the initialization of the multi-agent setting in the AI2Thor simulator in the supplementary material.

Guidelines:

- The answer NA means that the paper does not release new assets.

- Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.

- The paper should discuss whether and how consent was obtained from people whose asset is used.

- At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and Research with Human Subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

## Answer: [NA]

Justification: The paper does not involve crowdsourcing nor research with human subjects. Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.

- Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.

- According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional Review Board (IRB) Approvals or Equivalent for Research with Human Subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

## Answer: [NA]

Justification: The paper does not involve crowdsourcing or research with human subjects.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.

- Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.

- We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.

- For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.