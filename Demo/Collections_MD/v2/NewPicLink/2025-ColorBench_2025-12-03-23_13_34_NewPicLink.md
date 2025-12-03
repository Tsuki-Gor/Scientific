# ColorBench: Benchmarking Mobile Agents with Graph-Structured Framework for Complex Long- Horizon Tasks
# ColorBench：用于复杂长程任务的图结构移动代理基准


Yuanyi Song ${}^{1 *  \dagger  }$ ,Heyuan Huang ${}^{2}$ ; Qiqiang Lin ${}^{2}$ ,Yin Zhao ${}^{2}$ ,Xiangmou Qu ${}^{2}$ , Jun Wang ${}^{2}$ , Xingyu Lou ${}^{2 \ddagger  }$ , Weiwen Liu ${}^{1}$ , Zhuosheng Zhang ${}^{1}$ , Jun Wang ${}^{2}$ , Yong Yu ${}^{1}$ , Weinan Zhang ${}^{1 \ddagger  }$ , Zhaoxiang Wang ${}^{2 \ddagger  }$
宋元谊 ${}^{1 *  \dagger  }$ ,黄和远 ${}^{2}$ ; 林其强 ${}^{2}$ ,赵寅 ${}^{2}$ ,曲祥谋 ${}^{2}$ ,王峻 ${}^{2}$ ,娄星宇 ${}^{2 \ddagger  }$ ,刘威文 ${}^{1}$ ,张灼晟 ${}^{1}$ ,王峻 ${}^{2}$ ,余勇 ${}^{1}$ ,张伟南 ${}^{1 \ddagger  }$ ,王照祥 ${}^{2 \ddagger  }$


${}^{1}$ Shanghai Jiao Tong University ${}^{2}$ OPPO
${}^{1}$ 上海交通大学 ${}^{2}$ OPPO


norsheep919@sjtu.edu.cn louxingyu@oppo.com
norsheep919@sjtu.edu.cn louxingyu@oppo.com


wnzhang@sjtu.edu.cn steven.wangzx@gmail.com
wnzhang@sjtu.edu.cn steven.wangzx@gmail.com


## Abstract
## 摘要


The rapid advancement of multimodal large language models has enabled agents to operate mobile devices by directly interacting with graphical user interfaces, opening new possibilities for mobile automation. However, real-world mobile tasks are often complex and allow for multiple valid solutions. This contradicts current mobile agent evaluation standards: offline static benchmarks can only validate a single predefined "golden path", while online dynamic testing is constrained by the complexity and non-reproducibility of real devices, making both approaches inadequate for comprehensively assessing agent capabilities. To bridge the gap between offline and online evaluation and enhance testing stability, this paper introduces a novel graph-structured benchmarking framework. By modeling the finite states observed during real-device interactions, it achieves static simulation of dynamic behaviors. Building on this, we develop ColorBench, a benchmark focused on complex long-horizon tasks. It supports evaluation of multiple valid solutions, subtask completion rate statistics, and atomic-level capability analysis. ColorBench contains 175 tasks (74 single-app, 101 cross-app) with an average length of over 13 steps. Each task includes at least two correct paths and several typical error paths, enabling quasi-dynamic interaction. By evaluating ColorBench across various baselines, we discover limitations of existing models and propose improvement directions and feasible technical pathways to enhance agents' performance on complex, long-horizon problems based on experimental results. Code and data are available at: ColorBench.
多模态大模型的快速发展使得代理能够通过直接操作图形用户界面在移动设备上执行任务，为移动自动化开辟了新可能。然而，现实移动任务通常复杂且存在多种可接受解，这与当前移动代理评测标准相冲突：离线静态基准只能验证单一预定义的“黄金路径”，而在线动态测试受制于真实设备的复杂性与不可复现性，二者均不足以全面评估代理能力。为弥合离线与在线评测的差距并提升测试稳定性，本文提出一种新颖的图结构基准框架。通过对真实设备交互中观测到的有限状态建模，实现对动态行为的静态模拟。在此基础上，我们构建了面向复杂长程任务的基准 ColorBench。它支持多条有效解的评估、子任务完成率统计与原子级能力分析。ColorBench 包含 175 个任务（74 个单应用，101 个跨应用），平均长度超过 13 步。每个任务至少包含两条正确路径和若干典型错误路径，实现准动态交互。通过在多种基线上评测 ColorBench，我们发现现有模型的局限，并基于实验结果提出了改进方向与可行技术路径，以提升代理在复杂长程问题上的表现。代码和数据可在：ColorBench 获得。


## 1 Introduction
## 1 引言


As one of the primary human-computer interaction entry points for today's internet, mobile devices present an urgent need to enhance internet service accessibility and user experience by exploring their automation, calling for AI agents' graphical user interface (GUI) interaction capabilities (Wen et al., 2024; Hong et al., 2024; Li et al., 2025; Chen et al., 2025a). With the advancement of artificial intelligence, utilizing multimodal large language models (MLLMs) as agents to operate graphical user interfaces GUIs for mobile tasks has gained increasing attention (Jiang et al., 2025; Cheng et al., 2025; Wang et al., 2024b;a; Wu et al., 2025), giving rise to numerous outstanding benchmarks for mobile tasks (Rawles et al., 2023; Lu et al., 2024; Sun et al., 2022; Li et al., 2024a; Liu et al., 2025a; Huang et al., 2025).
作为当今互联网主要的人机交互入口之一，移动设备亟需通过探索自动化来提升互联网服务的可及性与用户体验，这要求 AI 代理具备图形用户界面（GUI）交互能力（Wen et al., 2024；Hong et al., 2024；Li et al., 2025；Chen et al., 2025a）。随着人工智能的发展，将多模态大语言模型（MLLMs）作为代理来操作 GUI 完成移动任务受到越来越多关注（Jiang et al., 2025；Cheng et al., 2025；Wang et al., 2024b;a；Wu et al., 2025），并催生了大量优秀的移动任务基准（Rawles et al., 2023；Lu et al., 2024；Sun et al., 2022；Li et al., 2024a；Liu et al., 2025a；Huang et al., 2025）。


---



*This work was done during Yuanyi Song's internship at OPPO.
*该工作在宋元谊于 OPPO 实习期间完成。


${}^{ \dagger  }\mathrm{Y}$ . Song and H. Huang contributed equally to this research.
${}^{ \dagger  }\mathrm{Y}$ . 宋与 H. 黄对本研究贡献相同。


${}^{ \ddagger  }\mathrm{X}$ . Lou,W. Zhang and Z. Wang are the corresponding authors.
${}^{ \ddagger  }\mathrm{X}$ . 娄、W. 张与 Z. 王为通讯作者。


---



Existing mobile GUI agent benchmarks can be broadly categorized into two paradigms (Zhang et al., 2024a; Xu et al., 2025):
现有移动 GUI 代理基准大致可分为两类范式（Zhang et al., 2024a；Xu et al., 2025）：


1) Offline Static Evaluation: This paradigm involves assessing agent performance using static image trajectory data. Although widely adopted due to the feasibility of rapid, large-scale data collection, this method is subject to several prominent limitations: a) Rigid Assessment. It relies on fixed trajectories for evaluation, failing to assess multiple potential solutions and leading to potential misjudgments. b) Oversimplified Metrics. Its evaluation metrics are singular, overemphasizing step-level success rates while lacking a comprehensive assessment of overall task completion. c) Coarse-Grained Diagnosis. Its evaluation dimensions are broad, considering only task-wise completion status and lacking fine-grained analysis of atomic capabilities. These issues cause a significant "offline-online evaluation discrepancy" during static testing, which means an agent's performance on offline static evaluations may not positively correlate with its actual device performance (Li et al., 2024a; Sun et al., 2022).
1) 离线静态评估：该范式使用静态图像轨迹数据评估代理性能。尽管由于可快速、大规模采集数据而被广泛采用，但该方法存在若干显著局限：a) 刻板评估。依赖固定轨迹进行评测，无法考察多种潜在解，容易导致误判。b) 过于简化的度量。评估指标单一，过分强调步级成功率，缺乏对整体任务完成度的综合考量。c) 粗粒度诊断。评估维度宽泛，仅考虑任务级完成情况，缺乏对原子能力的细化分析。这些问题在静态测试中造成显著的“离线-在线评估差异”，即代理在离线静态评估上的表现可能与其在真实设备上的实际性能不呈正相关 (Li et al., 2024a; Sun et al., 2022)。


<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2025/12/2025_12_03__23_17_55_cac816.jpg"/>



Figure 1: Structure of the ColorBench. It illustrates multi-path solutions, reflective backtracking and automated evaluation milestones, demonstrating how the graph is structured and utilized. Each node represents a screen list.
Figure 1: ColorBench 的结构。图示多路径解、可回溯反射与自动化评估里程碑，展示图的构建与使用方式。每个节点代表一个屏幕列表。


2) Online Dynamic Evaluation: This paradigm assesses models within a dynamic environment to deliver more authentic performance metrics. It can be further categorized into two types: virtual environment evaluation and real-device evaluation. However, existing virtual testing environments like AndroidWorld (Rawles et al., 2024) and AndroidLab (Xu et al., 2024) involve applications and interaction patterns that differ from real-world scenarios, making it challenging to accurately reflect an agent's true deployment performance. Moreover, its implementation is also relatively challenging. While real-device evaluation is highly valued for its ability to accurately reflect user scenarios, it suffers from the following defects: a) Instability. Variations in page loading delays and sudden ad pop-ups may cause unexpected interruptions, leading to ambiguous evaluation criteria; b) Inefficient Result Assessment. The highly dynamic GUI makes automated checks difficult, forcing reliance on time-consuming manual verification (Dai et al., 2025; Hu et al., 2024). c) Security Risks. Most applications require account login to access full functionality, posing risks of unintended payments, data deletion, or personal information leaks during evaluations (Ma et al., 2024; Chen et al., 2025c; Zhang et al., 2024c). These factors directly result in poor reproducibility and low efficiency during dynamic testing.
2) 在线动态评估：该范式在动态环境中评估模型，以提供更真实的性能度量。可细分为虚拟环境评估与真实设备评估两类。然而，现有虚拟测试环境如 AndroidWorld (Rawles et al., 2024) 与 AndroidLab (Xu et al., 2024) 中的应用与交互模式与真实场景存在差异，难以准确反映代理的真实部署表现。此外，其实现也相对困难。虽然后者能较准确反映用户场景，但真实设备评估存在以下缺陷：a) 不稳定性。页面加载延迟与突发广告可能导致意外中断，造成评估标准模糊；b) 结果评估低效。高度动态的 GUI 使自动化检查困难，迫使依赖耗时的人工验证 (Dai et al., 2025; Hu et al., 2024)。c) 安全风险。大多数应用需登录账户以访问全部功能，评估过程中可能存在意外支付、数据删除或个人信息泄露的风险 (Ma et al., 2024; Chen et al., 2025c; Zhang et al., 2024c)。这些因素直接导致动态测试中的可复现性差与效率低。


To address these problems, we propose ColorBench, a novel graph-structured mobile agent benchmark for complex long-horizon tasks. As presented in Table 1, ColorBench aims to strike a balance between offline static and online dynamic evaluation through a finite-state simulation, maintaining the stability of the former while incorporating the flexibility of the latter. Moreover, we focus on complex long-horizon tasks due to their composite nature, which is characterized by multiple atomic subtasks executed in sequential, parallel, or recursive patterns, along with the existence of multiple path solutions, thereby making a graph-structured benchmark ideally suited for this context.
为解决这些问题，我们提出 ColorBench，一种用于复杂长时序任务的新型图结构移动代理基准。如表 1 所示，ColorBench 通过有限状态模拟在离线静态与在线动态评估间取得平衡，保持前者的稳定性同时兼具后者的灵活性。此外，我们聚焦复杂长时序任务，因为其具有复合特性：由多个原子子任务以顺序、并行或递归方式组合，并存在多路径解，这使得图结构基准尤为适用。


We organize the colorbench into a strongly connected graph as shown in Figure 1, where mobile screen states serve as nodes and action transition relationships between nodes serve as edges, multiple solutions (paths of different colors), reflective backtracking (the blue path), and automated evaluation milestones (the red flag). Specifically, we designed an efficient graph-structured benchmark construction methodology, enabling subsequent researchers to expand or reconstruct graphs. It
我们将 ColorBench 构建为如图 1 所示的强连通图，移动屏幕状态作为节点，节点间的动作转移关系作为边，体现多解（不同颜色的路径）、可回溯反射（蓝色路径）与自动化评估里程碑（红旗）。具体而言，我们设计了高效的图结构基准构建方法，便于后续研究者扩展或重构图。它


Table 1: Comparison between ColorBench and other mobile agents benchmarks. Our ColorBench is a static graph environment but supports interaction similar to real world. The column "Interaction" indicates whether to support the agent to interact with provided environment when evaluating. The column "AtomCap" means whether to support atomic capability assessment. The "Step" only calculates the optimal solution for each task and does not account for randomly appearing advertisements or longer correct paths. It is worth mentioning that the benchmark statistics in the third section include the training set. collects static trajectories via breadth-first and depth-first algorithms, merges nodes and edges by computing screenshot similarity, and automatically labels bounding boxes as transition conditions. Manual validation is integrated throughout to ensure data quality.
Table 1: ColorBench 与其他移动代理基准的比较。我们的 ColorBench 是静态图环境，但支持类似真实世界的交互。“Interaction” 列表示在评估时是否支持代理与提供环境交互。“AtomCap” 表示是否支持原子能力评估。“Step” 仅计算每个任务的最优解，不考虑随机出现的广告或更长的正确路径。值得一提的是，第三节中的基准统计包含训练集。通过广度优先与深度优先算法采集静态轨迹，计算截图相似度合并节点与边，并自动标注作为转移条件的边界框，同时在各环节融入人工验证以确保数据质量。


<table><tr><td>Dataset</td><td>#Tasks</td><td>#Step</td><td>#Apps</td><td>Interaction</td><td>Multiple Solution</td><td>Atom-Cap</td></tr><tr><td>AndroidLab (Xu et al., 2024)</td><td>138</td><td>8.5</td><td>9</td><td>✓</td><td>✓</td><td>✘</td></tr><tr><td>AndroidWorld (Rawles et al., 2024)</td><td>116</td><td>-</td><td>20</td><td>✓</td><td>✓</td><td>✘</td></tr><tr><td>Mobile-Env (Zhang et al., 2023)</td><td>150</td><td>-</td><td>-</td><td>✓</td><td>✓</td><td>✘</td></tr><tr><td>MobileAgentBench (Wang et al., 2024c)</td><td>100</td><td>-</td><td>10</td><td>✘</td><td>✓</td><td>✘</td></tr><tr><td>UI-NEXUS (Guo et al., 2025)</td><td>100</td><td>14.05</td><td>50</td><td>✘</td><td>✓</td><td>✘</td></tr><tr><td>Mobile-Eval (Wang et al., 2024b)</td><td>33</td><td>5.5</td><td>10</td><td>✘</td><td>✓</td><td>✘</td></tr><tr><td>Mobile-Eval-E (Wang et al., 2025b)</td><td>25</td><td>14.56</td><td>15</td><td>✘</td><td>✓</td><td>✘</td></tr><tr><td>SPA-BENCH (Chen et al., 2024a)</td><td>340</td><td>8.2</td><td>66</td><td>✘</td><td>✓</td><td>✘</td></tr><tr><td>MVISU-Bench (Huang et al., 2025)</td><td>404</td><td>-</td><td>137</td><td>✘</td><td>✓</td><td>✘</td></tr><tr><td>AppAgent (Zhang et al., 2025)</td><td>50</td><td>-</td><td>10</td><td>✘</td><td>✓</td><td>✘</td></tr><tr><td>AndroidControl (Li et al., 2024a)</td><td>15283</td><td>4.8</td><td>833</td><td>✘</td><td>✘</td><td>✘</td></tr><tr><td>GUI-Odyssey (Lu et al., 2024)</td><td>7735</td><td>15.4</td><td>201</td><td>✘</td><td>✘</td><td>✘</td></tr><tr><td>Meta-GUI (Sun et al., 2022)</td><td>1125</td><td>4.3</td><td>-</td><td>✘</td><td>✘</td><td>✘</td></tr><tr><td>Mobile-Bench-v2 (Xu et al., 2025)</td><td>12,856</td><td>7.28</td><td>49</td><td>✘</td><td>✓</td><td>✘</td></tr><tr><td>ColorBench</td><td>175</td><td>13.13+</td><td>21</td><td>✓</td><td>✓</td><td>✓</td></tr></table>
<table><tbody><tr><td>数据集</td><td>#任务</td><td>#步骤</td><td>#应用</td><td>交互</td><td>多解</td><td>Atom-Cap</td></tr><tr><td>AndroidLab (Xu et al., 2024)</td><td>138</td><td>8.5</td><td>9</td><td>✓</td><td>✓</td><td>✘</td></tr><tr><td>AndroidWorld (Rawles et al., 2024)</td><td>116</td><td>-</td><td>20</td><td>✓</td><td>✓</td><td>✘</td></tr><tr><td>Mobile-Env (Zhang et al., 2023)</td><td>150</td><td>-</td><td>-</td><td>✓</td><td>✓</td><td>✘</td></tr><tr><td>MobileAgentBench (Wang et al., 2024c)</td><td>100</td><td>-</td><td>10</td><td>✘</td><td>✓</td><td>✘</td></tr><tr><td>UI-NEXUS (Guo et al., 2025)</td><td>100</td><td>14.05</td><td>50</td><td>✘</td><td>✓</td><td>✘</td></tr><tr><td>Mobile-Eval (Wang et al., 2024b)</td><td>33</td><td>5.5</td><td>10</td><td>✘</td><td>✓</td><td>✘</td></tr><tr><td>Mobile-Eval-E (Wang et al., 2025b)</td><td>25</td><td>14.56</td><td>15</td><td>✘</td><td>✓</td><td>✘</td></tr><tr><td>SPA-BENCH (Chen et al., 2024a)</td><td>340</td><td>8.2</td><td>66</td><td>✘</td><td>✓</td><td>✘</td></tr><tr><td>MVISU-Bench (Huang et al., 2025)</td><td>404</td><td>-</td><td>137</td><td>✘</td><td>✓</td><td>✘</td></tr><tr><td>AppAgent (Zhang et al., 2025)</td><td>50</td><td>-</td><td>10</td><td>✘</td><td>✓</td><td>✘</td></tr><tr><td>AndroidControl (Li et al., 2024a)</td><td>15283</td><td>4.8</td><td>833</td><td>✘</td><td>✘</td><td>✘</td></tr><tr><td>GUI-Odyssey (Lu et al., 2024)</td><td>7735</td><td>15.4</td><td>201</td><td>✘</td><td>✘</td><td>✘</td></tr><tr><td>Meta-GUI (Sun et al., 2022)</td><td>1125</td><td>4.3</td><td>-</td><td>✘</td><td>✘</td><td>✘</td></tr><tr><td>Mobile-Bench-v2 (Xu et al., 2025)</td><td>12,856</td><td>7.28</td><td>49</td><td>✘</td><td>✓</td><td>✘</td></tr><tr><td>ColorBench</td><td>175</td><td>13.13+</td><td>21</td><td>✓</td><td>✓</td><td>✓</td></tr></tbody></table>


The graph-structured benchmark effectively resolves the aforementioned limitations present in the existing benchmarks. As shown in Figure 2, it mainly has the following key advantages: 1) Multi-Solution Evaluation: It supports the evaluation of multiple valid solutions for a single task; 2) Enhanced Collaboration: It enables agents to fully utilize collaborative abilities such as reflection and backtracking, thereby preventing the underestimation of model capabilities due to environmental constraints; 3) Atomic Assessment: By setting milestones for subtasks, it enables stable and automated evaluation as well as fine-grained assessment of atomic task capabilities for diagnosing atomic-task level weaknesses that lead to agent failures; 4) Controllable Environment: It provides a secure, stable, and controllable testing environment that remains statically reliable while supporting rich interactions. Together, these properties allow the framework to effectively bridge the gap between offline and online evaluation.
图结构基准有效解决了现有基准中上述限制。如图2所示，主要具有以下关键优势：1) 多解评估：支持对单一任务的多种有效解进行评估；2) 增强协作性：使代理能够充分利用反思与回溯等协作能力，从而避免因环境约束而低估模型能力；3) 原子化评估：通过为子任务设置里程碑，实现稳定且自动化的评估，并对原子任务能力进行细粒度诊断，以定位导致代理失败的原子任务弱点；4) 可控环境：提供安全、稳定且可控的测试环境，在保持静态可靠性的同时支持丰富交互。综上，这些属性使该框架能够有效弥合离线与在线评估之间的差距。


We carefully select three commonly used closed-source models and ten open-source models, alongside two customized baselines, for a comprehensive evaluation on ColorBench. Through extensive experiments, we systematically reveal the limitations of existing agents in tackling complex long-horizon tasks and diagnose their underlying causes. Based on these findings, we provide concrete recommendations for developing more capable agents suitable for such challenging scenarios.
我们谨慎选择了三款常用闭源模型和十款开源模型，外加两种定制基线，在 ColorBench 上进行全面评估。通过大量实验，我们系统揭示了现有代理在应对复杂长时序任务时的局限性并诊断其根本原因。基于这些发现，我们给出面向此类挑战性场景的更强能力代理的具体建议。


Overall, the primary contributions of our work can be summarized as follows:
总体而言，我们工作的主要贡献可概括如下：


- Graph-Structured Benchmark. We propose a benchmark framework of graph structure that bridges the discrepancy between offline and online tests We design an effective construction methodology that balances quality and efficiency and verify the significance and feasibility of the graph paradigm by statistical experiments.
- 图结构基准。我们提出了一个图结构的基准框架，用以弥合离线与在线测试的差异。我们设计了兼顾质量与效率的有效构建方法，并通过统计实验验证了图范式的重要性与可行性。


- ColorBench for Complex Long-Horizon Task. It is the first comprehensive graph-structured mobile agent benchmark for complex long-horizon tasks. By extending step-level evaluation to the atomic-task level, we assess and pinpoint models' weaknesses in atomic capabilities through evaluations within complete complex tasks.
- 面向复杂长时序任务的 ColorBench。它是首个面向复杂长时序任务的全面图结构移动代理基准。通过将步级评估扩展到原子任务层面，我们在完整复杂任务评估中评估并定位模型在原子能力上的弱点。


- Pathways and Insights for ColorBench Solutions. We conduct systematic experiments and analyses on ColorBench and offer improvement directions and insights for future complex long-horizon task solutions based on experimental findings.
- ColorBench 的路径与洞见。我们在 ColorBench 上进行了系统实验与分析，并基于实验结果为未来复杂长时序任务的解决方案提供改进方向与洞见。


## 2 Related Work
## 2 相关工作


### 2.1 Mobile GUI Agent Benchmark
### 2.1 移动 GUI 代理基准


Traditional GUI agent evaluation methods can be broadly categorized into two types: offline static evaluation based on trajectory chains, and end-to-end online dynamic evaluation (Rawles et al., 2023; Sun et al., 2022; Li et al., 2024a). Dynamic evaluation can be further subdivided into virtual sandbox environments and real devices (Rawles et al., 2024; Xu et al., 2024; Zhang et al., 2023). For applications requiring login credentials, there is no fundamental difference in their impact on the real world. Static evaluation is straightforward and comprehensive. Using image-answer pairs alone, benchmarks can be created for step tasks such as UI recognition, grounding capabilities(Deka et al., 2017; Wang et al., 2021; Li et al., 2020), page transition relationship identification (Chen et al., 2024b), and simple atomic tasks (Li et al., 2024a; Deng et al., 2023; Taleby Ahvanooey et al., 2016; Zhang et al., 2024b; Wu et al., 2025), as well as demonstration learning capabilities (Liu et al., 2025a). However, for complex long-horizon tasks composed of multiple atomic tasks or even spanning multiple apps, multiple solutions often exist (Guo et al., 2025; Lu et al., 2024; Wang et al., 2025b). Static datasets, constrained by their single standard answer, struggle to accurately assess real-world model capabilities (Wang et al., 2025a). Existing GUI benchmarks for complex long-horizon tasks across various platforms and scenarios typically employ dynamic testing in real or simulated environments (Liu et al., 2025b; Ye et al., 2025b). Unlike the web environment, mobile GUI operations lack accessible URLs, and inherent issues with mobile applications make dynamic benchmarks unsuitable as unified evaluation standards. Addressing this challenge is crucial for advancing mobile GUI agent deployment. Therefore, we innovatively propose a graph-structured benchmark for complex long-horizon tasks on mobile devices, which bridges the offline-online test discrepancy with enhanced stability compared to dynamic evaluation, addressing the shortcomings of this research. Table 1 compares ColorBench with other mobile GUI agent benchmarks.
传统的 GUI 代理评估方法大致可分为两类：基于轨迹链的离线静态评估和端到端的在线动态评估（Rawles et al., 2023；Sun et al., 2022；Li et al., 2024a）。动态评估又可细分为虚拟沙箱环境和真实设备（Rawles et al., 2024；Xu et al., 2024；Zhang et al., 2023）。对于需要登录凭据的应用，其对现实世界的影响并无本质差异。静态评估直观且全面。仅凭图像-答案对即可为步骤任务构建基准，如 UI 识别、定位能力（Deka et al., 2017；Wang et al., 2021；Li et al., 2020）、页面跳转关系识别（Chen et al., 2024b）和简单原子任务（Li et al., 2024a；Deng et al., 2023；Taleby Ahvanooey et al., 2016；Zhang et al., 2024b；Wu et al., 2025），以及演示学习能力（Liu et al., 2025a）。然而，对于由多个原子任务组成或跨多应用的复杂长时序任务，通常存在多种解法（Guo et al., 2025；Lu et al., 2024；Wang et al., 2025b）。受限于单一标准答案的静态数据集难以准确评估真实世界中的模型能力（Wang et al., 2025a）。现有面向各平台与场景的复杂长时序任务的 GUI 基准通常在真实或模拟环境中采用动态测试（Liu et al., 2025b；Ye et al., 2025b）。不同于 Web 环境，移动 GUI 操作缺乏可访问的 URL，且移动应用的固有问题使得动态基准无法作为统一评估标准。解决这一挑战对于推进移动 GUI 代理部署至关重要。因此，我们创新性地提出了面向移动设备复杂长时序任务的图结构基准，该基准在相较动态评估具有更高稳定性的同时弥合了离线与在线测试差异，针对该研究的不足提出了解决方案。表1 比较了 ColorBench 与其他移动 GUI 代理基准。


### 2.2 Comparison with Prior Mobile Agent Graph
### 2.2 与既有移动代理图的比较


The graph-structured benchmark and the User Interaction Transition Graph (UTG) share a similar high-level structure but differ critically in their granularity and purpose. To meet benchmarking standards, the graph-structured benchmark constructs the topology of state transitions, whereas the UTG defines them based on page relationships. Consequently, the UTGs are primarily used for knowledge exploration during the execution of a single task (Wen et al., 2024; Fan et al., 2025), representing a subset of the evaluable environment. In contrast, the graph-structured benchmark is designed for evaluating multiple tasks. It aggregates all possible execution paths, including error states, into a comprehensive evaluation environment that serves as a subset of the real world. As such, the benchmark constitutes a superset of the exploration space covered by any single UTG.
图结构基准与用户交互迁移图(UTG)在高层结构上相似，但在粒度与目的上有关键区别。为了满足基准化要求，图结构基准构建状态迁移的拓扑，而UTG则基于页面关系来定义迁移。因此，UTG主要用于单次任务执行过程中的知识探索(Wen et al., 2024; Fan et al., 2025)，代表可评估环境的一个子集。相反，图结构基准用于评估多项任务，汇总所有可能的执行路径（包括错误状态）为一个全面的评估环境，作为现实世界的一个子集。因此，该基准构成了任何单一UTG所覆盖探索空间的超集。


Additionally, graphs have extensive application in other aspects of GUI agents. For task generation and evaluation, OmniBench automates the synthesis of complex tasks by combining subtasks based on graphs and utilizes these subtask structure graphs for evaluation (Li et al., 2024b). MobiFlow models tasks as directed acyclic graphs for evaluation purpose(Bera et al., 2018). For model training, MobileM3 connects pages collected through breadth-first exploration into graphs to learn UI transition relationships (Wu et al., 2024a). Methodologically, Xplore-Agent constructs graph-structured page relationships during exploration (Sun et al., 2025), while PG-Agent pre-builts page-relationship graphs of specific applications for RAG to augment knowledge (Chen et al., 2025b). These works demonstrate the advantages of graphs in GUI tasks: precise action transition relationships, natural page navigation modeling, and a clear, controllable global perspective. Mobile-Bench-v2 utilizes MobileM3's graphs to generate multi-path tasks, introducing a graph-structured benchmark for the first time, but it lacks systematic explanations (Xu et al., 2025). Our work fills this research gap and proposes a feasible methodology for building a graph-structured benchmark from scratch.
此外，图在GUI代理的其他方面也有广泛应用。为任务生成与评估，OmniBench通过基于图的子任务组合自动合成复杂任务，并使用这些子任务结构图进行评估(Li et al., 2024b)。MobiFlow将任务建模为用于评估的有向无环图(Bera et al., 2018)。在模型训练方面，MobileM3通过宽度优先探索采集页面并将其连接成图，以学习UI的转移关系(Wu et al., 2024a)。在方法层面，Xplore-Agent在探索过程中构建图结构的页面关系(Sun et al., 2025)，而PG-Agent预先构建特定应用的页面关系图以用于RAG增强知识(Chen et al., 2025b)。这些工作展示了图在GUI任务中的优势：精确的动作迁移关系、自然的页面导航建模，以及清晰可控的全局视角。Mobile-Bench-v2使用MobileM3的图生成多路径任务，首次引入了图结构基准，但缺乏系统性说明(Xu et al., 2025)。我们的工作填补了该研究空白，并提出了从零构建图结构基准的可行方法。


## 3 Graph-Structured Benchmark
## 3 图结构基准


### 3.1 Definition of Graph-Structured Benchmark
### 3.1 图结构基准的定义


The graph abstracts the finite states of real mobile environment into a strongly connected directed graph $G = \left( {V,E}\right)$ ,comprising two main components:
该图将真实移动环境的有限状态抽象为一个强连通有向图$G = \left( {V,E}\right)$，由两部分组成：


- $V = \left\{  {{N}_{1}\left\lbrack  {{p}_{1},{p}_{2},\ldots }\right\rbrack  ,{N}_{2}\left\lbrack  \ldots \right\rbrack  ,\ldots }\right\}$ : The node set,modeling all screen states ${N}_{i}$ that may be encountered in actual evaluations. From the HOME page to detailed app screens and inter-app navigation pages, different screen states are modeled as distinct nodes at the granularity of action transitions. For screens with random elements,they are treated as different screens ${p}_{1},{p}_{2},\ldots$ within a unified node ${N}_{i}$ when recommended content does not involve action transitions. This effectively models the randomness observed in real-device testing.
- $V = \left\{  {{N}_{1}\left\lbrack  {{p}_{1},{p}_{2},\ldots }\right\rbrack  ,{N}_{2}\left\lbrack  \ldots \right\rbrack  ,\ldots }\right\}$：节点集，用于建模在实际评估中可能遇到的所有屏幕状态${N}_{i}$。从主屏到应用内细节页面与应用间导航页面，不同屏幕状态按动作迁移的粒度被建模为不同节点。对于含随机元素的屏幕，当推荐内容不涉及动作迁移时，它们在统一节点${N}_{i}$下作为不同屏幕${p}_{1},{p}_{2},\ldots$处理。这有效模拟了真实设备测试中的随机性。


- $E = \left\{  {\left( {{N}_{i},{N}_{j},a}\right) ,\ldots }\right\}$ : The edge set models the action transition relationships between screen states at the node granularity. Each edge corresponds to a specific action a transitioning from one node ${N}_{i}$ to another ${N}_{j}$ ,such as "click bbox[x1,y1,x2,y2]" or "type[TEXT]",aligning one-to-one with the action space used in real-device evaluations.
- $E = \left\{  {\left( {{N}_{i},{N}_{j},a}\right) ,\ldots }\right\}$：边集在节点粒度上建模屏幕状态之间的动作迁移关系。每条边对应从一个节点${N}_{i}$到另一个节点${N}_{j}$的具体动作a，例如"click bbox[x1,y1,x2,y2]"或"type[TEXT]"，与真实设备评估中使用的动作空间一一对应。


It integrates all tasks into a single, invariant graph structure, simplifying the evaluation pipeline. When evaluating, each task originates from a unified "HOME" node. The agent determines actions based on user queries and the current screen, then transitions to the next screen via actions within the graph.
它将所有任务集成到单一不变的图结构中，简化评估流程。评估时，每个任务都始于统一的"HOME"节点。代理根据用户查询与当前屏幕决定动作，然后通过图内的动作转移到下一个屏幕。


### 3.2 Advantages of Graph-Structured Benchmark
### 3.2 图结构基准的优点


In this section, we detail the advantages of the graph-structured benchmark over the other two types of evaluation paradigms. Compared to static evaluation, the graph structure inherently simulates page transitions from a real mobile environment, which allows it to naturally support multi-path solutions to a problem. It is designed to encompass not only the optimal path but also sub-optimal ones and, importantly, recovery paths where the agent corrects its errors. Therefore, graphs offer superior fault tolerance, allowing models to reflect on past errors and return to re-execute without terminating the evaluation. This enables the assessment of collaborative intelligence within agent systems. Compared to dynamic evaluation, the static data construction ensures a stable, controllable, and reproducible testing process. It enables automated evaluation through predefined nodes, which represent task completion, while also avoiding security risks such as unintended payments or information leaks associated with real-world third-party app logins.
在本节中，我们详细说明图结构基准相较于另外两种评估范式的优势。与静态评估相比，图结构固有地模拟了真实移动环境中的页面跳转，使其天然支持问题的多路径解法。它不仅涵盖最优路径，还包括次优路径，且重要的是涵盖代理纠错的恢复路径。因此，图具有更强的容错性，允许模型反思过往错误并返回重新执行而不终止评估，从而能够评估代理系统中的协作智能。与动态评估相比，静态数据构建确保了测试过程的稳定、可控和可复现。通过预定义节点表示任务完成，它支持自动化评估，同时避免了真实世界第三方应用登录可能带来的意外支付或信息泄露等安全风险。


In terms of evaluation metrics, this approach breaks through the limitations of static step-level and dynamic result-level assessments. By setting sub-task milestones on the graph, it avoids misjudgments of single-step execution details. This enables effective evaluation of completion rates for complex, long-horizon tasks composed of multiple atomic tasks, while pinpointing the completion status of subtasks. Consequently, it achieves evaluation at the atomic task capability level, allowing targeted identification of an agent's shortcomings in specific tasks. Furthermore, to simulate real-world environmental variability, multiple images of functionally identical pages are retained. This approach replicates random effects in practical scenarios, striking a balance between the stability of static evaluation and the randomness of dynamic testing.
在评估指标方面，该方法突破了静态步级和动态结果级评估的局限。通过在图上设置子任务里程碑，避免了对单步执行细节的误判。它能够有效评估由多个原子任务组成的复杂长时序任务的完成率，并定位子任务的完成状态。因此，可在原子任务能力层面进行评估，针对性识别代理在具体任务上的短板。此外，为了模拟真实环境的变动性，保留了功能相同页面的多张图像，此做法复现了实际场景中的随机效应，在静态评估的稳定性与动态测试的随机性之间取得平衡。


<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2025/12/2025_12_03__23_17_55_3799e8.jpg"/>



Figure 2: Advantage of graph-structured benchmark. Our constructed ColorBench possesses these advantages. The upper-left corner demonstrates multiple solutions to one problem, supporting diverse execution paths for the same task. The lower-left corner illustrates how the graph's strongly connected structure effectively enables model collaboration capabilities such as backtracking and reflection. The upper-right corner showcases the graph's inherent suitability for complex long-horizon tasks and its support for atomic capability evaluation. The lower-right corner highlights the graph's superior controllability and reproducibility compared to dynamic evaluation.
图2：图结构基准的优势。我们构建的 ColorBench 具备这些优势。左上角展示了针对同一问题的多种解法，支持同一任务的多条执行路径。左下角说明图的强连通结构如何有效促成模型的协作能力，如回溯与反思。右上角体现了图固有地适合复杂长时序任务并支持原子能力评估。右下角强调相较于动态评估，图在可控性与可复现性方面的优势。


## 4 ColorBench
## 4 ColorBench


### 4.1 Overview and Statistical Analysis
### 4.1 概述与统计分析


To construct complex, long-horizon tasks with multiple valid solutions, we employed partially ambiguous instructions, similar to those in Mobile-Bench-v2 (Xu et al., 2025), to create a finite and controlled set of correct solutions. Our resulting benchmark, ColorBench, comprises 175 such tasks -101 cross-app and 74 single-app. After excluding random advertisements, pop-ups, and sub-optimal paths, the average optimal path length exceeds 13.13 steps. Representative tasks include price comparison and multi-content sharing (cross-app), as well as food ordering and specific content queries (single-app). The diverse combinations and strong inter-dependencies among subtasks mean each one critically influences the final outcome, thus realistically simulating real-world interactions and narrowing the gap between offline and online evaluation. From the commonalities of these subtasks, we have summarized 15 atomic task capabilities. Figure 3 displays the statistics of ColorBench, and the supported action space is detailed in Appendix A.1.
为构建具有多种有效解法的复杂长时序任务，我们采用了与 Mobile-Bench-v2 (Xu et al., 2025) 类似的部分模糊指令，以生成有限且可控的正确解集。得到的基准 ColorBench 包含 175 个此类任务——101 个跨应用任务和 74 个单应用任务。排除随机广告、弹窗和次优路径后，平均最优路径长度超过 13.13 步。具有代表性的任务包括价格比较和多内容分享（跨应用），以及点餐和特定内容查询（单应用）。子任务的多样组合及其强依赖性意味着每个子任务均对最终结果有关键影响，因而能真实模拟现实交互并缩小离线与在线评估之间的差距。基于这些子任务的共性，我们总结出 15 项原子任务能力。图3 展示了 ColorBench 的统计信息，支持的动作空间详见附录 A.1。


### 4.2 Dataset Construction
### 4.2 数据集构建


We design a graph construction strategy for benchmarking that balances quality and automation. The strategy comprises two primary phases: trajectory collection and graph merging. The trajectory collection phase aims to fully capture UI elements with high interaction probabilities on key pages within evaluation paths. It integrates both breadth-based and depth-based trajectory collection methods, simultaneously covering high-frequency short-distance tasks and complex long-distance tasks. The graph merging phase constructs an interactive evaluation environment from the collected high-quality trajectories. This phase employs semantic and visual criteria, utilizing action transitions as the key basis for distinguishing different page nodes, thereby accurately identifying effective interaction boundaries between interface states. Subsequently, we employ models to automatically annotate complete UI bounding boxes, with the detailed annotation method provided in Appendix A.2. All aforementioned processes underwent final manual quality inspection and refinement. Details regarding the models used in the automation strategy, data collection scales, and cleaning results are introduced in Appendix A.3. Throughout the construction process, we prioritized the fidelity of the graph in simulating dynamic environments. The following subsections will elaborate on the complete construction process of ColorBench.
我们设计了一种在质量与自动化之间取得平衡的图构建策略。该策略由两大阶段组成：轨迹收集与图合并。轨迹收集阶段旨在充分捕获评估路径关键页面上具有高交互概率的 UI 元素。它整合了基于广度与基于深度的轨迹收集方法，同时覆盖高频短距离任务与复杂长距离任务。图合并阶段则从收集到的高质量轨迹构建交互式评估环境。此阶段采用语义与视觉准则，以动作转移作为区分不同页面节点的关键依据，从而准确识别界面状态之间的有效交互边界。随后我们使用模型自动标注完整的 UI 包围框，详细标注方法见附录 A.2。上述所有流程均经过最终的人工质检与完善。关于自动化策略中使用的模型、数据采集规模与清洗结果的细节见附录 A.3。在整个构建过程中，我们优先保证图在模拟动态环境时的保真性。下文小节将详细阐述 ColorBench 的完整构建过程。


<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2025/12/2025_12_03__23_17_55_af664a.jpg"/>



Figure 3: ColorBench statistics. The results do not include actions "navigate back", "navigate home", or "open app".
图3：ColorBench 统计。结果不包含“返回”、“回到首页”或“打开应用”动作。


#### 4.2.1 Trajectory Collection
#### 4.2.1 轨迹收集


Step 1: Breadth-First Search (BFS) BFS trajectory collection focuses on capturing interactions across shallow-level pages within each application. These pages exhibit high frequency and short-distance characteristics in daily user operations, ensuring the dataset covers the most common trajectories within the target app. Specifically, we employ two VLMs: VLM $A$ handles the identification of interactive UI elements. Upon entering a new shallow-level page within the target application, VLM $A$ first identifies and stores all interactive UI elements on the page. Subsequently, VLM $B$ performs interaction operations on each identified UI element and records the resulting trajectory. This dual-model collaboration ensures accuracy and consistency in the collection process while reducing manual intervention.
步骤 1：广度优先搜索 (BFS) BFS 轨迹收集侧重捕捉每个应用内浅层页面之间的交互。这些页面在日常用户操作中具有高频次、短距离的特点，确保数据集覆盖目标应用中最常见的轨迹。具体而言，我们使用两种 VLM：VLM $A$ 负责识别交互性 UI 元素。进入目标应用的新浅层页面后，VLM $A$ 首先识别并存储页面上所有交互性 UI 元素。随后，VLM $B$ 对每个识别出的 UI 元素执行交互操作并记录由此产生的轨迹。此双模型协作在减少人工干预的同时确保了采集过程的准确性与一致性。


Step 2: Depth-First Search (DFS) DFS trajectory collection targets complex long-horizon tasks that require sustained interaction across multiple applications. Automation tools struggle to capture the complete trajectory of such tasks. To address this, we designed a trajectory collection method based on "screenshot-based action completion" and "branch trajectory supplementation." This approach effectively reduces the difficulty of collecting complex trajectories while ensuring data integrity and authenticity.
步骤 2：深度优先搜索 (DFS) DFS 轨迹收集针对需要跨多应用持续交互的复杂长时程任务。自动化工具难以捕捉此类任务的完整轨迹。为此，我们设计了基于“基于截图的动作补全”和“分支轨迹补充”的轨迹采集方法。该方法在确保数据完整性与真实性的同时，有效降低了采集复杂轨迹的难度。


Screenshot-Based Action Completion. We manually capture screenshots of each step in long-horizon tasks, then use VLM to fill in missing actions between screenshots, thereby reconstructing the complete task trajectory. The process involves three steps: 1) Task Definition and Screenshot Capture: An expert annotation team designs a set of typical, complex long-horizon tasks supporting multiple paths. For each task, trained operators manually execute it on real mobile devices, collecting multiple task completion trajectories and capturing screenshots corresponding to each action. 2) Trajectory Construction: For the acquired sequence of captured screenshots $\left( {{S}_{1},{S}_{2},\ldots ,{S}_{n}}\right)$ ,each input consecutive screenshots ${S}_{i}$ and ${S}_{i + 1}$ . The action completion model predicts all correct interaction actions ${A}_{i}$ . Subsequently,the sequences $\left( {{S}_{1} \rightarrow  {A}_{1} \rightarrow  {S}_{2} \rightarrow  {A}_{2} \rightarrow  \cdots  \rightarrow  {A}_{n - 1} \rightarrow  {S}_{n}}\right)$ are combined to form the complete trajectory. 3) Manual Verification: After the model predicts the action, the research team manually verifies all trajectories to ensure their accuracy.
基于截图的动作补全。我们手工采集长时程任务每一步的截图，然后使用 VLM 在截图间补全缺失动作，从而重构完整的任务轨迹。该过程包括三步：1) 任务定义与截图采集：专家标注团队设计一组支持多路径的典型复杂长时程任务。针对每个任务，训练有素的操作员在真实移动设备上手动执行，采集多条任务完成轨迹并捕捉对应每个动作的截图。2) 轨迹构建：对于获取的截图序列 $\left( {{S}_{1},{S}_{2},\ldots ,{S}_{n}}\right)$，每一对相邻截图 ${S}_{i}$ 和 ${S}_{i + 1}$。动作补全模型预测所有正确的交互动作 ${A}_{i}$。随后，将序列 $\left( {{S}_{1} \rightarrow  {A}_{1} \rightarrow  {S}_{2} \rightarrow  {A}_{2} \rightarrow  \cdots  \rightarrow  {A}_{n - 1} \rightarrow  {S}_{n}}\right)$ 组合以形成完整轨迹。3) 人工验证：在模型预测动作后，研究团队对所有轨迹进行人工核验以确保其准确性。


Branch Trajectory Supplementation. The trajectories obtained through the above steps represent only multiple possible operational paths when humans execute complex long-horizon tasks. Additionally, there may be some correct paths with high model click probabilities (this portion is extremely rare) and erroneous options. These branching paths constitute a crucial component of the model's potential click space, enabling a more realistic simulation of real scenarios. Therefore, we supplement the existing trajectories through the following steps: 1) Let the model perform the same tasks and compare its execution trajectories with the annotated trajectories; 2) Treat other erroneous regions clicked by the model as high-probability potential trajectory spaces, and construct corresponding branch trajectories to supplement the dataset.
分支轨迹补充。上述步骤获得的轨迹仅代表人在执行复杂长时程任务时的多种可能操作路径。此外，模型可能点击概率较高的某些正确路径（此类情况极少）和错误选项也可能存在。这些分支路径构成模型潜在点击空间的重要组成部分，使场景模拟更贴近真实。因此，我们通过以下步骤补充现有轨迹：1) 让模型执行相同任务并将其执行轨迹与标注轨迹比对；2) 将模型点击的其他错误区域视为高概率的潜在轨迹空间，并构建相应的分支轨迹以补充数据集。


#### 4.2.2 Merge Trajectories into a Graph
#### 4.2.2 将轨迹合并为图


The graph merging strategy aims to automatically construct evaluation-ready interaction graphs from quality interaction traces. It distinguishes page nodes by identifying action transitions that cause substantive interface state changes. To achieve this goal, we design a two-stage filtering pipeline based on VLMs, with the following workflow:
图合并策略旨在从高质量交互轨迹自动构建可用于评估的交互图。该策略通过识别导致实质性界面状态变化的动作转换来区分页面节点。为实现此目标，我们设计了基于 VLM 的两阶段过滤流水线，工作流程如下：


Semantics-Based Coarse Screening. First, we employ large language models to generate semantic descriptions for each screenshot, covering page layout and core functionalities. Subsequently, we compute embedding vectors for all descriptive texts and filter image pairs with semantic similarity exceeding a preset threshold through pairwise comparisons. This step efficiently narrows the candidate pool, focusing on interfaces that are highly similar in both visual appearance and functionality.
基于语义的粗筛。首先，我们使用大语言模型为每张截图生成覆盖页面布局和核心功能的语义描述。随后，对所有描述文本计算嵌入向量，并通过成对比较过滤出语义相似度超过预设阈值的图像对。此步骤高效缩小候选范围，聚焦在视觉外观和功能上高度相似的界面。


Action Transition-Based Node Discrimination. For image pairs that pass semantic filtering, we feed them into a VLM for fine-grained discrimination. Since candidate pairs are already semantically highly similar, the model's core discrimination criterion shifts from content to whether an "action transition" that drives state change exists between them. For instance, interactive operations like "Click to Follow" or "Add to Cart" may not significantly alter page layouts but trigger substantive updates to interface states, such as button status changes and product count updates. The model classifies image pairs exhibiting such action transitions as belonging to different page nodes, otherwise the same node.
基于动作转换的节点判别。对于通过语义过滤的图像对，我们将其输入 VLM 进行细粒度判别。由于候选对在语义上已高度相似，模型的核心判别标准从内容转向它们之间是否存在驱动状态变化的“动作转换”。例如，“点击关注”或“加入购物车”等交互操作可能不会显著改变页面布局，但会触发界面状态的实质性更新，如按钮状态变化或商品数量更新。模型将表现出此类动作转换的图像对判为不同页面节点，否则判为同一节点。


Manual Verification and Graph Enhancement. After automated merging, we introduced a manual verification process to further enhance the quality of the graph. This step primarily accomplishes two tasks: first, it corrects node classifications that may have been misjudged during automation; second, it supplements annotations for executable common actions across different state nodes on the same page. Ultimately, we obtained a verified, fully annotated graph that serves as the evaluation environment for ColorBench.
人工验证与图增强。在自动合并之后，我们引入了人工验证流程以进一步提升图的质量。此步骤主要完成两项任务：其一，纠正自动化过程中可能误判的节点分类；其二，补充同一页面不同状态节点上可执行的常见动作注释。最终，我们得到了经过验证、完整注释的图，作为 ColorBench 的评估环境。


## 5 Experiments
## 5 实验


In this section, we conduct extensive experiments to answer the following research questions:
在本节中，我们进行了大量实验以回答以下研究问题：


RQ1 Why graph-structured benchmark are necessary and feasible?
RQ1 为什么图结构基准是必要且可行的？


RQ2 How do existing models perform on complex long-horizon tasks?
RQ2 现有模型在复杂长时序任务中的表现如何？


RQ3 What capabilities do existing models lack in complex long-horizon tasks?
RQ3 在复杂长时序任务中，现有模型缺乏哪些能力？


RQ4 Which modules are essential for complex long-horizon tasks?
RQ4 哪些模块对于复杂长时序任务是关键的？


### 5.1 Experiment Setup
### 5.1 实验设置


#### 5.1.1 Evaluation Metrics
#### 5.1.1 评价指标


To evaluate the performance of models on ColorBench, we leverage SR (success rate) and CR (completion rate) (Liu et al., 2025b; Zhang et al., 2024a), as well as Atomic Task Capability (AC), which we propose to diagnose capability-level weaknesses that lead to agent failures. In complex long-horizon tasks, each milestone represents the completion of a subtask, and the number of milestones reached reflects the CR of the task. When all milestones are reached, the task is considered successful; otherwise, it is considered a failure. By extracting common characteristics across all milestone points, we categorize them into 15 atomic task capabilities. For each atomic task, the AC is calculated as:
为了评估模型在 ColorBench 上的表现，我们采用 SR（成功率）与 CR（完成率）（Liu et al., 2025b；Zhang et al., 2024a），以及我们提出用于诊断导致代理失败的能力层面薄弱点的原子任务能力（AC）。在复杂长时序任务中，每个里程碑表示一个子任务的完成，到达的里程碑数量反映了任务的 CR。当所有里程碑均被达成时，任务被视为成功；否则视为失败。通过提取所有里程碑点的共性，我们将其归类为 15 种原子任务能力。每个原子任务的 AC 计算如下：


$$
\mathrm{{AC}} = \frac{\# \text{ Successfully Reached milestones of an Atomic Task Executed During Evaluation }}{\# \text{ the Atomic Task Executed During Evaluation }}.
$$



The denominator excludes subtasks that were never executed due to failure in preceding subtasks.
分母排除了因前置子任务失败而未被执行的子任务。


#### 5.1.2 Baselines
#### 5.1.2 基线


We evaluated Colorbench on open-source models and closed-source models commonly used in mobile GUI agents. Common open-source models for GUI tasks can be categorized into two types: general VLMs such as the Qwen-VL (Bai et al., 2023; Team, 2025) series, and specialized foundation models fine-tuned on extensive GUI data based on general VLMs, including UI-TARS (Qin et al., 2025), OS-Atlas (Wu et al., 2024b), and GUI-OWL series (Ye et al., 2025a). Turning to closed-source VLMs, we carefully selected three models: Qwen-VL Max (Team, 2023), GLM-4.5V (AI, 2024), and GPT-40 (OpenAI, 2024), which are commonly used in mobile GUI agents. Due to the poor grounding capabilities of closed-source models, we employed Qwen2.5-VL-7b (Bai et al., 2025) as the grounding model to identify the target UI element for click and long-press actions.
我们在移动 GUI 代理中常用的开源与闭源模型上评估了 ColorBench。常见的 GUI 任务开源模型可分为两类：通用 VLM（如 Qwen-VL（Bai et al., 2023；Team, 2025）系列）和基于通用 VLM 在大量 GUI 数据上微调的专用基础模型，包括 UI-TARS（Qin et al., 2025）、OS-Atlas（Wu et al., 2024b）和 GUI-OWL 系列（Ye et al., 2025a）。在闭源 VLM 方面，我们精心挑选了三款常用于移动 GUI 代理的模型：Qwen-VL Max（Team, 2023）、GLM-4.5V（AI, 2024）和 GPT-40（OpenAI, 2024）。鉴于闭源模型定位能力较弱，我们采用 Qwen2.5-VL-7b（Bai et al., 2025）作为定位模型以识别点击和长按动作的目标 UI 元素。


Moreover, we evaluated two common approaches: 1) we finetuned the base GUI-OWL (Ye et al., 2025a) model using our static Chinese app training dataset to supplement its knowledge of Chinese applications; 2) we designed a simple multi-agent system incorporating planning, reflection, and memory modules. To investigate the function of these modules for solving complex long-horizon tasks, we conducted ablation experiments on them. Comprehensively considering the model's professional and general ability, we selected Qwen2.5-VL-32B (Bai et al., 2025) and GUI-OWL-32B (Ye et al., 2025a) for the ablation study. Due to the page limitation, we further provide implementation details in Appendix B.1.
此外，我们评估了两种常见方法：1）使用我们静态的中文应用训练集对基础 GUI-OWL（Ye et al., 2025a）模型进行微调，以补充其对中文应用的知识；2）设计了一个包含规划、反思和记忆模块的简单多智能体系统。为探究这些模块在解决复杂长时序任务中的作用，我们对它们进行了消融实验。综合考虑模型的专业性与通用能力，我们选取了 Qwen2.5-VL-32B（Bai et al., 2025）和 GUI-OWL-32B（Ye et al., 2025a）进行消融研究。由于篇幅限制，进一步的实现细节见附录 B.1。


### 5.2 Significance of Graph Structure (RQ1)
### 5.2 图结构的重要性（RQ1）


The real-world mobile environment is a vast, strongly connected directed graph with an infinite number of nodes and edges, exhibiting immense temporal randomness simultaneously. Replicating such an environment one-to-one is impractical. Given this, a critical question arises: Can graph-structured evaluation datasets effectively simulate dynamic testing environments to bridge the gap between dynamic and static evaluations?
真实的移动环境是一个巨大的、强连通的有向图，节点与边数量近乎无限，并同时呈现巨大的时间随机性。逐一复刻这样的环境并不可行。在此背景下，一个关键问题是：图结构的评估数据集能否有效模拟动态测试环境，从而弥合动态评测与静态评测之间的差距？


<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2025/12/2025_12_03__23_17_55_a002a3.jpg"/>



Figure 4: Statistics on the number of actions that all experienced models at each node in the graph tend to perform, do not include "navigate back", "navigate home" and "open app" which can be executed on any interface.
图 4：统计所有模型在图中各节点倾向执行的动作数量，不包括可在任意界面执行的“后退”、“回到主屏”和“打开应用”。


To address this question, we conducted two statistical experiments: first, we quantified the potential execution paths for the model operating in a real environment, and second, we compared the outcomes of identical tasks evaluated on a physical device versus on a graph. Specifically, for the first experiment, we analyzed the execution trajectories of all models on ColorBench. By calculating the potential action space at each node (excluding "navigate back" and "home" actions), we determined the number of successor nodes required at that node in the graph. As shown in Figure 4, the maximum value is 17 corresponding to the APP's home page, and the minimum value
为了解答该问题，我们进行了两项统计实验：首先，我们量化了模型在真实环境中可能的执行路径；其次，我们对比了同一任务在实体设备与图上评测的结果。具体而言，在第一项实验中，我们分析了所有模型在 ColorBench 上的执行轨迹。通过计算每个节点的潜在动作空间（不含“后退”和“回到主屏”动作），我们确定了该节点在图中所需的后继节点数量。如图 4 所示，最大值为 17，对应于 APP 的主页，最小值


Table 2: Performance comparison of closed-source and open-source models on our Col-orBench. SR means task success rate and CR means task completion rate (proportion of completed subtasks). Bold represents optimal performance, while underlining represents suboptimal. The results of atomic capabilities are presented in Table 5 in the appendix.
表 2：闭源与开源模型在我们 Col-orBench 上的性能对比。SR 表示任务成功率，CR 表示任务完成率（已完成子任务的比例）。加粗表示最佳表现，下划线表示次优。原子能力的结果见附录表 5。


<table><tr><td rowspan="2">Baseline</td><td rowspan="2">Model</td><td colspan="2">Single APP</td><td colspan="2">Cross APP</td><td colspan="2">Average</td></tr><tr><td>SR(%)</td><td>CR(%)</td><td>SR(%)</td><td>CR(%)</td><td>SR(%)</td><td>CR(%)</td></tr><tr><td rowspan="3">Closed-source Model with Grounding</td><td>GPT-40</td><td>20.27</td><td>27.59</td><td>11.88</td><td>21.53</td><td>15.43</td><td>24.10</td></tr><tr><td>Qwen-VL Max</td><td>20.27</td><td>32.55</td><td>14.85</td><td>29.42</td><td>17.14</td><td>30.74</td></tr><tr><td>GLM-4.5V</td><td>36.49</td><td>56.64</td><td>23.23</td><td>47.72</td><td>28.57</td><td>51.54</td></tr><tr><td rowspan="10">Open-source Model</td><td>OS-Atlas-Pro-7B</td><td>10.81</td><td>17.91</td><td>3.96</td><td>19.87</td><td>6.86</td><td>19.04</td></tr><tr><td>UI-TARS-1.5-7B</td><td>9.46</td><td>16.89</td><td>0.00</td><td>13.88</td><td>4.00</td><td>15.15</td></tr><tr><td>UI-TARS-7B-DPO</td><td>8.11</td><td>12.05</td><td>0.99</td><td>11.82</td><td>4.00</td><td>11.91</td></tr><tr><td>GUI-OWL-7B</td><td>25.68</td><td>39.19</td><td>16.83</td><td>34.16</td><td>20.57</td><td>36.29</td></tr><tr><td>Qwen2.5-VL-7B</td><td>22.97</td><td>34.80</td><td>9.90</td><td>32.57</td><td>15.43</td><td>33.51</td></tr><tr><td>GUI-OWL-32B</td><td>36.49</td><td>47.52</td><td>22.77</td><td>39.31</td><td>28.57</td><td>42.78</td></tr><tr><td>Qwen2.5-VL-32B</td><td>24.32</td><td>36.49</td><td>11.88</td><td>29.62</td><td>17.14</td><td>32.53</td></tr><tr><td>UI-TARS-72B-DPO</td><td>33.78</td><td>46.85</td><td>23.76</td><td>45.40</td><td>28.00</td><td>46.01</td></tr><tr><td>Qwen2.5-VL-72B</td><td>21.62</td><td>34.23</td><td>16.83</td><td>32.38</td><td>18.86</td><td>33.16</td></tr><tr><td>Qwen3-VL-235B-A22B-Instruct</td><td>35.14</td><td>49.66</td><td>27.72</td><td>43.42</td><td>30.86</td><td>46.06</td></tr><tr><td rowspan="2">Fine-tuned with ColorBench-train</td><td>GUI-OWL-7B-RL</td><td>31.08</td><td>46.62</td><td>19.80</td><td>41.79</td><td>24.57</td><td>43.83</td></tr><tr><td>GUI-OWL-32B-RL</td><td>40.54</td><td>53.72</td><td>26.73</td><td>42.25</td><td>32.57</td><td>47.10</td></tr><tr><td rowspan="2">Multi-Agent System</td><td>GUI-OWL-32B</td><td>43.24</td><td>55.86</td><td>22.77</td><td>43.40</td><td>31.43</td><td>48.67</td></tr><tr><td>Qwen2.5-VL-32B</td><td>40.54</td><td>55.32</td><td>20.79</td><td>40.18</td><td>29.14</td><td>46.58</td></tr></table>
<table><tbody><tr><td rowspan="2">基线</td><td rowspan="2">模型</td><td colspan="2">单应用</td><td colspan="2">跨应用</td><td colspan="2">平均</td></tr><tr><td>SR(%)</td><td>CR(%)</td><td>SR(%)</td><td>CR(%)</td><td>SR(%)</td><td>CR(%)</td></tr><tr><td rowspan="3">闭源带定位模型</td><td>GPT-40</td><td>20.27</td><td>27.59</td><td>11.88</td><td>21.53</td><td>15.43</td><td>24.10</td></tr><tr><td>Qwen-VL Max</td><td>20.27</td><td>32.55</td><td>14.85</td><td>29.42</td><td>17.14</td><td>30.74</td></tr><tr><td>GLM-4.5V</td><td>36.49</td><td>56.64</td><td>23.23</td><td>47.72</td><td>28.57</td><td>51.54</td></tr><tr><td rowspan="10">开源模型</td><td>OS-Atlas-Pro-7B</td><td>10.81</td><td>17.91</td><td>3.96</td><td>19.87</td><td>6.86</td><td>19.04</td></tr><tr><td>UI-TARS-1.5-7B</td><td>9.46</td><td>16.89</td><td>0.00</td><td>13.88</td><td>4.00</td><td>15.15</td></tr><tr><td>UI-TARS-7B-DPO</td><td>8.11</td><td>12.05</td><td>0.99</td><td>11.82</td><td>4.00</td><td>11.91</td></tr><tr><td>GUI-OWL-7B</td><td>25.68</td><td>39.19</td><td>16.83</td><td>34.16</td><td>20.57</td><td>36.29</td></tr><tr><td>Qwen2.5-VL-7B</td><td>22.97</td><td>34.80</td><td>9.90</td><td>32.57</td><td>15.43</td><td>33.51</td></tr><tr><td>GUI-OWL-32B</td><td>36.49</td><td>47.52</td><td>22.77</td><td>39.31</td><td>28.57</td><td>42.78</td></tr><tr><td>Qwen2.5-VL-32B</td><td>24.32</td><td>36.49</td><td>11.88</td><td>29.62</td><td>17.14</td><td>32.53</td></tr><tr><td>UI-TARS-72B-DPO</td><td>33.78</td><td>46.85</td><td>23.76</td><td>45.40</td><td>28.00</td><td>46.01</td></tr><tr><td>Qwen2.5-VL-72B</td><td>21.62</td><td>34.23</td><td>16.83</td><td>32.38</td><td>18.86</td><td>33.16</td></tr><tr><td>Qwen3-VL-235B-A22B-Instruct</td><td>35.14</td><td>49.66</td><td>27.72</td><td>43.42</td><td>30.86</td><td>46.06</td></tr><tr><td rowspan="2">使用 ColorBench-train 微调</td><td>GUI-OWL-7B-RL</td><td>31.08</td><td>46.62</td><td>19.80</td><td>41.79</td><td>24.57</td><td>43.83</td></tr><tr><td>GUI-OWL-32B-RL</td><td>40.54</td><td>53.72</td><td>26.73</td><td>42.25</td><td>32.57</td><td>47.10</td></tr><tr><td rowspan="2">多智能体系统</td><td>GUI-OWL-32B</td><td>43.24</td><td>55.86</td><td>22.77</td><td>43.40</td><td>31.43</td><td>48.67</td></tr><tr><td>Qwen2.5-VL-32B</td><td>40.54</td><td>55.32</td><td>20.79</td><td>40.18</td><td>29.14</td><td>46.58</td></tr></tbody></table>


is 0 , indicating a task endpoint. The average value is 1.9 , reflecting the constructibility of the graph. Consequently, we have demonstrated that the variety of trajectories and screen states that models experienced in real-world evaluations is limited, which validates both the conceptual soundness and practical feasibility of our graph-structured benchmark.
值为0，表示任务终点。平均值为1.9，反映了图的可构建性。因此，我们证明了模型在真实世界评估中经历的轨迹与屏幕状态的多样性有限，这验证了我们图结构基准的概念合理性和实践可行性。


For the second experiment, we selected a subset of executable tasks from ColorBench for real-device testing, adapting to the dynamically changing real-world information. Partial results are presented in Table 6 in the appendix. These examples demonstrate that ColorBench covers the primary causes of task failure in real-device evaluations. Therefore, we can assess the authenticity of the graph and demonstrate that a graph-structured benchmark can effectively reduce the gap between offline and online assessments. In fact, when constructing the graph, some of the erroneous trajectories supported by the graph originate from the model's actual mistakes, which essentially creates a customized testing environment for the model and further ensures the validity of the graph.
在第二个实验中，我们从 ColorBench 中选择了一部分可执行任务在真实设备上测试，以适应动态变化的真实世界信息。部分结果见附录的表6。这些示例表明 ColorBench 覆盖了真实设备评估中任务失败的主要原因。因此，我们可以评估图的真实性，并证明图结构基准能有效缩小离线与在线评估的差距。事实上，在构建图时，图所支持的一些错误轨迹源自模型的真实错误，这本质上为模型创建了定制化的测试环境，进一步确保了图的有效性。


### 5.3 Overall Performance (RQ2)
### 5.3 整体表现（RQ2）


Table 2 presents the main experimental results. For closed-source models, we observe that:
表2给出主要实验结果。对于闭源模型，我们观察到：


- GLM-4.5V significantly outperforms both Qwen-VL Max and GPT-4o. Although its SR is not the highest, its notably high CR demonstrates stronger capabilities in comprehending and planning for complex, long-horizon tasks. Furthermore, Table 5 in the appendix indicates that GLM-4.5V can actively memorize essential historical information, an ability that surpasses most compared models. In contrast, GPT-40 underperforms due to a lack of training on relevant operational knowledge, while Qwen-VL Max exhibits deficiencies in decomposing and planning complex tasks. Additionally, all three models demonstrate generally weak UI grounding abilities.
- GLM-4.5V 显著优于 Qwen-VL Max 与 GPT-4o。尽管其 SR 不是最高，但其明显更高的 CR 展示了在理解与规划复杂长航时任务方面的更强能力。此外，附录表5显示 GLM-4.5V 能主动记忆重要历史信息，这一能力超过大多数对比模型。相反，GPT-4o 表现欠佳，原因是缺乏相关操作知识的训练，而 Qwen-VL Max 在分解与规划复杂任务方面存在不足。另外，三者的 UI 落地能力普遍较弱。


- A lack of knowledge about mobile phone operation severely impedes task execution. For instance, despite its powerful multi-modal understanding, GPT-40's unfamiliarity with basic mobile actions (i.e., copying, pasting, and sharing) often results in it providing only high-level plans without being able to execute the specific operational steps required.
- 对手机操作知识的缺乏严重阻碍了任务执行。例如，尽管 GPT-4o 具备强大的多模态理解，但对基本手机操作（如复制、粘贴与分享）不熟悉，经常只能给出高层计划而无法执行所需的具体操作步骤。


As for open-source models, we can draw the following observations from Table 2:
对于开源模型，从表2我们可以得出以下观察：


- Models with larger parameter scales generally achieve better performance on complex long-horizon tasks. For instance, both the GUI-OWL and Qwen series show improved results as the model size increases, and their largest models, alongside GLM-4.5V, are the only three capable of actively memorizing essential historical information, as shown in Table 5 in the appendix. This stems from their stronger general capabilities, enabling better comprehension and planning for complex tasks.
- 参数规模更大的模型在复杂长航时任务上通常表现更好。例如，GUI-OWL 与 Qwen 系列随着模型规模增大结果改善，它们的最大型号与 GLM-4.5V 是附录表5中唯一能主动记忆重要历史信息的三者。这来自于其更强的通用能力，使其能更好地理解与规划复杂任务。


- Specialized foundation models do not necessarily outperform general-purpose models. Fine-tuning can easily lead to overfitting, reducing their ability to generalize to complex long-horizon tasks, as evidenced by the results of UI-TARS and OS-Atlas-Pro (7B). In contrast, the strong performance of the GUI-OWL series indicates that specialized foundation models remain promising and warrant further exploration.
- 专用基础模型不一定优于通用模型。微调容易导致过拟合，降低其在复杂长航时任务上的泛化能力，如 UI-TARS 与 OS-Atlas-Pro (7B) 的结果所示。相反，GUI-OWL 系列的强劲表现表明专用基础模型仍然有前景，值得进一步探索。


- The foundational mobile GUI abilities of these models remain unstable. Although the open-source models have been trained to varying degrees on mobile GUI operations, we still observe issues such as execution step errors, recognition and grounding deviations, and instruction-following failures. Therefore, basic mobile operation capabilities still require improvement. Comparisons with closed-source models further indicate that domain-specific knowledge forms a critical foundation for successfully accomplishing these tasks.
- 这些模型的基础移动 GUI 能力仍不稳定。尽管开源模型在不同程度上接受过移动 GUI 操作训练，我们仍观察到执行步骤错误、识别与落地偏差以及指令执行失败等问题。因此，基础移动操作能力仍需提升。与闭源模型的比较进一步表明，领域特定知识是成功完成这些任务的重要基础。


As shown in Table 5, models fine-tuned with app-specific data demonstrate higher task accuracy due to their improved familiarity with application layouts, content, and functionality. In multi-agent systems, decomposing complex tasks into structured modules with enforced execution further enhances performance reliability. Additionally, the appendix records each model's atomic capability scores, which directly influence the success or failure of the task. Further analysis of these atomic capabilities is provided in Appendix B.2.
如表5所示，经应用特定数据微调的模型因更熟悉应用布局、内容与功能而表现出更高的任务准确率。在多智能体系统中，将复杂任务分解为结构化模块并强制执行，进一步提高了性能可靠性。此外，附录记录了每个模型的原子能力得分，这些直接影响任务的成败。对这些原子能力的进一步分析见附录 B.2。


### 5.4 Key Capabilities (RQ3)
### 5.4 关键能力（RQ3）


We manually examined the task execution logs of each model and analyzed their performance in essential high-level cognitive capabilities (those requiring reasoning). We identified the following common issues in existing models: incomplete decomposition of complex long-horizon tasks, vague memory of essential historical information, and a lack of effective reflection on recurring erroneous actions.
我们手动检查了各模型的任务执行日志，并分析了它们在需要推理的高层认知能力上的表现。我们识别出现有模型的以下常见问题：对复杂长航时任务分解不完整、对重要历史信息记忆模糊，以及对重复错误操作缺乏有效反思。


For example, in the task "Go to Xiaohongshu to search for 'AI Agent', share the paper on 'optimized memory management' with WeChat Contact 1, then go to Baidu Arxiv to search and download the PDF of that paper, and finally send it to Contact 1", some models incorrectly considered the task complete after executing the "share with contact" subtask. Other models reached the Arxiv website but failed to retain the specific paper title identified earlier on Xiaohongshu, causing the task to fall into an infinite loop. Most models were unable to recognize their own errors in these situations.
例如，在任务“去小红书搜索‘AI Agent’，把关于‘优化内存管理’的论文分享给微信联系人1，然后去百度 Arxiv 搜索并下载该论文的 PDF，最后发送给联系人1”中，一些模型在执行“与联系人分享”子任务后错误地认为任务完成。其他模型到达 Arxiv 网站但未能保留此前在小红书上确定的具体论文标题，导致任务陷入无限循环。大多数模型在这些情况下无法识别自己的错误。


In conclusion, to handle complex long-horizon tasks, models require not only basic GUI capabilities but also the following key advanced competencies: the ability to comprehend and analyze complex task requirements, the capacity to decompose long-horizon tasks into structured subtasks, the capability to actively memorize and recall critical information across tasks, and the faculty for reflection and self-correction. Therefore, the accurate execution of atomic tasks is the foundation for solving complex long-horizon tasks, while high-quality planning, reflection, and memory are the critical support that effectively string these atomic tasks together and ensure their stable execution.
总之，为了处理复杂的长时序任务，模型不仅需要基本的 GUI 能力，还需具备以下关键高级能力：理解与分析复杂任务需求的能力、将长时序任务分解为结构化子任务的能力、在任务间主动记忆与召回关键信息的能力，以及反思与自我修正的能力。因此，原子任务的准确执行是解决复杂长时序任务的基础，而高质量的规划、反思与记忆是将这些原子任务有效串联并确保其稳定执行的关键支撑。


### 5.5 Ablation Study (RQ4)
### 5.5 消融研究 (RQ4)


Ablation studies on individual modules reveal their distinct contributions to solving complex long-horizon tasks. Table 3, while each module improves overall performance, the extent of this improvement varies across models. For instance, introducing only the reflection module brings substantial gains for Qwen-2.5-VL-32B but limited improvement for GUI-Owl-32B, reflecting differences in their inherent capabilities.
对各个模块的消融研究揭示了它们在解决复杂长时序任务方面的不同贡献。表 3 显示，虽然每个模块都能提升整体性能，但不同模型的提升幅度各不相同。例如，仅加入反思模块对 Qwen-2.5-VL-32B 带来了显著提升，而对 GUI-Owl-32B 的改进则有限，反映出它们固有能力的差异。


Table 3: Ablation experiment results of the multi-agent system modules on our Color-Bench. The numbers in the footnote indicate changes relative to the baseline.
表 3：多智能体系统模块在我们 Color-Bench 上的消融实验结果。脚注中的数字表示相对于基线的变化。


<table><tr><td rowspan="2">Base Model</td><td colspan="3">Module</td><td colspan="2">Single APP</td><td colspan="2">Cross APP</td><td colspan="2">Average</td></tr><tr><td>Plan</td><td>Reflection</td><td>Memory</td><td>SR(%)</td><td>CR(%)</td><td>SR(%)</td><td>CR(%)</td><td>SR(%)</td><td>CR(%)</td></tr><tr><td rowspan="6">Qwen2.5-VL (32B)</td><td>✘</td><td>✘</td><td>✘</td><td>24.32</td><td>36.49</td><td>11.88</td><td>29.62</td><td>17.14</td><td>32.53</td></tr><tr><td>✓</td><td>✘</td><td>✘</td><td>32.43+8.11</td><td>46.28+9.79</td><td>17.82 +5.94</td><td>35.54+5.92</td><td>24.00 +6.86</td><td>40.09+7.56</td></tr><tr><td>✘</td><td>✓</td><td>✘</td><td>37.84+13.52</td><td>48.65+12.16</td><td>15.84 +3.96</td><td>33.48+3.86</td><td>25.14+8.00</td><td>39.90+7.37</td></tr><tr><td>✘</td><td>✘</td><td>✓</td><td>31.08+6.76</td><td>47.52+11.03</td><td>18.81 +6.93</td><td>35.30+5.68</td><td>24.00+6.86</td><td>40.47+7.94</td></tr><tr><td>✓</td><td>✓</td><td>✘</td><td>35.14+10.82</td><td>51.35+14.86</td><td>19.80+7.92</td><td>39.01+9.39</td><td>26.29+9.15</td><td>44.23+11.70</td></tr><tr><td>✓</td><td>✓</td><td>✓</td><td>40.54</td><td>55.32+18.83</td><td>20.79+8.91</td><td>40.18+10.56</td><td>29.14+12.00</td><td>46.58+14.05</td></tr><tr><td rowspan="6">GUI-OWL (32B)</td><td>✘</td><td>✘</td><td>✘</td><td>36.49</td><td>47.52</td><td>22.77</td><td>39.31</td><td>28.57</td><td>42.78</td></tr><tr><td>✓</td><td>✘</td><td>✘</td><td>${35.14}_{-{1.35}}$</td><td>49.66+2.14</td><td>22.770.00</td><td>46.03+6.72</td><td>28.00 -0.57</td><td>47.56+4.78</td></tr><tr><td>✘</td><td>✓</td><td>✘</td><td>${37.84}_{+{1.35}}$</td><td>51.35+3.83</td><td>27.55+4.78</td><td>46.26+6.95</td><td>31.43+2.86</td><td>48.45+5.67</td></tr><tr><td>✘</td><td>✘</td><td>✓</td><td>36.490.00</td><td>49.55+2.03</td><td>25.74+2.97</td><td>43.86+4.55</td><td>30.29+1.72</td><td>46.27+3.49</td></tr><tr><td>✓</td><td>✓</td><td>✘</td><td>41.89+5.40</td><td>51.91+4.39</td><td>22.770.00</td><td>44.82+5.51</td><td>30.86+2.29</td><td>47.82+5.04</td></tr><tr><td>✓</td><td>✓</td><td>✓</td><td>43.24+6.75</td><td>55.86+8.34</td><td>22.770.00</td><td>43.40+4.09</td><td>31.43+2.86</td><td>48.67+5.89</td></tr></table>
<table><tbody><tr><td rowspan="2">基础模型</td><td colspan="3">模块</td><td colspan="2">单一应用</td><td colspan="2">跨应用</td><td colspan="2">平均值</td></tr><tr><td>计划</td><td>反思</td><td>记忆</td><td>SR(%)</td><td>CR(%)</td><td>SR(%)</td><td>CR(%)</td><td>SR(%)</td><td>CR(%)</td></tr><tr><td rowspan="6">Qwen2.5-VL (32B)</td><td>✘</td><td>✘</td><td>✘</td><td>24.32</td><td>36.49</td><td>11.88</td><td>29.62</td><td>17.14</td><td>32.53</td></tr><tr><td>✓</td><td>✘</td><td>✘</td><td>32.43+8.11</td><td>46.28+9.79</td><td>17.82 +5.94</td><td>35.54+5.92</td><td>24.00 +6.86</td><td>40.09+7.56</td></tr><tr><td>✘</td><td>✓</td><td>✘</td><td>37.84+13.52</td><td>48.65+12.16</td><td>15.84 +3.96</td><td>33.48+3.86</td><td>25.14+8.00</td><td>39.90+7.37</td></tr><tr><td>✘</td><td>✘</td><td>✓</td><td>31.08+6.76</td><td>47.52+11.03</td><td>18.81 +6.93</td><td>35.30+5.68</td><td>24.00+6.86</td><td>40.47+7.94</td></tr><tr><td>✓</td><td>✓</td><td>✘</td><td>35.14+10.82</td><td>51.35+14.86</td><td>19.80+7.92</td><td>39.01+9.39</td><td>26.29+9.15</td><td>44.23+11.70</td></tr><tr><td>✓</td><td>✓</td><td>✓</td><td>40.54</td><td>55.32+18.83</td><td>20.79+8.91</td><td>40.18+10.56</td><td>29.14+12.00</td><td>46.58+14.05</td></tr><tr><td rowspan="6">GUI-OWL (32B)</td><td>✘</td><td>✘</td><td>✘</td><td>36.49</td><td>47.52</td><td>22.77</td><td>39.31</td><td>28.57</td><td>42.78</td></tr><tr><td>✓</td><td>✘</td><td>✘</td><td>${35.14}_{-{1.35}}$</td><td>49.66+2.14</td><td>22.770.00</td><td>46.03+6.72</td><td>28.00 -0.57</td><td>47.56+4.78</td></tr><tr><td>✘</td><td>✓</td><td>✘</td><td>${37.84}_{+{1.35}}$</td><td>51.35+3.83</td><td>27.55+4.78</td><td>46.26+6.95</td><td>31.43+2.86</td><td>48.45+5.67</td></tr><tr><td>✘</td><td>✘</td><td>✓</td><td>36.490.00</td><td>49.55+2.03</td><td>25.74+2.97</td><td>43.86+4.55</td><td>30.29+1.72</td><td>46.27+3.49</td></tr><tr><td>✓</td><td>✓</td><td>✘</td><td>41.89+5.40</td><td>51.91+4.39</td><td>22.770.00</td><td>44.82+5.51</td><td>30.86+2.29</td><td>47.82+5.04</td></tr><tr><td>✓</td><td>✓</td><td>✓</td><td>43.24+6.75</td><td>55.86+8.34</td><td>22.770.00</td><td>43.40+4.09</td><td>31.43+2.86</td><td>48.67+5.89</td></tr></tbody></table>


We also observed nuanced interactions between modules. When Qwen-2.5-VL-32B evolves from using only reflection to incorporating both reflection and planning, SR on single-app tasks decreases. Meanwhile, CR increases and remains higher than when planning alone is used. Conversely, GUI-Owl-32B's CR on cross-app tasks declines as more modules are added, though it still exceeds the baseline. This highlights instability introduced by multi-agent systems. Overly complex module combinations can cause erroneous coupling and error accumulation. As a result, tasks that were initially solvable may become unresolvable.
我们还观察到模块之间的细微交互。当 Qwen-2.5-VL-32B 从仅使用反思演进到同时采用反思与规划时，单应用任务的 SR 降低。同时，CR 增加并保持高于仅使用规划时的水平。相反，GUI-Owl-32B 在跨应用任务上的 CR 随着更多模块的加入而下降，但仍高于基线。这凸显了多代理系统引入的不稳定性。过于复杂的模块组合会导致错误耦合和错误累积，从而使原本可解的任务变得不可解。


We attribute this to imbalanced capability distributions, where weaker modules can inhibit stronger ones. This occurs because increased low-quality information raises systemic entropy, while high-quality signals are underutilized. Therefore, multi-agent approaches to long-horizon tasks require not only individual module effectiveness but also balanced integration - otherwise, they risk degrading performance.
我们将此归因于能力分布不平衡，较弱的模块会抑制较强模块的发挥。其原因在于低质量信息的增加提高了系统熵，而高质量信号未被充分利用。因此，应对长时序任务的多代理方法不仅需要各模块具备有效性，还需平衡整合——否则存在性能下降的风险。


## 6 Conclusion
## 6 结论


To bridge the gap between offline static evaluation and online dynamic evaluation for Mobile GUI Agents, we propose a graph-structured benchmark with an effective construction methodology. Based on this framework, we develop ColorBench, a benchmark specifically designed for complex long-horizon tasks that balances static stability and dynamic randomness through finite-state modeling of a dynamic environment. ColorBench supports multiple valid solutions for individual tasks and extends evaluation from step-level and result-level metrics to atomic task capability assessment, enabling effective diagnosis of model deficiencies. We extensively evaluate ColorBench across numerous models, validating the necessity and feasibility of the graph-based benchmark. Based on experimental results, we analyze limitations of existing models and provide concrete suggestions and potential approaches to enhance agents' capabilities in solving complex long-horizon problems.
为弥合移动 GUI 代理的离线静态评估与在线动态评估之间的差距，我们提出了一个图结构基准及其有效构建方法论。在此框架下，我们构建了 ColorBench，这一基准针对复杂长时序任务设计，通过对动态环境的有限状态建模在静态稳定性与动态随机性之间取得平衡。ColorBench 支持单个任务的多种有效解，并将评估从步级与结果级指标扩展到原子任务能力评估，从而实现对模型缺陷的有效诊断。我们在大量模型上对 ColorBench 进行了广泛评估，验证了基于图的基准的必要性与可行性。基于实验结果，我们分析了现有模型的局限并提出了具体建议与可行方案，以增强代理在解决复杂长时序问题上的能力。


## References
## 参考文献


Zhipu AI. Glm-4.5v technical report. https://zhipu.ai/en/blog/glm-4-5v, 2024. Accessed: 2024-12-19.
Zhipu AI. Glm-4.5v technical report. https://zhipu.ai/en/blog/glm-4-5v, 2024. Accessed: 2024-12-19.


Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen-vl: A versatile vision-language model for understanding, localization, text reading, and beyond, 2023. URL https://arxiv.org/abs/2308.12966.
Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen-vl: A versatile vision-language model for understanding, localization, text reading, and beyond, 2023. URL https://arxiv.org/abs/2308.12966.


Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923, 2025.
Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923, 2025.


Samaresh Bera, Sudip Misra, and Mohammad S Obaidat. Mobi-flow: Mobility-aware adaptive flow-rule placement in software-defined access network. IEEE Transactions on Mobile Computing, 18 (8):1831-1842, 2018.
Samaresh Bera, Sudip Misra, and Mohammad S Obaidat. Mobi-flow: Mobility-aware adaptive flow-rule placement in software-defined access network. IEEE Transactions on Mobile Computing, 18 (8):1831-1842, 2018.


Gongwei Chen, Xurui Zhou, Rui Shao, Yibo Lyu, Kaiwen Zhou, Shuai Wang, Wentao Li, Yinchuan Li, Zhongang Qi, and Liqiang Nie. Less is more: Empowering gui agent with context-aware simplification. arXiv preprint arXiv:2507.03730, 2025a.
Gongwei Chen, Xurui Zhou, Rui Shao, Yibo Lyu, Kaiwen Zhou, Shuai Wang, Wentao Li, Yinchuan Li, Zhongang Qi, and Liqiang Nie. Less is more: Empowering gui agent with context-aware simplification. arXiv preprint arXiv:2507.03730, 2025a.


Jingxuan Chen, Derek Yuen, Bin Xie, Yuhao Yang, Gongwei Chen, Zhihao Wu, Li Yixing, Xurui Zhou, Weiwen Liu, Shuai Wang, et al. Spa-bench: A comprehensive benchmark for smartphone agent evaluation. In NeurIPS 2024 Workshop on Open-World Agents, 2024a.
Jingxuan Chen, Derek Yuen, Bin Xie, Yuhao Yang, Gongwei Chen, Zhihao Wu, Li Yixing, Xurui Zhou, Weiwen Liu, Shuai Wang, et al. Spa-bench: A comprehensive benchmark for smartphone agent evaluation. In NeurIPS 2024 Workshop on Open-World Agents, 2024a.


Weizhi Chen, Ziwei Wang, Leyang Yang, Sheng Zhou, Xiaoxuan Tang, Jiajun Bu, Yong Li, and Wei Jiang. Pg-agent: An agent powered by page graph. arXiv preprint arXiv:2509.03536, 2025b.
Weizhi Chen, Ziwei Wang, Leyang Yang, Sheng Zhou, Xiaoxuan Tang, Jiajun Bu, Yong Li, and Wei Jiang. Pg-agent: An agent powered by page graph. arXiv preprint arXiv:2509.03536, 2025b.


Wentong Chen, Junbo Cui, Jinyi Hu, Yujia Qin, Junjie Fang, Yue Zhao, Chongyi Wang, Jun Liu, Guirong Chen, Yupeng Huo, et al. Guicourse: From general vision language models to versatile gui agents. arXiv preprint arXiv:2406.11317, 2024b.
Wentong Chen, Junbo Cui, Jinyi Hu, Yujia Qin, Junjie Fang, Yue Zhao, Chongyi Wang, Jun Liu, Guirong Chen, Yupeng Huo, et al. Guicourse: From general vision language models to versatile gui agents. arXiv preprint arXiv:2406.11317, 2024b.


Yurun Chen, Xavier Hu, Yuhan Liu, Keting Yin, Juncheng Li, Zhuosheng Zhang, and Shengyu Zhang. Harmonyguard: Toward safety and utility in web agents via adaptive policy enhancement and dual-objective optimization. arXiv preprint arXiv:2508.04010, 2025c.
Yurun Chen, Xavier Hu, Yuhan Liu, Keting Yin, Juncheng Li, Zhuosheng Zhang, and Shengyu Zhang. Harmonyguard: Toward safety and utility in web agents via adaptive policy enhancement and dual-objective optimization. arXiv preprint arXiv:2508.04010, 2025c.


Pengzhou Cheng, Zheng Wu, Zongru Wu, Aston Zhang, Zhuosheng Zhang, and Gongshen Liu. Os-kairos: Adaptive interaction for mllm-powered gui agents. arXiv preprint arXiv:2503.16465, 2025.
Pengzhou Cheng, Zheng Wu, Zongru Wu, Aston Zhang, Zhuosheng Zhang, and Gongshen Liu. Os-kairos：用于由 mllm 驱动的 GUI 代理的自适应交互。arXiv preprint arXiv:2503.16465, 2025.


Gaole Dai, Shiqi Jiang, Ting Cao, Yuanchun Li, Yuqing Yang, Rui Tan, Mo Li, and Lili Qiu. Advancing mobile gui agents: A verifier-driven approach to practical deployment. arXiv preprint arXiv:2503.15937, 2025.
Gaole Dai, Shiqi Jiang, Ting Cao, Yuanchun Li, Yuqing Yang, Rui Tan, Mo Li, and Lili Qiu. 推进移动 GUI 代理：一种面向实用部署的验证器驱动方法。arXiv preprint arXiv:2503.15937, 2025.


Biplab Deka, Zifeng Huang, Chad Franzen, Joshua Hibschman, Daniel Afergan, Yang Li, Jeffrey Nichols, and Ranjitha Kumar. Rico: A mobile app dataset for building data-driven design applications. In Proceedings of the 30th annual ACM symposium on user interface software and technology, pp. 845-854, 2017.
Biplab Deka, Zifeng Huang, Chad Franzen, Joshua Hibschman, Daniel Afergan, Yang Li, Jeffrey Nichols, and Ranjitha Kumar. Rico：用于构建数据驱动设计应用的移动应用数据集。收录于第 30 届 ACM 用户界面软件与技术年度研讨会论文集，页 845-854，2017.


Xiang Deng, Yu Gu, Boyuan Zheng, Shijie Chen, Sam Stevens, Boshi Wang, Huan Sun, and Yu Su. Mind2web: Towards a generalist agent for the web. Advances in Neural Information Processing Systems, 36:28091-28114, 2023.
Xiang Deng, Yu Gu, Boyuan Zheng, Shijie Chen, Sam Stevens, Boshi Wang, Huan Sun, and Yu Su. Mind2web：迈向通用网页代理。Advances in Neural Information Processing Systems, 36:28091-28114, 2023.


Yue Fan, Handong Zhao, Ruiyi Zhang, Yu Shen, Xin Eric Wang, and Gang Wu. Gui-bee: Align gui action grounding to novel environments via autonomous exploration. arXiv preprint arXiv:2501.13896, 2025.
Yue Fan, Handong Zhao, Ruiyi Zhang, Yu Shen, Xin Eric Wang, and Gang Wu. Gui-bee：通过自主探索将 GUI 操作落地对齐到新环境。arXiv preprint arXiv:2501.13896, 2025.


Yuan Guo, Tingjia Miao, Zheng Wu, Pengzhou Cheng, Ming Zhou, and Zhuosheng Zhang. Atomic-to-compositional generalization for mobile agents with a new benchmark and scheduling system. arXiv preprint arXiv:2506.08972, 2025.
Yuan Guo, Tingjia Miao, Zheng Wu, Pengzhou Cheng, Ming Zhou, and Zhuosheng Zhang. 面向移动代理的从原子到组合泛化：新基准与调度系统。arXiv preprint arXiv:2506.08972, 2025.


Wenyi Hong, Weihan Wang, Qingsong Lv, Jiazheng Xu, Wenmeng Yu, Junhui Ji, Yan Wang, Zihan Wang, Yuxiao Dong, Ming Ding, et al. Cogagent: A visual language model for gui agents. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 14281-14290, 2024.
Wenyi Hong, Weihan Wang, Qingsong Lv, Jiazheng Xu, Wenmeng Yu, Junhui Ji, Yan Wang, Zihan Wang, Yuxiao Dong, Ming Ding, et al. Cogagent：面向 GUI 代理的视觉语言模型。收录于 IEEE/CVF 计算机视觉与模式识别会议论文集，页 14281-14290，2024.


Yongxiang Hu, Xuan Wang, Yingchuan Wang, Yu Zhang, Shiyu Guo, Chaoyi Chen, Xin Wang, and Yangfan Zhou. Auitestagent: Automatic requirements oriented gui function testing. arXiv preprint arXiv:2407.09018, 2024.
Yongxiang Hu, Xuan Wang, Yingchuan Wang, Yu Zhang, Shiyu Guo, Chaoyi Chen, Xin Wang, and Yangfan Zhou. Auitestagent：面向自动化需求的 GUI 功能测试。arXiv preprint arXiv:2407.09018, 2024.


Zeyu Huang, Juyuan Wang, Longfeng Chen, Boyi Xiao, Leng Cai, Yawen Zeng, and Jin Xu. Mvisu-bench: Benchmarking mobile agents for real-world tasks by multi-app, vague, interactive, single-app and unethical instructions. arXiv preprint arXiv:2508.09057, 2025.
Zeyu Huang, Juyuan Wang, Longfeng Chen, Boyi Xiao, Leng Cai, Yawen Zeng, and Jin Xu. Mvisu-bench：通过多应用、模糊、交互、单应用与不道德指令对移动代理进行真实任务基准测试。arXiv preprint arXiv:2508.09057, 2025.


Wenjia Jiang, Yangyang Zhuang, Chenxi Song, Xu Yang, Joey Tianyi Zhou, and Chi Zhang. Ap-pagentx: Evolving gui agents as proficient smartphone users. arXiv preprint arXiv:2503.02268, 2025.
Wenjia Jiang, Yangyang Zhuang, Chenxi Song, Xu Yang, Joey Tianyi Zhou, and Chi Zhang. Ap-pagentx：将 GUI 代理演化为熟练的智能手机用户。arXiv preprint arXiv:2503.02268, 2025.


Hongxin Li, Jingfan Chen, Jingran Su, Yuntao Chen, Qing Li, and Zhaoxiang Zhang. Auto-gui: Scaling gui grounding with automatic functionality annotations from llms. arXiv preprint arXiv:2502.01977, 2025.
Hongxin Li, Jingfan Chen, Jingran Su, Yuntao Chen, Qing Li, and Zhaoxiang Zhang. Auto-gui：通过来自 LLM 的自动功能注释扩展 GUI 落地。arXiv preprint arXiv:2502.01977, 2025.


Wei Li, William Bishop, Alice Li, Chris Rawles, Folawiyo Campbell-Ajala, Divya Tyamagundlu, and Oriana Riva. On the effects of data scale on computer control agents. arXiv e-prints, pp. arXiv-2406, 2024a.
Wei Li, William Bishop, Alice Li, Chris Rawles, Folawiyo Campbell-Ajala, Divya Tyamagundlu, and Oriana Riva. 关于数据规模对计算机控制代理影响的研究。arXiv e-prints, 页 arXiv-2406, 2024a.


Yang Li, Gang Li, Luheng He, Jingjie Zheng, Hong Li, and Zhiwei Guan. Widget captioning: Generating natural language description for mobile user interface elements. arXiv preprint arXiv:2010.04295, 2020.
Yang Li, Gang Li, Luheng He, Jingjie Zheng, Hong Li, and Zhiwei Guan. Widget captioning：为移动用户界面元素生成自然语言描述。arXiv preprint arXiv:2010.04295, 2020.


Yizhi Li, Ge Zhang, Yinghao Ma, Ruibin Yuan, Kang Zhu, Hangyu Guo, Yiming Liang, Jiaheng Liu, Zekun Wang, Jian Yang, et al. Omnibench: Towards the future of universal omni-language models. arXiv preprint arXiv:2409.15272, 2024b.
Yizhi Li, Ge Zhang, Yinghao Ma, Ruibin Yuan, Kang Zhu, Hangyu Guo, Yiming Liang, Jiaheng Liu, Zekun Wang, Jian Yang, et al. Omnibench：迈向通用全能语言模型的未来。arXiv preprint arXiv:2409.15272, 2024b.


Guangyi Liu, Pengxiang Zhao, Liang Liu, Zhiming Chen, Yuxiang Chai, Shuai Ren, Hao Wang, Shibo He, and Wenchao Meng. Learnact: Few-shot mobile gui agent with a unified demonstration benchmark. arXiv preprint arXiv:2504.13805, 2025a.
Guangyi Liu, Pengxiang Zhao, Liang Liu, Zhiming Chen, Yuxiang Chai, Shuai Ren, Hao Wang, Shibo He, and Wenchao Meng. Learnact: 少样本移动 GUI 代理与统一示范基准. arXiv preprint arXiv:2504.13805, 2025a.


Shunyu Liu, Minghao Liu, Huichi Zhou, Zhenyu Cui, Yang Zhou, Yuhao Zhou, Wendong Fan, Ge Zhang, Jiajun Shi, Weihao Xuan, et al. Verigui: Verifiable long-chain gui dataset. arXiv preprint arXiv:2508.04026, 2025b.
Shunyu Liu, Minghao Liu, Huichi Zhou, Zhenyu Cui, Yang Zhou, Yuhao Zhou, Wendong Fan, Ge Zhang, Jiajun Shi, Weihao Xuan, et al. Verigui: 可验证的长链 GUI 数据集. arXiv preprint arXiv:2508.04026, 2025b.


Quanfeng Lu, Wenqi Shao, Zitao Liu, Fanqing Meng, Boxuan Li, Botong Chen, Siyuan Huang, Kaipeng Zhang, Yu Qiao, and Ping Luo. Gui odyssey: A comprehensive dataset for cross-app gui navigation on mobile devices. arXiv preprint arXiv:2406.08451, 2024.
Quanfeng Lu, Wenqi Shao, Zitao Liu, Fanqing Meng, Boxuan Li, Botong Chen, Siyuan Huang, Kaipeng Zhang, Yu Qiao, and Ping Luo. Gui odyssey: 跨应用移动设备 GUI 导航的综合数据集. arXiv preprint arXiv:2406.08451, 2024.


Xinbei Ma, Yiting Wang, Yao Yao, Tongxin Yuan, Aston Zhang, Zhuosheng Zhang, and Hai Zhao. Caution for the environment: Multimodal agents are susceptible to environmental distractions. arXiv preprint arXiv:2408.02544, 2024.
Xinbei Ma, Yiting Wang, Yao Yao, Tongxin Yuan, Aston Zhang, Zhuosheng Zhang, and Hai Zhao. Caution for the environment: 多模态代理易受环境干扰. arXiv preprint arXiv:2408.02544, 2024.


OpenAI. Gpt-40 system card. https://cdn.openai.com/gpt-40-system-card.pdf, 2024.
OpenAI. Gpt-40 system card. https://cdn.openai.com/gpt-40-system-card.pdf, 2024.


Yujia Qin, Yining Ye, Junjie Fang, Haoming Wang, Shihao Liang, Shizuo Tian, Junda Zhang, Jiahao Li, Yunxin Li, Shijue Huang, et al. Ui-tars: Pioneering automated gui interaction with native agents. arXiv preprint arXiv:2501.12326, 2025.
Yujia Qin, Yining Ye, Junjie Fang, Haoming Wang, Shihao Liang, Shizuo Tian, Junda Zhang, Jiahao Li, Yunxin Li, Shijue Huang, et al. Ui-tars: 开创性的原生代理自动化 GUI 交互. arXiv preprint arXiv:2501.12326, 2025.


Christopher Rawles, Alice Li, Daniel Rodriguez, Oriana Riva, and Timothy Lillicrap. An-droidinthewild: A large-scale dataset for android device control. Advances in Neural Information Processing Systems, 36:59708-59728, 2023.
Christopher Rawles, Alice Li, Daniel Rodriguez, Oriana Riva, and Timothy Lillicrap. An-droidinthewild: 大规模安卓设备控制数据集. Advances in Neural Information Processing Systems, 36:59708-59728, 2023.


Christopher Rawles, Sarah Clinckemaillie, Yifan Chang, Jonathan Waltz, Gabrielle Lau, Marybeth Fair, Alice Li, William Bishop, Wei Li, Folawiyo Campbell-Ajala, et al. Androidworld: A dynamic benchmarking environment for autonomous agents. arXiv preprint arXiv:2405.14573, 2024.
Christopher Rawles, Sarah Clinckemaillie, Yifan Chang, Jonathan Waltz, Gabrielle Lau, Marybeth Fair, Alice Li, William Bishop, Wei Li, Folawiyo Campbell-Ajala, et al. Androidworld: 面向自主代理的动态基准环境. arXiv preprint arXiv:2405.14573, 2024.


Liangtai Sun, Xingyu Chen, Lu Chen, Tianle Dai, Zichen Zhu, and Kai Yu. Meta-gui: Towards multi-modal conversational agents on mobile gui. arXiv preprint arXiv:2205.11029, 2022.
Liangtai Sun, Xingyu Chen, Lu Chen, Tianle Dai, Zichen Zhu, and Kai Yu. Meta-gui: 面向移动 GUI 的多模态对话代理. arXiv preprint arXiv:2205.11029, 2022.


Yuchen Sun, Shanhui Zhao, Tao Yu, Hao Wen, Samith Va, Mengwei Xu, Yuanchun Li, and Chongyang Zhang. Gui-xplore: Empowering generalizable gui agents with one exploration. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 19477-19486, 2025.
Yuchen Sun, Shanhui Zhao, Tao Yu, Hao Wen, Samith Va, Mengwei Xu, Yuanchun Li, and Chongyang Zhang. Gui-xplore: 以一次探索赋能可泛化的 GUI 代理. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 19477-19486, 2025.


Milad Taleby Ahvanooey, Hassan Dana Mazraeh, and Seyed Hashem Tabasi. An innovative technique for web text watermarking (aitw). Information Security Journal: A Global Perspective, 25(4-6): 191-196, 2016.
Milad Taleby Ahvanooey, Hassan Dana Mazraeh, and Seyed Hashem Tabasi. 用于网页文本水印的创新技术 (aitw). Information Security Journal: A Global Perspective, 25(4-6): 191-196, 2016.


Qwen Team. Qwen-vl max, 2023. URL https://github.com/QwenLM/Qwen-VL.
Qwen Team. Qwen-vl max, 2023. URL https://github.com/QwenLM/Qwen-VL.


Qwen Team. Qwen3-vl, 2025. URL https://github.com/QwenLM/Qwen3-VL.
Qwen Team. Qwen3-vl, 2025. URL https://github.com/QwenLM/Qwen3-VL.


Bryan Wang, Gang Li, Xin Zhou, Zhourong Chen, Tovi Grossman, and Yang Li. Screen2words: Automatic mobile ui summarization with multimodal learning. In The 34th Annual ACM Symposium on User Interface Software and Technology, pp. 498-510, 2021.
Bryan Wang, Gang Li, Xin Zhou, Zhourong Chen, Tovi Grossman, and Yang Li. Screen2words: 基于多模态学习的自动移动界面摘要. In The 34th Annual ACM Symposium on User Interface Software and Technology, pp. 498-510, 2021.


Junyang Wang, Haiyang Xu, Haitao Jia, Xi Zhang, Ming Yan, Weizhou Shen, Ji Zhang, Fei Huang, and Jitao Sang. Mobile-agent-v2: Mobile device operation assistant with effective navigation via multi-agent collaboration. Advances in Neural Information Processing Systems, 37:2686-2710, 2024a.
Junyang Wang, Haiyang Xu, Haitao Jia, Xi Zhang, Ming Yan, Weizhou Shen, Ji Zhang, Fei Huang, and Jitao Sang. Mobile-agent-v2: 通过多代理协作实现有效导航的移动设备操作助手. Advances in Neural Information Processing Systems, 37:2686-2710, 2024a.


Junyang Wang, Haiyang Xu, Jiabo Ye, Ming Yan, Weizhou Shen, Ji Zhang, Fei Huang, and Jitao Sang. Mobile-agent: Autonomous multi-modal mobile device agent with visual perception. arXiv preprint arXiv:2401.16158, 2024b.
Junyang Wang, Haiyang Xu, Jiabo Ye, Ming Yan, Weizhou Shen, Ji Zhang, Fei Huang, and Jitao Sang. Mobile-agent: 具视觉感知的自主多模态移动设备代理. arXiv preprint arXiv:2401.16158, 2024b.


Luyuan Wang, Yongyu Deng, Yiwei Zha, Guodong Mao, Qinmin Wang, Tianchen Min, Wei Chen, and Shoufa Chen. Mobileagentbench: An efficient and user-friendly benchmark for mobile llm agents. arXiv preprint arXiv:2406.08184, 2024c.
王禄元, 邓永宇, 查怡玮, 茅国东, 王钦旻, 闵天辰, 陈威, 和 陈守发. Mobileagentbench：一个高效且用户友好的移动端 LLM 代理基准. arXiv preprint arXiv:2406.08184, 2024c.


Weixuan Wang, Dongge Han, Daniel Madrigal Diaz, Jin Xu, Victor Rühle, and Saravan Rajmohan. Odysseybench: Evaluating llm agents on long-horizon complex office application workflows. arXiv preprint arXiv:2508.09124, 2025a.
王玮轩, 韩东格, Daniel Madrigal Diaz, 许晋, Victor Rühle, 和 Saravan Rajmohan. Odysseybench：在长期复杂办公应用工作流上评估 LLM 代理. arXiv preprint arXiv:2508.09124, 2025a.


Zhenhailong Wang, Haiyang Xu, Junyang Wang, Xi Zhang, Ming Yan, Ji Zhang, Fei Huang, and Heng Ji. Mobile-agent-e: Self-evolving mobile assistant for complex tasks. arXiv preprint arXiv:2501.11733, 2025b.
王振海龙, 许海洋, 王俊阳, 张熙, 晏明, 张骥, 黄飞, 和 姜恒. Mobile-agent-e：面向复杂任务的自我进化移动助手. arXiv preprint arXiv:2501.11733, 2025b.


Hao Wen, Yuanchun Li, Guohong Liu, Shanhui Zhao, Tao Yu, Toby Jia-Jun Li, Shiqi Jiang, Yunhao Liu, Yaqin Zhang, and Yunxin Liu. Autodroid: Llm-powered task automation in android. In Proceedings of the 30th Annual International Conference on Mobile Computing and Networking, pp. 543-557, 2024.
闻浩, 李远春, 刘国宏, 赵山辉, 余涛, 李家俊 (Toby Jia-Jun Li), 蒋世琦, 刘云浩, 张雅琴, 和 刘运鑫. Autodroid：在 Android 上由 LLM 驱动的任务自动化. 收录于第30届移动计算与网络国际年会论文集, 页543-557, 2024.


Qinzhuo Wu, Weikai Xu, Wei Liu, Tao Tan, Jianfeng Liu, Ang Li, Jian Luan, Bin Wang, and Shuo Shang. Mobilevlm: A vision-language model for better intra-and inter-ui understanding. arXiv preprint arXiv:2409.14818, 2024a.
吴勤卓, 许伟楷, 刘伟, 谭涛, 刘建峰, 李昂, 栾健, 王斌, 和 商硕. MobileVLM：用于更好理解界面内与界面间的视觉-语言模型. arXiv preprint arXiv:2409.14818, 2024a.


Zheng Wu, Heyuan Huang, Yanjia Yang, Yuanyi Song, Xingyu Lou, Wei̇wen Liu, Weinan Zhang, Jun Wang, and Zhuosheng Zhang. Quick on the uptake: Eliciting implicit intents from human demonstrations for personalized mobile-use agents. arXiv preprint arXiv:2508.08645, 2025.
吴正, 黄和元, 杨晏佳, 宋远毅, 娄星煜, 刘维文, 张伟楠, 王俊, 和 张卓生. 快速领会：从人类示范中引导隐含意图以实现个性化移动使用代理. arXiv preprint arXiv:2508.08645, 2025.


Zhiyong Wu, Zhenyu Wu, Fangzhi Xu, Yian Wang, Qiushi Sun, Chengyou Jia, Kanzhi Cheng, Zichen Ding, Liheng Chen, Paul Pu Liang, et al. Os-atlas: A foundation action model for generalist gui agents. arXiv preprint arXiv:2410.23218, 2024b.
吴志勇, 吴振宇, 徐方志, 王奕安, 孙秋实, 贾成友, 程坤志, 丁子辰, 陈立衡, Paul Pu Liang, 等. OS-Atlas：面向通用 GUI 代理的基础动作模型. arXiv preprint arXiv:2410.23218, 2024b.


Weikai Xu, Zhizheng Jiang, Yuxuan Liu, Pengzhi Gao, Wei Liu, Jian Luan, Yuanchun Li, Yunxin Liu, Bin Wang, and Bo An. Mobile-bench-v2: A more realistic and comprehensive benchmark for vlm-based mobile agents. arXiv preprint arXiv:2505.11891, 2025.
许伟楷, 姜志正, 刘宇轩, 高鹏致, 刘伟, 栾健, 李远春, 刘运鑫, 王斌, 和 安博. Mobile-bench-v2：更真实更全面的基于 VLM 的移动代理基准. arXiv preprint arXiv:2505.11891, 2025.


Yifan Xu, Xiao Liu, Xueqiao Sun, Siyi Cheng, Hao Yu, Hanyu Lai, Shudan Zhang, Dan Zhang, Jie Tang, and Yuxiao Dong. Androidlab: Training and systematic benchmarking of android autonomous agents. arXiv preprint arXiv:2410.24024, 2024.
许一凡, 刘晓, 孙学乔, 程思怡, 余浩, 赖翰宇, 张淑丹, 张丹, 唐杰, 和 董宇霄. AndroidLab：Android 自主代理的训练与系统基准测试. arXiv preprint arXiv:2410.24024, 2024.


Jiabo Ye, Xi Zhang, Haiyang Xu, Haowei Liu, Junyang Wang, Zhaoqing Zhu, Ziwei Zheng, Feiyu Gao, Junjie Cao, Zhengxi Lu, et al. Mobile-agent-v3: Foundamental agents for gui automation. arXiv preprint arXiv:2508.15144, 2025a.
叶佳博, 张熙, 许海洋, 刘浩威, 王俊阳, 朱昭清, 郑子为, 高飞宇, 曹俊杰, 陆正熙, 等. Mobile-agent-v3：面向 GUI 自动化的基础代理. arXiv preprint arXiv:2508.15144, 2025a.


Suyu Ye, Haojun Shi, Darren Shih, Hyokun Yun, Tanya Roosta, and Tianmin Shu. Realwebassist: A benchmark for long-horizon web assistance with real-world users. arXiv preprint arXiv:2504.10445, 2025b.
叶素雨, 石浩君, Darren Shih, 尹赫坤, Tanya Roosta, 和 舒天民. RealWebAssist：面向真实用户的长期网页辅助基准. arXiv preprint arXiv:2504.10445, 2025b.


Chaoyun Zhang, Shilin He, Jiaxu Qian, Bowen Li, Liqun Li, Si Qin, Yu Kang, Minghua Ma, Guyue Liu, Qingwei Lin, et al. Large language model-brained gui agents: A survey. arXiv preprint arXiv:2411.18279, 2024a.
张朝云, 何世霖, 钱佳旭, 李博文, 李立群, 秦思, 康宇, 马明华, 刘谷越, 林庆威, 等. 大型语言模型驱动的 GUI 代理：一项综述. arXiv preprint arXiv:2411.18279, 2024a.


Chi Zhang, Zhao Yang, Jiaxuan Liu, Yanda Li, Yucheng Han, Xin Chen, Zebiao Huang, Bin Fu, and Gang Yu. Appagent: Multimodal agents as smartphone users. In Proceedings of the 2025 CHI Conference on Human Factors in Computing Systems, pp. 1-20, 2025.
张驰, 杨钊, 刘佳轩, 李言达, 韩宇程, 陈昕, 黄泽彪, 傅斌, 和 余刚. AppAgent：作为智能手机用户的多模态代理. 收录于2025年 CHI 人因与计算系统会议论文集, 页1-20, 2025.


Danyang Zhang, Zhennan Shen, Rui Xie, Situo Zhang, Tianbao Xie, Zihan Zhao, Siyuan Chen, Lu Chen, Hongshen Xu, Ruisheng Cao, et al. Mobile-env: Building qualified evaluation benchmarks for llm-gui interaction. arXiv preprint arXiv:2305.08144, 2023.
Danyang Zhang, Zhennan Shen, Rui Xie, Situo Zhang, Tianbao Xie, Zihan Zhao, Siyuan Chen, Lu Chen, Hongshen Xu, Ruisheng Cao, et al. Mobile-env: 为 llm-gui 交互构建合格的评估基准。arXiv preprint arXiv:2305.08144, 2023.


Jiwen Zhang, Jihao Wu, Yihua Teng, Minghui Liao, Nuo Xu, Xiao Xiao, Zhongyu Wei, and Duyu Tang. Android in the zoo: Chain-of-action-thought for gui agents. arXiv preprint arXiv:2403.02713, 2024b.
Jiwen Zhang, Jihao Wu, Yihua Teng, Minghui Liao, Nuo Xu, Xiao Xiao, Zhongyu Wei, and Duyu Tang. Android in the zoo: 针对 GUI 代理的链式行动-思维。arXiv preprint arXiv:2403.02713, 2024b.


Zaibin Zhang, Yongting Zhang, Lijun Li, Hongzhi Gao, Lijun Wang, Huchuan Lu, Feng Zhao, Yu Qiao, and Jing Shao. Psysafe: A comprehensive framework for psychological-based attack, defense, and evaluation of multi-agent system safety. arXiv preprint arXiv:2401.11880, 2024c.
Zaibin Zhang, Yongting Zhang, Lijun Li, Hongzhi Gao, Lijun Wang, Huchuan Lu, Feng Zhao, Yu Qiao, and Jing Shao. Psysafe: 一个用于基于心理的攻击、防御与多智能体系统安全评估的综合框架。arXiv preprint arXiv:2401.11880, 2024c.


## A Additional Information of ColorBench
## A ColorBench 的补充信息


### A.1 Action Space
### A.1 行动空间


Our ColorBench includes nine actions, and during evaluation, the outputs of different models need to be aligned to the action space.
我们的 ColorBench 包含九种动作，评估时需将不同模型的输出对齐到该行动空间。


Table 4: Action space of ColorBench.
表 4：ColorBench 的行动空间。


<table><tr><td>Action Type</td><td>Parameter</td></tr><tr><td>click</td><td>coordinate $= \left( {x,y}\right)$</td></tr><tr><td>long press</td><td>coordinate $= \left( {x,y}\right)$</td></tr><tr><td>swipe</td><td>direction $\in  \{$ up,down,left,right $\}$</td></tr><tr><td>type</td><td>content=[TEXT]</td></tr><tr><td>wait</td><td>coordinate= $\left( {x,y}\right)$</td></tr><tr><td>open</td><td>app= app name</td></tr><tr><td>navigate back</td><td>none</td></tr><tr><td>navigate home</td><td>none</td></tr><tr><td>complete</td><td>answer=[TEXT]</td></tr></table>
<table><tbody><tr><td>操作类型</td><td>参数</td></tr><tr><td>点击</td><td>坐标 $= \left( {x,y}\right)$</td></tr><tr><td>长按</td><td>坐标 $= \left( {x,y}\right)$</td></tr><tr><td>滑动</td><td>方向 $\in  \{$ 上,下,左,右 $\}$</td></tr><tr><td>输入</td><td>内容=[TEXT]</td></tr><tr><td>等待</td><td>坐标= $\left( {x,y}\right)$</td></tr><tr><td>打开</td><td>应用= 应用名</td></tr><tr><td>返回</td><td>无</td></tr><tr><td>回到主屏</td><td>无</td></tr><tr><td>完成</td><td>回答=[TEXT]</td></tr></tbody></table>


### A.2 Annotation of Bounding Box
### A.2 边界框标注


Accurate bounding boxes are crucial for GUIs, as they define the precise spatial scope of interactive UI elements, preventing mapping errors between user actions and irrelevant areas. Therefore, this paper proposes an annotation method combining multiple VLM integration with human verification, divided into three concise steps:
准确的边界框对 GUI 至关重要，因为它们界定了交互式 UI 元素的精确空间范围，避免用户操作与无关区域的映射错误。因此，本文提出了一种结合多 VLM 集成与人工核验的标注方法，分为三步简明流程：


- Step 1: Input actual interaction point coordinates and corresponding images into two VLMs, each generating an interaction region. One VLM outputs a larger bounding box (ensuring complete target coverage), while the other outputs a smaller bounding box (minimizing background inclusion).
- Step 1: 将实际交互点坐标和对应图像输入两个 VLM，各自生成一个交互区域。一个 VLM 输出较大的边界框（确保目标完整覆盖），另一个输出较小的边界框（尽量减少背景包含）。


- Step 2: A third VLM evaluates the two candidate boxes and selects the optimal one. If neither candidate is satisfactory, a new bounding box is generated.
- Step 2: 第三个 VLM 对两个候选框进行评估并选择最优者。若均不满意，则生成新的边界框。


- Step 3: Domain experts verify the preliminary results from Step 2, correcting errors (e.g., missing target parts, redundant background) and confirming the final annotation results.
- Step 3: 领域专家对步骤 2 的初步结果进行核验，修正错误（如目标缺失、冗余背景）并确认最终标注结果。


### A.3 Supplement on ColorBench Construction
### A.3 关于 ColorBench 构建的补充


#### A.3.1 Automated Strategy Model Selection
#### A.3.1 自动化策略模型选择


Our automated pipeline leverages specialized models according to their core capabilities. Qwen2.5- VL-72B, chosen for its strong general reasoning, handles UI element identification and interaction during broad-coverage trajectory collection, predicts the correct action between model outputs for deep trajectory action annotation, and generates page state descriptions with visual similarity judgment during graph merging. Page descriptions are encoded by the "models-BAAI-bge-large-zh-v1.5" model for semantic similarity calculation. GUI-OWL-32B, recognized for its exceptional visual grounding capability, is employed exclusively for bounding box annotation.
我们的自动化流水线根据模型的核心能力调度专用模型。Qwen2.5- VL-72B 因其强大的通用推理能力，被用于广覆盖轨迹采集中的 UI 元素识别与交互、对模型输出间正确动作的预测以进行深度轨迹动作标注，以及在图合并过程中进行页面状态描述与视觉相似度判断。页面描述由 "models-BAAI-bge-large-zh-v1.5" 模型进行编码以计算语义相似度。GUI-OWL-32B 以其出色的视觉定位能力专用于边界框标注。


#### A.3.2 Data Collection and Quality Assurance Results
#### A.3.2 数据采集与质量保证结果


During the BFS phase, trajectory collection was performed for four individual applications: Weixin, Meituan, Jingdong, Xiaohongshu, yielding 6300 trajectory screenshots. During the DFS phase, an additional 1343 trajectory screenshots for complex long-horizon tasks (including single-app and cross-app ) were collected. Using a graph merging strategy to integrate these trajectories into a unified graph structure, we conducted manual quality control to remove near-duplicate images from identical nodes, retaining only a limited set of representative screenshots. To address coverage gaps in the graph, we manually supplemented 50 screenshots. The final constructed graph contains 1989 carefully curated screenshots.
在 BFS 阶段，对四个单体应用（微信、美团、京东、小红书）进行了轨迹采集，产出 6300 张轨迹截图。DFS 阶段又采集了 1343 张用于复杂长时序任务（包含单应用与跨应用）的轨迹截图。使用图合并策略将这些轨迹整合为统一图结构后，我们进行了人工质检以去除相同节点的近重复图片，仅保留有限的代表性截图。为弥补图中的覆盖空白，我们人工补充了 50 张截图。最终构建的图包含 1989 张经慎重挑选的截图。


#### A.3.3 System Prompt in Graph Merging
#### A.3.3 图合并中的系统提示


Figure 5 presents the system prompt template used in our automated graph merging process. Due to page limitations, additional implementation details will be made available in our future open-source release.
图 5 展示了我们自动化图合并过程中使用的系统提示模板。由于篇幅限制，更多实现细节将在我们未来的开源发布中提供。


## B Experimental Supplement
## B 实验补充


### B.1 Implementation Details
### B.1 实现细节


All open-source models tested in our experiments were deployed using VLLM(0.10.1) on A100 80G hardware. Since each node in the image dataset may contain multiple images, we implemented random returns and set the random seed to 2025. We configured model inference parameters as follows: temperature $= {0.1}$ ,top_k=5,top_p=0.9,max_tokens $= {1024}$ . To ensure experimental fairness,we uniformly employed the open-source prompts provided by each model while preserving their original action spaces and performing additional alignment on outputs. We also designed identical prompts for closed-source models. For these, we supplemented each input with the model's historical output records and captured its current reasoning and actions. We will open-source the project code in the future.
我们实验中测试的所有开源模型均在 A100 80G 硬件上使用 VLLM(0.10.1) 部署。因图像数据集中每个节点可能包含多张图像，我们实现了随机返回并将随机种子设为 2025。模型推理参数配置如下：temperature $= {0.1}$ ,top_k=5,top_p=0.9,max_tokens $= {1024}$ 。为确保实验公平性，我们统一采用各模型提供的开源提示，同时保留其原始动作空间并对输出进行额外对齐。我们也为闭源模型设计了相同的提示，并为其补充每次输入的历史输出记录以捕获其当前推理与动作。项目代码将在未来开源。


## Prompt 1: Generating Page Description fonttitle
## Prompt 1: 生成页面描述 fonttitle


You are a GUI AGENT. Please define in one sentence what page the given mobile screenshot represents, and provide a general description of the page layout, its purpose, and other key information. You should ignore changes in the screenshot caused by time or updates, and ultimately form a brief textual description.
你是一个 GUI 代理。请用一句话定义给定移动截图代表的页面，并简要描述页面布局、用途及其他关键信息。应忽略因时间或更新引起的截图差异，最终形成一段简洁的文本描述。


## Prompt 2: Judgement of Same Screen State fonttitle
## 提示 2：判定是否为相同页面状态 fonttitle


You are a GUI AGENT. Please ignore changes in the screenshots caused by time, and judge whether the two given images belong to the same page state based on aspects such as page layout, page function, and action relationships.
你是一个 GUI 代理。请忽略因时间导致的截图差异，并根据页面布局、页面功能和操作关系等方面判断两张给定图片是否属于相同的页面状态。


Notes: 1. Treat all mobile home screen pages as the same page;
注意：1. 将所有手机主屏页视为相同页面；


2. Ignore changes in the page caused by time and updates to recommended content, and focus only on changes caused by actions;
2. 忽略因时间及推荐内容更新引起的页面变化，仅关注由操作引起的变化；


3. Pay special attention to navigation bar tabs. In the application, different navigation bar tabs indicate different pages;
3. 特别注意导航栏选项卡；在应用中，不同的导航栏选项卡代表不同页面；


4. Page state changes caused by actions should be considered different pages. If performing an action on one image is required to turn it into the other image, then these two images represent different page states.
4. 由操作引起的页面状态变化应视为不同页面。如果需要对一张图执行某个操作才能变成另一张图，则这两张图代表不同的页面状态。


Figure 5: System prompt templates for merging graph. In practical use, given the language of the application, we employed Chinese-language prompts.
图 5：用于合并图的系统提示模板。实际使用中，根据应用语言，我们使用了中文提示。


### B.2 Result of Atomic Capabilities
### B.2 原子能力结果


Table 5 presents evaluation results for atomic task capabilities across open-source and closed-source models. A "-" in the table indicates the model never executed that subtask, having failed at a preceding subtask, thus rendering the subtask capability unverifiable. Higher scores in the table do not directly reflect the model's capability; rather, it is the poorer results that effectively reveal the causes of task failure. For example, GLM-4.5V exhibits high SR and CR but a "save" score of 0, indicating it cannot perform the save action. UI-TARS-72B-DPO also exhibits high SR and CR, yet its "filter" score is as low as 22.22, indicating this is the critical factor constraining the model's success.
表 5 展示了开源与闭源模型在原子任务能力上的评估结果。表中“ - ”表示模型从未执行该子任务，因其在前一子任务失败，导致该子任务能力无法验证。表中较高分数并不直接反映模型能力；反而是较差的结果能有效揭示任务失败原因。例如，GLM-4.5V 显示出较高的 SR 与 CR，但“保存”得分为 0，表明其无法执行保存操作。UI-TARS-72B-DPO 同样显示出较高的 SR 与 CR，然而其“筛选”得分仅为 22.22，表明这是限制模型成功的关键因素。


Table 5: Success rate of atomic capabilities of closed-source and open-source models on our ColorBench. All values are in percentage (%). The symbol '-' indicates that the model did not encounter this type of subtask during the evaluation process, usually due to the failure of a preceding task.
表 5：在我们的 ColorBench 上，闭源与开源模型的原子能力成功率。所有数值以百分比（%）表示。符号“ - ”表示该模型在评估过程中未遇到该类型的子任务，通常是由于前置任务失败。


<table><tr><td>Model</td><td>follow</td><td>pay</td><td>save</td><td>search</td><td>share</td><td>set</td><td>find</td><td>copy</td><td>filter</td><td>like</td><td>send</td><td>location</td><td>navigation</td><td>others</td><td>memory</td></tr><tr><td>GPT-40</td><td>100.00</td><td>50.00</td><td>50.00</td><td>33.06</td><td>76.47</td><td>84.62</td><td>56.67</td><td>0.00</td><td>18.75</td><td>80.00</td><td>25.00</td><td>25.00</td><td>0.00</td><td>62.50</td><td>0.00</td></tr><tr><td>Qwen-VL Max</td><td>40</td><td>33.33</td><td>16.67</td><td>43.8</td><td>70.37</td><td>83.33</td><td>60.56</td><td>33.33</td><td>36.36</td><td>80</td><td>50.00</td><td>25.00</td><td>-</td><td>68.57</td><td>0.00</td></tr><tr><td>GLM-4.5V</td><td>90.91</td><td>55.56</td><td>0.00</td><td>72.95</td><td>90.62</td><td>75.00</td><td>76.34</td><td>50.00</td><td>26.32</td><td>81.82</td><td>77.78</td><td>66.67</td><td>75.00</td><td>60.00</td><td>45.45</td></tr><tr><td>OS-Atlas-Pro-7B</td><td>100.00</td><td>0.00</td><td>75.00</td><td>43.22</td><td>35.71</td><td>66.67</td><td>35.94</td><td>0.00</td><td>0.00</td><td>71.43</td><td>25.00</td><td>40.00</td><td>100.00</td><td>26.09</td><td>0.00</td></tr><tr><td>UI-TARS-1.5-7B</td><td>50.00</td><td>0.00</td><td>33.33</td><td>30.25</td><td>7.69</td><td>40.00</td><td>37.93</td><td>100.00</td><td>20.00</td><td>100.00</td><td>0.00</td><td>20.00</td><td>0.00</td><td>40.00</td><td>0.00</td></tr><tr><td>UI-TARS-7B-DPO</td><td>50.00</td><td>0.00</td><td>50.00</td><td>24.58</td><td>37.50</td><td>66.67</td><td>25.49</td><td>0.00</td><td>22.22</td><td>71.43</td><td>0.00</td><td>0.00</td><td>-</td><td>34.78</td><td>0.00</td></tr><tr><td>GUI-OWL-7B</td><td>85.71</td><td>60.00</td><td>25.00</td><td>57.85</td><td>88.46</td><td>69.23</td><td>62.67</td><td>50.00</td><td>11.11</td><td>75.00</td><td>50.00</td><td>33.33</td><td>0.00</td><td>58.97</td><td>0.00</td></tr><tr><td>Qwen2.5-VL-7B</td><td>83.33</td><td>75</td><td>16.67</td><td>61.48</td><td>50.00</td><td>75.00</td><td>52.00</td><td>100.00</td><td>25</td><td>77.78</td><td>83.33</td><td>16.67</td><td>0</td><td>51.52</td><td>0.00</td></tr><tr><td>GUI-OWL-32B</td><td>87.50</td><td>62.50</td><td>0.00</td><td>64.23</td><td>81.82</td><td>84.62</td><td>64.10</td><td>50.00</td><td>31.25</td><td>100.00</td><td>57.14</td><td>40.00</td><td>66.67</td><td>69.23</td><td>0.00</td></tr><tr><td>Qwen2.5-VL-32B</td><td>33.33</td><td>83.33</td><td>0.00</td><td>54.55</td><td>71.43</td><td>84.62</td><td>52.05</td><td>50.00</td><td>16.67</td><td>88.89</td><td>75.00</td><td>50.00</td><td>50.00</td><td>59.38</td><td>0.00</td></tr><tr><td>UI-TARS-72B-DPO</td><td>77.78</td><td>55.56</td><td>28.57</td><td>68.50</td><td>85.00</td><td>85.71</td><td>63.29</td><td>100.00</td><td>22.22</td><td>88.89</td><td>66.67</td><td>50.00</td><td>66.67</td><td>73.17</td><td>28.57</td></tr><tr><td>Qwen2.5-VL-72B</td><td>42.86</td><td>75.00</td><td>0.00</td><td>61.48</td><td>66.67</td><td>84.62</td><td>52.00</td><td>50.00</td><td>6.67</td><td>90.00</td><td>60.00</td><td>20.00</td><td>50.00</td><td>54.84</td><td>0.00</td></tr><tr><td>Qwen3-VL-235B-A22B-Instruct</td><td>90.00</td><td>71.43</td><td>28.57</td><td>62.90</td><td>86.49</td><td>92.31</td><td>68.75</td><td>100.00</td><td>35.00</td><td>90.91</td><td>83.33</td><td>16.67</td><td>0.00</td><td>71.43</td><td>50.00</td></tr></table>
<table><tbody><tr><td>模型</td><td>关注</td><td>支付</td><td>保存</td><td>搜索</td><td>分享</td><td>设置</td><td>查找</td><td>复制</td><td>筛选</td><td>赞</td><td>发送</td><td>位置</td><td>导航</td><td>其它</td><td>记忆</td></tr><tr><td>GPT-40</td><td>100.00</td><td>50.00</td><td>50.00</td><td>33.06</td><td>76.47</td><td>84.62</td><td>56.67</td><td>0.00</td><td>18.75</td><td>80.00</td><td>25.00</td><td>25.00</td><td>0.00</td><td>62.50</td><td>0.00</td></tr><tr><td>Qwen-VL Max</td><td>40</td><td>33.33</td><td>16.67</td><td>43.8</td><td>70.37</td><td>83.33</td><td>60.56</td><td>33.33</td><td>36.36</td><td>80</td><td>50.00</td><td>25.00</td><td>-</td><td>68.57</td><td>0.00</td></tr><tr><td>GLM-4.5V</td><td>90.91</td><td>55.56</td><td>0.00</td><td>72.95</td><td>90.62</td><td>75.00</td><td>76.34</td><td>50.00</td><td>26.32</td><td>81.82</td><td>77.78</td><td>66.67</td><td>75.00</td><td>60.00</td><td>45.45</td></tr><tr><td>OS-Atlas-Pro-7B</td><td>100.00</td><td>0.00</td><td>75.00</td><td>43.22</td><td>35.71</td><td>66.67</td><td>35.94</td><td>0.00</td><td>0.00</td><td>71.43</td><td>25.00</td><td>40.00</td><td>100.00</td><td>26.09</td><td>0.00</td></tr><tr><td>UI-TARS-1.5-7B</td><td>50.00</td><td>0.00</td><td>33.33</td><td>30.25</td><td>7.69</td><td>40.00</td><td>37.93</td><td>100.00</td><td>20.00</td><td>100.00</td><td>0.00</td><td>20.00</td><td>0.00</td><td>40.00</td><td>0.00</td></tr><tr><td>UI-TARS-7B-DPO</td><td>50.00</td><td>0.00</td><td>50.00</td><td>24.58</td><td>37.50</td><td>66.67</td><td>25.49</td><td>0.00</td><td>22.22</td><td>71.43</td><td>0.00</td><td>0.00</td><td>-</td><td>34.78</td><td>0.00</td></tr><tr><td>GUI-OWL-7B</td><td>85.71</td><td>60.00</td><td>25.00</td><td>57.85</td><td>88.46</td><td>69.23</td><td>62.67</td><td>50.00</td><td>11.11</td><td>75.00</td><td>50.00</td><td>33.33</td><td>0.00</td><td>58.97</td><td>0.00</td></tr><tr><td>Qwen2.5-VL-7B</td><td>83.33</td><td>75</td><td>16.67</td><td>61.48</td><td>50.00</td><td>75.00</td><td>52.00</td><td>100.00</td><td>25</td><td>77.78</td><td>83.33</td><td>16.67</td><td>0</td><td>51.52</td><td>0.00</td></tr><tr><td>GUI-OWL-32B</td><td>87.50</td><td>62.50</td><td>0.00</td><td>64.23</td><td>81.82</td><td>84.62</td><td>64.10</td><td>50.00</td><td>31.25</td><td>100.00</td><td>57.14</td><td>40.00</td><td>66.67</td><td>69.23</td><td>0.00</td></tr><tr><td>Qwen2.5-VL-32B</td><td>33.33</td><td>83.33</td><td>0.00</td><td>54.55</td><td>71.43</td><td>84.62</td><td>52.05</td><td>50.00</td><td>16.67</td><td>88.89</td><td>75.00</td><td>50.00</td><td>50.00</td><td>59.38</td><td>0.00</td></tr><tr><td>UI-TARS-72B-DPO</td><td>77.78</td><td>55.56</td><td>28.57</td><td>68.50</td><td>85.00</td><td>85.71</td><td>63.29</td><td>100.00</td><td>22.22</td><td>88.89</td><td>66.67</td><td>50.00</td><td>66.67</td><td>73.17</td><td>28.57</td></tr><tr><td>Qwen2.5-VL-72B</td><td>42.86</td><td>75.00</td><td>0.00</td><td>61.48</td><td>66.67</td><td>84.62</td><td>52.00</td><td>50.00</td><td>6.67</td><td>90.00</td><td>60.00</td><td>20.00</td><td>50.00</td><td>54.84</td><td>0.00</td></tr><tr><td>Qwen3-VL-235B-A22B-Instruct</td><td>90.00</td><td>71.43</td><td>28.57</td><td>62.90</td><td>86.49</td><td>92.31</td><td>68.75</td><td>100.00</td><td>35.00</td><td>90.91</td><td>83.33</td><td>16.67</td><td>0.00</td><td>71.43</td><td>50.00</td></tr></tbody></table>


Table 6: Comparison of some failure results evaluated using the ColorBench task on real devices and on the graph dataset. Failure reasons are recorded separately for the specific reasons under the two evaluation environments. The tasks in the table are actually evaluated in Chinese.
表6：在真实设备与图数据集上使用 ColorBench 任务评估的一些失败结果比较。两种评估环境下的具体失败原因分别记录。表中任务实际以中文评估。


<table><tr><td>Task</td><td>Failure Reason of Real-Device</td><td>Failure Reason of Graph</td></tr><tr><td>Search for "Guangzhou travel guides" on Xiao-hongshu, share the first one with Xiaohongshu friend Hh Yuan, and ask "Which place do you want to visit the most?" Then use Xiaohongshu to search for food near Canton Tower, remember the name of the first restaurant, search for that restaurant on Meituan and order a set meal for two.</td><td>Accidentally clicked on the advertisement when searching for the store on Meituan.</td><td>Failed to search for food on Meituan.</td></tr><tr><td>Search for "Guangzhou travel guides" on Xi-aohongshu, find the first one, check the blog-ger's followers, likes, and favorites, and then tell WeChat friend 1.</td><td>There is an error in the data sent to WeChat friends.</td><td>Did not enter the blogger's homepage.</td></tr><tr><td>Search for "Guangzhou travel guides" on Xiao-hongshu, use the "Q&A" to let AI reply, and save the generated content as an image to the album.</td><td>Fail to find "Q&A".</td><td>Failed to save as an image.</td></tr><tr><td>Compare the price of the "Xin Dou Ji Guangzhou Tower" double set meal on Meituan, Dazhong Dianping, and TikTok, then forward the cheapest one to friend 1.</td><td>It ends after executing on Meituan.</td><td>Failed to search after Meituan.</td></tr><tr><td>Search for "Baheli Beef Hotpot" on Meituan, forward all double set meals to WeChat friend 1, and ask him to "choose one" after sharing.</td><td>0 No sharing, just repeatedly clicking to grab the deal.</td><td>Only shared one set meal.</td></tr><tr><td>Order takeaway from "Baheli Beef Hotpot" on Meituan: one dry-fried beef, without dinnerware.</td><td>Mistakenly viewed the in-store set meal instead of selecting takeout.</td><td>Did not accurately find takeout.</td></tr><tr><td>Search for "Baheli Beef Hotpot" on Meituan, choose the first restaurant and navigate directly using Gaode Map.</td><td>Did not use Gaode Map navigation.</td><td>Did not switch to Gaode Map.</td></tr><tr><td>Go to JD xxx official flagship store, search for "a certain mobile phone" in the store, check what colors are available for the 16+512GB configuration, and then tell WeChat friend 1.</td><td>Did not enter the official flagship store.</td><td>Did not search in the official flagship store.</td></tr><tr><td>Go to JD xxx official flagship store, forward the store detail page to Moments.</td><td>Could not find the store details page, only accessed the homepage.</td><td>Could not find the store details page.</td></tr><tr><td>Go to JD xxx official flagship store, share the homepage with WeChat friend 1.</td><td>Could not find the share UI.</td><td>Could not find the share option.</td></tr></table>
<table><tbody><tr><td>任务</td><td>真机失败原因</td><td>图像失败原因</td></tr><tr><td>在小红书搜索“广州旅游攻略”，将第一个分享到小红书好友Hh Yuan，并问“你最想去哪个地方？”然后在小红书搜索广州塔附近的美食，记住第一个餐厅的名字，在美团搜索该餐厅并点一份两人餐。</td><td>在美团搜索店铺时误点了广告。</td><td>未能在美团搜索到美食。</td></tr><tr><td>在小红书搜索“广州旅游攻略”，找到第一个，查看博主的粉丝、点赞和收藏，然后告诉微信好友1。</td><td>发送给微信好友的数据有误。</td><td>没有进入博主主页。</td></tr><tr><td>在小红书搜索“广州旅游攻略”，用“问答”让AI回复，并将生成的内容保存为图片到相册。</td><td>未找到“问答”。</td><td>保存为图片失败。</td></tr><tr><td>比较美团、大众点评和抖音上“新斗记广州塔”双人餐的价格，转发最便宜的给好友1。</td><td>在执行到美团后就结束了。</td><td>在美团之后搜索失败。</td></tr><tr><td>在美团搜索“巴赫利牛肉火锅”，把所有双人餐转发给微信好友1，分享后让他“选一个”。</td><td>没有分享，只是反复点击抢购。</td><td>只分享了一份套餐。</td></tr><tr><td>在美团点外卖“巴赫利牛肉火锅”：一份干锅牛肉，不要餐具。</td><td>误看成店内套餐而非外卖。</td><td>未准确找到外卖。</td></tr><tr><td>在美团搜索“巴赫利牛肉火锅”，选择第一个餐厅并用高德地图直接导航。</td><td>未使用高德地图导航。</td><td>未切换到高德地图。</td></tr><tr><td>进入京东xxx官方旗舰店，在店内搜索“某款手机”，查看16+512GB配置有哪些颜色，然后告诉微信好友1。</td><td>未进入官方旗舰店。</td><td>未在官方旗舰店内搜索。</td></tr><tr><td>进入京东xxx官方旗舰店，将店铺详情页转发到朋友圈。</td><td>找不到店铺详情页，只进入了首页。</td><td>找不到店铺详情页。</td></tr><tr><td>进入京东xxx官方旗舰店，与微信好友1分享首页。</td><td>找不到分享界面。</td><td>找不到分享选项。</td></tr></tbody></table>