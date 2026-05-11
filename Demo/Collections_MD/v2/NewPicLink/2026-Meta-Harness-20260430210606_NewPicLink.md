# Meta-Harness: End-to-End Optimization of Model Harnesses
# Meta-Harness：模型外壳的端到端优化


Yoonho Lee
Yoonho Lee


Stanford
斯坦福大学


Roshen Nair
Roshen Nair


Stanford
斯坦福大学


Qizheng Zhang
Qizheng Zhang


Stanford
斯坦福大学


Kangwook Lee
Kangwook Lee


KRAFTON
KRAFTON


Omar Khattab
Omar Khattab


MIT



Chelsea Finn
Chelsea Finn


Stanford
斯坦福大学


Project page w/ interactive demo: https://yoonholee.com/meta-harness/ Optimized harness: https://github.com/stanford-iris-lab/meta-harness-tbench2-artifact
项目主页及交互式演示：https://yoonholee.com/meta-harness/ 优化后的外壳代码：https://github.com/stanford-iris-lab/meta-harness-tbench2-artifact


<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/05/2026_05_11__23_09_36_daef5f.jpg"/>



Figure 1: (Left) On text classification, Meta-Harness outperforms the best prior hand-designed harnesses (ACE) and existing text optimizers (TTT-Discover, OpenEvolve), matching the next-best method's final accuracy after just 4 evaluations. (Right) On TerminalBench- 2, Meta-Harness outperforms all reported Claude Haiku 4.5 harnesses.
图 1：（左）在文本分类任务中，Meta-Harness 的表现优于此前最佳的人工设计外壳（ACE）及现有文本优化器（TTT-Discover、OpenEvolve），仅需 4 次评估即可达到次优方法的最终准确率。（右）在 TerminalBench-2 上，Meta-Harness 的表现优于所有已报道的 Claude Haiku 4.5 外壳。


## Abstract
## 摘要


The performance of large language model (LLM) systems depends not only on model weights, but also on their harness: the code that determines what information to store, retrieve, and present to the model. Yet harnesses are still designed largely by hand, and existing text optimizers are poorly matched to this setting because they compress feedback too aggressively: they are memoryless, condition only on scalar scores, or restrict feedback to short templates or summaries. We introduce Meta-Harness, an outer-loop system that searches over harness code for LLM applications. It uses an agentic proposer that accesses the source code, scores, and execution traces of all prior candidates through a filesystem. On online text classification, Meta-Harness improves over a state-of-the-art context management system by 7.7 points while using $4 \times$ fewer context tokens. On retrieval-augmented math reasoning, a single discovered harness improves accuracy on 200 IMO-level problems by 4.7 points on average across five held-out models. On agentic coding, discovered harnesses surpass the best hand-engineered baselines on TerminalBench-2. Together, these results show that richer access to prior experience can enable automated harness engineering.
大语言模型（LLM）系统的性能不仅取决于模型权重，还取决于其外壳（harness）：即决定向模型存储、检索及呈现何种信息的代码。然而，外壳目前仍主要依靠人工设计，且现有文本优化器因反馈压缩过于激进，难以适配此场景：它们往往是无记忆的，仅依赖标量分数，或将反馈限制在简短的模板或摘要中。我们引入了 Meta-Harness，这是一个针对 LLM 应用外壳代码进行搜索的外环系统。它利用一个代理提案器，通过文件系统访问所有先前候选方案的源代码、分数及执行轨迹。在在线文本分类任务中，Meta-Harness 在使用 $4 \times$ 更少上下文 Token 的情况下，较最先进的上下文管理系统提升了 7.7 个百分点。在检索增强数学推理任务中，单个被发现的外壳在五个留出模型上，使 200 道 IMO 级别问题的平均准确率提升了 4.7 个百分点。在代理编码任务中，被发现的外壳在 TerminalBench-2 上超越了最佳人工工程基准。综上所述，这些结果表明，更丰富地利用过往经验能够实现自动化的外壳工程。


## 1 Introduction
## 1 引言


Changing the harness around a fixed large language model (LLM) can produce a $6 \times$ performance gap on the same benchmark [47]. The harness-the code that determines what to store, retrieve, and show to the model-often matters as much as the model itself. This sensitivity has led to growing interest in harness engineering, the practice of refining the code around an LLM to improve the overall system's performance [36,21,10,9]. But despite its importance, harness engineering remains largely manual: practitioners inspect failures,
在固定大语言模型（LLM）的前提下，改变其外围的调用框架（harness）可在同一基准测试上产生 $6 \times$ 的性能差距 [47]。调用框架——即决定向模型存储、检索及展示何种信息的代码——往往与模型本身同样重要。这种敏感性引发了对“调用框架工程”的日益关注，即通过优化 LLM 周围的代码来提升系统整体性能的实践 [36,21,10,9]。然而，尽管其至关重要，调用框架工程目前仍主要依赖人工：从业者通过检查失败案例，


<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/05/2026_05_11__23_09_36_0cbbc7.jpg"/>



Figure 2: Meta-Harness search loop. (1) An agent reads a filesystem containing all prior candidates' source code, execution traces, and scores, and proposes a new harness. (2) We evaluate the proposed harness on evaluation tasks. (3) All logs (proposed code, reasoning traces, evaluation scores) are stored in the filesystem in a new directory, and the loop repeats.
图 2：Meta-Harness 搜索循环。(1) 智能体读取包含所有先前候选方案源代码、执行轨迹和评分的文件系统，并提出一个新的调用框架。(2) 我们在评估任务上对提出的调用框架进行评估。(3) 所有日志（提出的代码、推理轨迹、评估分数）被存储在文件系统的新目录中，随后循环重复。


<table><tr><td>Method</td><td>History</td><td>Log content</td><td>MTok/iter</td></tr><tr><td>OPRO 51</td><td>Window</td><td>past (solution, score) pairs</td><td>0.002</td></tr><tr><td>TextGrad |53|</td><td>Last</td><td>textual feedback on current artifact</td><td>0.015</td></tr><tr><td>AlphaEvolve 35</td><td>Window</td><td>program database + eval. scores</td><td>0.022</td></tr><tr><td>GEPA [1]</td><td>Summary</td><td>reflective feedback from rollout traces</td><td>0.008</td></tr><tr><td>Feedback Descent 26</td><td>Summary</td><td>comparison + textual feedback</td><td>0.012</td></tr><tr><td>TTT-Discover |54|</td><td>Window</td><td>prev. solution fragment</td><td>0.026</td></tr><tr><td>Meta-Harness</td><td>Full</td><td>all logs and scores</td><td>10.0</td></tr></table>
<table><tbody><tr><td>方法</td><td>历史记录</td><td>日志内容</td><td>百万词/迭代</td></tr><tr><td>OPRO 51</td><td>窗口</td><td>过往（解法，分数）对</td><td>0.002</td></tr><tr><td>TextGrad |53|</td><td>最后</td><td>当前制品的文本反馈</td><td>0.015</td></tr><tr><td>AlphaEvolve 35</td><td>窗口</td><td>程序库 + 评估分数</td><td>0.022</td></tr><tr><td>GEPA [1]</td><td>摘要</td><td>来自执行轨迹的反思性反馈</td><td>0.008</td></tr><tr><td>Feedback Descent 26</td><td>摘要</td><td>比较 + 文本反馈</td><td>0.012</td></tr><tr><td>TTT-Discover |54|</td><td>窗口</td><td>先前的解法片段</td><td>0.026</td></tr><tr><td>Meta-Harness</td><td>全部</td><td>所有日志和分数</td><td>10.0</td></tr></tbody></table>


Table 1: Comparison of text optimization methods and their settings. Each row represents a method collapsed across tasks. Mtok/iter is our best estimate of the full context generated from one evaluation of a text artifact in the largest setting considered in each paper. This paper considers settings that yield orders-of-magnitude more context per artifact evaluation.
表1：文本优化方法及其设置的比较。每一行代表跨任务合并后的方法。“每迭代百万Token数”（Mtok/iter）是我们对各论文所考虑的最大设置下，评估单个文本工件所生成的完整上下文的最佳估算值。本文所考虑的设置在每次工件评估中产生的上下文量级要高出几个数量级。


adjust heuristics, and iterate on a small number of designs. In this paper, we ask whether this process itself can be automated.
调整启发式规则，并对少量设计进行迭代。在本文中，我们探讨这一过程本身能否实现自动化。


A natural starting point is recent work on text optimization, since harness engineering also involves iteratively improving text and code artifacts using feedback from prior attempts 38 393526 [1]. However, these methods are poorly matched to harness engineering because they typically operate with short-horizon or heavily compressed feedback: some condition only on the current candidate [31, 51, 53], others rely primarily on scalar scores [35, 12], and others restrict feedback to short templates or LLM-generated summaries [1, 26]. This is a pragmatic scalability choice, not evidence that longer-range dependencies are uninformative. Harnesses act over long horizons: a single choice about what to store, when to retrieve it, or how to present it can affect behavior many reasoning steps later. Compressed feedback often removes the information needed to trace downstream failures to earlier harness decisions. Across the tasks studied by several representative text optimizers, the available context per optimization step ranges from only 100 to 30,000 tokens (Table 1), far below the diagnostic footprint of harness search. More broadly, work on retrieval and memory-augmented language models suggests that useful context should often be accessed adaptively rather than monolithically packed into a single prompt [28, 48, 37, 56].
一个自然的起点是近期关于文本优化的工作，因为模型框架工程也涉及利用先前尝试的反馈来迭代改进文本和代码工件 [1]。然而，这些方法与模型框架工程并不匹配，因为它们通常使用短视或高度压缩的反馈：有些仅基于当前候选方案 [31, 51, 53]，有些主要依赖标量分数 [35, 12]，还有些将反馈限制在短模板或大语言模型生成的摘要中 [1, 26]。这是一种实用的可扩展性选择，并非表明长距离依赖没有信息价值。模型框架的作用具有长时性：关于存储内容、检索时间或呈现方式的单一选择，可能会在多个推理步骤后影响行为。压缩反馈往往会去除将下游失败追溯到早期框架决策所需的信息。在几个有代表性的文本优化器所研究的任务中，每个优化步骤的可用上下文范围仅为 100 到 30,000 个标记（表 1），远低于模型框架搜索的诊断范围。更广泛地说，关于检索和记忆增强语言模型的研究表明，有用的上下文通常应自适应地访问，而不是一次性打包到单个提示中 [28, 48, 37, 56]。


We address this limitation with Meta-Harness, an agentic harness for optimizing harnesses via end-to-end search (Figure 2). Its proposer is a coding agent, i.e., a language-model-based system that can invoke developer tools and modify code. The choice of coding agent (rather than raw LLM) matters because the amount of experience quickly exceeds context limits, so the proposer must decide what to inspect and validate edits through direct interaction with the codebase. Its key design choice is to expose full history through a filesystem, enabling selective diagnosis of raw prior code and execution traces rather than optimization from compressed per-candidate summaries. For every previous candidate harness, the filesystem stores the source code, evaluation scores, and execution traces, which the proposer retrieves via standard operations such as grep and cat rather than ingesting them as a single prompt. In practice, the proposer reads a median of 82 files per iteration in our most demanding setting, referencing over 20 prior candidates per step (Appendix A). In the settings we study, a single evaluation can produce up to 10,000,000 tokens of diagnostic information, roughly three orders of magnitude beyond the largest feedback budgets used in prior text optimization settings (Table 1).
我们通过 Meta-Harness 解决了这一局限性，这是一种通过端到端搜索来优化工具链的智能体工具链（图2）。其提议者是一个编码智能体，即一种可以调用开发者工具并修改代码的基于语言模型的系统。选择编码智能体（而非原始大模型）至关重要，因为经验量会迅速超出上下文限制，因此提议者必须决定检查什么，并通过与代码库的直接交互来验证修改。其核心设计选择是通过文件系统暴露完整历史记录，从而实现对原始先前代码和执行轨迹的选择性诊断，而非基于每个候选方案的压缩摘要进行优化。对于每一个先前的候选工具链，文件系统都会存储源代码、评估分数和执行轨迹，提议者通过 grep 和 cat 等标准操作来检索这些信息，而不是将其作为单个提示词摄入。在实践中，在最严苛的设置下，提议者每次迭代平均读取 82 个文件，每步引用超过 20 个先前的候选方案（附录 A）。在我们研究的设置中，单次评估可产生多达 10,000,000 个 Token 的诊断信息，比先前文本优化设置中使用的最大反馈预算高出约三个数量级（表1）。


We evaluate Meta-Harness on online text classification, mathematical reasoning, and agentic coding. On online text classification, harnesses discovered by Meta-Harness improve over Agentic Context Engineering (ACE,Zhang et al. [59]) by 7.7 points while using $4 \times$ fewer context tokens, and match the next-best text optimizer's final performance after 60 proposals with only four (Figure 1). On retrieval-augmented math reasoning, a single discovered harness improves accuracy on 200 IMO-level problems by 4.7 points on average across five held-out models. On TerminalBench-2, the discovered harness surpasses Terminus-KIRA and ranks #1 among all Haiku 4.5 agents.
我们在在线文本分类、数学推理和智能体编码任务上评估了 Meta-Harness。在在线文本分类任务中，Meta-Harness 发现的工具链（harnesses）在比 Agentic Context Engineering (ACE, Zhang et al. [59]) 少使用 $4 \times$ 个上下文 token 的情况下，性能提升了 7.7 个百分点，且仅需 4 次提案即可达到其他最优文本优化器 60 次提案后的最终性能（图 1）。在检索增强数学推理任务中，单个发现的工具链在 5 个留出模型上，使 200 道 IMO 级别问题的平均准确率提升了 4.7 个百分点。在 TerminalBench-2 上，该工具链超越了 Terminus-KIRA，在所有 Haiku 4.5 智能体中排名第一。


## 2 Related Work
## 2 相关工作


At a high level, Meta-Harness brings ideas from the broader literature on credit assignment and meta-learning [40, 46, 37, 44, 2] in a new regime enabled by recent advances in coding agents. Rather than updating model weights, the system assigns credit at the harness level: it uses experience from past rollouts to deliberately reason about which steps and components are responsible for failures, then rewrites the external code that governs future behavior. More specifically, the method lies at the intersection of several recent research threads; it is most directly related to work on adaptive access to external context, executable code search, and text optimization.
从宏观层面来看，Meta - Harness 在近期编码代理技术进步所开创的新领域中，引入了更广泛的信用分配和元学习文献中的理念[40, 46, 37, 44, 2]。该系统并非更新模型权重，而是在测试框架层面进行信用分配：它利用过去推演的经验，审慎地推断哪些步骤和组件导致了失败，然后重写控制未来行为的外部代码。更具体地说，该方法处于多个近期研究方向的交叉点；它与自适应访问外部上下文、可执行代码搜索和文本优化方面的工作最为直接相关。


External memory and adaptive access. Several prior works note the benefits of treating large knowledge sources or long inputs as external resources that a language model accesses adaptively, rather than consuming them in a single pass. Specifically, retrieval-augmented generation [28], interleaved retrieval and reasoning [48], memory-based agents [37], or recursive language models [56] are mechanisms for adaptive access to external context. Meta-Harness uses a similar access pattern, but in the more demanding setting of harness engineering, where the proposer selectively inspects a large external history of code, scores, and execution traces to improve context-management procedures themselves.
外部内存与自适应访问。一些先前的研究指出，将大型知识源或长输入视为语言模型可自适应访问的外部资源，而非一次性处理，具有诸多好处。具体而言，检索增强生成 [28]、交错检索与推理 [48]、基于内存的智能体 [37] 或递归语言模型 [56] 都是自适应访问外部上下文的机制。Meta-Harness 采用了类似的访问模式，但应用于更具挑战性的测试框架工程场景中，其中提议者会有选择地检查代码、分数和执行轨迹的大量外部历史记录，以改进上下文管理程序本身。


Executable code search. Recent methods search over executable code for functions, work-flows, or agent designs. Early work proposes using large models as mutation and crossover operators in evolutionary program search [27]. Later methods evolve designated functions within fixed program scaffolds [39], use meta-agents to program new agents from prior discoveries [20], or search over workflow graphs for agentic systems [58]. Another line of work searches over memory designs for continual-learning agents, where memory persists across task streams [57, 50]. In contrast, Meta-Harness searches over domain-specific harnesses, including prompt construction, retrieval, and state update strategies that reset between tasks. Its outer loop is deliberately minimal: instead of relying on a fixed scaffold, an archive of prior discoveries, or a persistent memory mechanism, it gives the proposer unrestricted filesystem access to prior experience. This lets the agent decide what information to inspect and enables search over full harness implementations rather than a predefined space of context-management procedures.
可执行代码搜索。近期的方法会在可执行代码中搜索函数、工作流或智能体设计。早期的工作提出在进化程序搜索中使用大模型作为变异和交叉算子 [27]。后来的方法在固定的程序框架内进化指定的函数 [39]，使用元智能体根据先前的发现来编程新的智能体 [20]，或者在智能体系统的工作流图中进行搜索 [58]。另一类工作则是在持续学习智能体的内存设计中进行搜索，其中内存在任务流中持续存在 [57, 50]。相比之下，Meta-Harness 在特定领域的测试框架中进行搜索，包括提示构造、检索和在任务之间重置的状态更新策略。它的外循环经过精心设计，极为精简：它不依赖于固定的框架、先前发现的存档或持久的内存机制，而是给予提议者不受限制的文件系统访问权限，使其能够获取先前的经验。这让智能体能够决定检查哪些信息，并能够在完整的测试框架实现中进行搜索，而不是在预定义的上下文管理程序空间中进行搜索。


Text optimization methods. Meta-Harness is also closely related to methods such as ProTeGi, TextGrad, OPRO, GEPA, AlphaEvolve/OpenEvolve, and Feedback Descent, which iteratively improve prompts or other text artifacts using feedback from prior attempts [38] 31,53,51,13,54,32,56]. However, these methods are less well suited to harness engineering, where optimization targets a complete executable procedure, and the relevant environmental feedback is distributed across code, scores, and execution traces in a way that is hard to summarize up front. Rather than reacting only to aggregate scores or summaries, the proposer in Meta-Harness can reason over failed examples and their execution traces to propose targeted edits. See Table 1 for a comparison of problem scale considered in those papers and ours, and Figures 1 and 4 for a direct comparison with OpenEvolve, GEPA, and TTT-Discover in our problem setting.
文本优化方法。Meta-Harness 还与 ProTeGi、TextGrad、OPRO、GEPA、AlphaEvolve/OpenEvolve 和反馈下降等方法密切相关，这些方法利用先前尝试的反馈来迭代改进提示或其他文本工件 [38,31,53,51,13,54,32,56]。然而，这些方法不太适合框架工程，因为在框架工程中，优化的目标是一个完整的可执行程序，并且相关的环境反馈分布在代码、分数和执行轨迹中，很难预先总结。Meta-Harness 中的提议者不是仅对汇总分数或摘要做出反应，而是可以对失败的示例及其执行轨迹进行推理，以提出有针对性的编辑。有关这些论文和我们的论文所考虑的问题规模的比较，请参见表 1；有关在我们的问题设置中与 OpenEvolve、GEPA 和 TTT-Discover 的直接比较，请参见图 1 和图 4。


## 3 Meta-Harness: A Harness for Optimizing Harnesses
## 3 Meta-Harness：用于优化工具框架的工具框架


This section describes Meta-Harness, our outer-loop procedure for searching over task-specific harnesses. Meta-Harness is built on the idea that harness optimization benefits from allowing a proposer to selectively inspect prior code and execution traces via filesystem access, rather than optimizing from lossy summaries or an additional hand-designed search structure. At a high level, it repeatedly proposes, evaluates, and logs new harnesses.
本节介绍 Meta-Harness，即我们用于搜索特定任务工具框架的外循环过程。Meta-Harness 基于这样一个理念：工具框架的优化得益于允许提议者通过文件系统访问选择性地检查既往代码和执行轨迹，而非基于有损摘要或额外的人工设计搜索结构进行优化。从宏观上看，它通过重复提出、评估和记录新的工具框架来实现优化。


Meta-Harness is itself a harness in the broad sense (hence the name), since it determines what information the proposer model sees during search. Unless otherwise noted, we use harness to refer to the task-specific programs being optimized.
Meta-Harness 本身在广义上也是一种工具框架（因此得名），因为它决定了提议者模型在搜索过程中能看到哪些信息。除非另有说明，我们所指的“工具框架”均为正在被优化的特定任务程序。


Objective. A harness is a stateful program that wraps a language model and determines what context the model sees at each step. The goal is simple: find the harness that makes the underlying model perform best on the target task distribution. Formally,let $M$ denote a fixed language model and $\mathcal{X}$ a task distribution. For a harness $H$ and task instance $x \sim  \mathcal{X}$ , we execute a rollout trajectory $\tau  \sim  {p}_{M}\left( {H,x}\right)$ . The harness constructs prompts for $M$ ,the model responds, and the harness updates its state after each interaction. A task-specific reward function $r\left( {\tau ,x}\right)$ scores the trajectory. The objective of harness optimization is to find the harness that maximizes the expected final reward:
目标。工具框架是一个有状态的程序，它封装了语言模型并决定模型在每一步所见的上下文。其目标很简单：找到能使底层模型在目标任务分布上表现最优的工具框架。形式上，令 $M$ 表示固定的语言模型，$\mathcal{X}$ 表示任务分布。对于工具框架 $H$ 和任务实例 $x \sim  \mathcal{X}$，我们执行一次展开轨迹 $\tau  \sim  {p}_{M}\left( {H,x}\right)$。工具框架为 $M$ 构建提示词，模型做出响应，随后工具框架在每次交互后更新其状态。特定任务的奖励函数 $r\left( {\tau ,x}\right)$ 对该轨迹进行评分。工具框架优化的目标是找到使预期最终奖励最大化的工具框架：


$$
{H}^{ * } = \underset{H}{\arg \max }{\mathbb{E}}_{x \sim  \mathcal{X},\tau  \sim  {p}_{M}\left( {H,x}\right) }r\left( {\tau ,x}\right) ,
$$



When multiple objectives are relevant (e.g., accuracy and context cost), we evaluate candidates under Pareto dominance and report the resulting frontier. In practice, this search has traditionally been carried out by human engineers and researchers, who iteratively refine prompts, context-management rules, and tool-use logic by hand.
当存在多个相关目标（例如准确率和上下文成本）时，我们根据帕累托支配原则评估候选方案并报告所得的前沿面。在实践中，这种搜索传统上由人类工程师和研究人员完成，他们通过手动迭代优化提示词、上下文管理规则和工具使用逻辑。


Meta-Harness search loop. Meta-Harness uses a single coding-agent proposer with access to a growing filesystem $\mathcal{D}$ that serves as its feedback channel language-model-based system that can invoke developer tools and modify code. Unlike prior systems that externalize the improvement logic in a hand-designed search loop, Meta-Harness delegates diagnosis and proposal to the coding agent itself: it decides which prior artifacts to inspect, which failure modes to address, and whether to make a local edit or a more substantial rewrite. Equivalently, the proposer is not a raw next-token model operating on a fixed prompt assembled by the outer loop; it is an agent that retrieves information, navigates prior artifacts, and edits code as part of the search itself. Each evaluated harness contributes a directory containing its source code, scores, and execution traces (such as prompts, tool calls, model outputs, and state updates). The filesystem is typically far larger than the proposer's context window, so the proposer queries it through terminal tools such as grep and cat rather than ingesting it as a single prompt. At each iteration, the proposer first inspects prior code, scores, and execution traces, then reasons about likely failure modes before generating a new harness.
Meta-Harness 搜索循环。Meta-Harness 使用单个编码代理提议者，该提议者可访问一个不断增长的文件系统 $\mathcal{D}$，作为其基于语言模型的反馈通道，能够调用开发工具并修改代码。与以往将改进逻辑外置于人工设计搜索循环的系统不同，Meta-Harness 将诊断和提议任务委托给编码代理本身：它决定检查哪些先前的工件、解决哪些故障模式，以及是进行局部编辑还是进行更实质性的重写。换言之，提议者并非在外部循环组装的固定提示词上运行的原始下一个词预测模型；它是一个在搜索过程中检索信息、导航先前工件并编辑代码的代理。每个被评估的工具链（harness）都会贡献一个包含其源代码、分数和执行轨迹（如提示词、工具调用、模型输出和状态更新）的目录。由于文件系统通常远大于提议者的上下文窗口，提议者通过 grep 和 cat 等终端工具进行查询，而非将其作为单一提示词输入。在每次迭代中，提议者首先检查先前的代码、分数和执行轨迹，然后在生成新的工具链之前推断可能的故障模式。


Meta-Harness maintains a population $\mathcal{H}$ and a Pareto frontier over evaluated harnesses,but imposes no parent-selection rule: the proposer is free to inspect any prior harness and its execution trace when proposing new ones. We run evolution for a fixed number of iterations and perform a final test-set evaluation on the Pareto frontier. This simplicity is deliberate: by leaving diagnosis and edit decisions to the proposer rather than hard-coding search heuristics, Meta-Harness can improve automatically as coding agents become more capable. The proposer never sees test-set results; its only feedback comes from the search set, the subset of task instances used to evaluate candidate harnesses during search and generate the feedback signal for improvement, and from execution traces logged during those search runs.
Meta-Harness 维护一个种群 $\mathcal{H}$ 和已评估工具链的帕累托前沿，但不强制执行父代选择规则：提议者在提出新工具链时，可以自由检查任何先前的工具链及其执行轨迹。我们运行固定次数的进化迭代，并对帕累托前沿进行最终测试集评估。这种简单性是有意为之：通过将诊断和编辑决策留给提议者，而非硬编码搜索启发式算法，Meta-Harness 可以随着编码代理能力的提升而自动改进。提议者从不查看测试集结果；其唯一的反馈来自搜索集（即在搜索过程中用于评估候选工具链并生成改进反馈信号的任务实例子集）以及在这些搜索运行期间记录的执行轨迹。


Advantages of code-space search. Harness optimization occurs in code space, where small changes to retrieval, memory, or prompt-construction logic can affect behavior many steps later, making local search heuristics poorly matched to the problem. By inspecting execution traces, the proposer can often infer why a harness failed and which earlier design choices likely contributed to the failure, not just that it failed, as illustrated by the search trajectories in Appendices A and A.2. There, we see that the proposer reads broadly across prior code and logs, then uses those traces to identify confounded edits, isolate likely causal changes, and shift toward safer modifications after repeated regressions. The proposer can therefore modify the harness at the level of algorithmic structure, ranging from changes to retrieval, memory, or prompt-construction logic to full program rewrites, rather than filling in templates or applying predefined mutation operators. In practice, it often starts from a strong prior harness, but this is an emergent strategy rather than a hard-coded rule. Although the search space is large, representing harnesses as programs provides a natural regularization bias: coding models tend to propose coherent algorithms rather than brittle, hard-coded solutions, which biases the search toward reusable context-management procedures. This action space is closely aligned with the read-write-execute workflows on which frontier coding assistants are trained.
代码空间搜索的优势。工具链优化发生在代码空间中，对检索、记忆或提示词构建逻辑的微小改动可能会在许多步骤后影响行为，这使得局部搜索启发式算法难以匹配该问题。通过检查执行轨迹，提议者通常能推断出工具链失败的原因以及哪些早期的设计选择可能导致了失败，而不仅仅是知道它失败了，正如附录 A 和 A.2 中的搜索轨迹所示。在那里，我们看到提议者广泛阅读先前的代码和日志，然后利用这些轨迹识别混淆的编辑，隔离可能的因果变化，并在反复回归后转向更安全的修改。因此，提议者可以在算法结构层面修改工具链，范围涵盖从检索、记忆或提示词构建逻辑的变更到完整的程序重写，而不是仅仅填充模板或应用预定义的变异算子。在实践中，它通常从一个强大的先验工具链开始，但这是一种涌现策略而非硬编码规则。尽管搜索空间很大，但将工具链表示为程序提供了一种自然的正则化偏差：编码模型倾向于提出连贯的算法，而非脆弱的硬编码解决方案，这使搜索偏向于可重用的上下文管理过程。该动作空间与前沿编码助手所训练的读-写-执行工作流高度一致。


---



${}^{1}$ Based on earlier exploration,we think this workflow only became practical recently,following major improvements in coding-agent capabilities around early 2026.
${}^{1}$ 基于早期的探索，我们认为这种工作流直到 2026 年初编码代理能力取得重大进展后才变得切实可行。


---



Algorithm 1 Meta-Harness outer loop over harnesses
算法 1 Meta-Harness 工具链外循环


---



Input: tasks $\mathcal{X}$ ,LLM $M$ ,proposer $P$ ,iterations $N$
输入：任务 $\mathcal{X}$，LLM $M$，提议者 $P$，迭代次数 $N$


Initialize: population $\mathcal{H}$ 																						$\vartriangleright$ Initial set of valid harnesses
初始化：种群 $\mathcal{H}$ $\vartriangleright$ 初始有效工具链集合


Initialize: filesystem $\mathcal{D} \leftarrow  \varnothing$ 																							$\vartriangleright$ stores code,scores,traces
初始化：文件系统 $\mathcal{D} \leftarrow  \varnothing$ $\vartriangleright$ 存储代码、分数、轨迹


for $H \in  \mathcal{H}$ do
对于 $H \in  \mathcal{H}$ 执行


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;${E}_{H} \leftarrow  \operatorname{Evaluate}\left( {H,M,\mathcal{X}}\right)$



&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$\mathcal{D} \leftarrow  \mathcal{D} \cup  \left\{  \left( {H,{E}_{H}}\right) \right\}$



for $t = 1\ldots N$ do
对于 $t = 1\ldots N$ 执行


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Proposer $P$ queries filesystem $\mathcal{D}\; \vartriangleright$ inspects prior harnesses and scores
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;提议者 $P$ 查询文件系统 $\mathcal{D}\; \vartriangleright$ 检查先前的工具链和分数


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Proposer $P$ proposes $k$ new harnesses $\left\{  {{H}_{1},\ldots ,{H}_{k}}\right\}$
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;提议者 $P$ 提出 $k$ 新工具链 $\left\{  {{H}_{1},\ldots ,{H}_{k}}\right\}$


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;for $H$ in $\left\{  {{H}_{1},\ldots ,{H}_{k}}\right\}$ do
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;对于 $H$ 中的 $\left\{  {{H}_{1},\ldots ,{H}_{k}}\right\}$ 执行


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;if $H$ passes interface validation then
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;如果 $H$ 通过接口验证则


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$\mathcal{D} \leftarrow  \mathcal{D} \cup  \{ \left( {H,\operatorname{EVALUATE}\left( {H,M,\mathcal{X}}\right) }\right) \}$



return Pareto frontier of harnesses stored in $\mathcal{D}$
返回存储在 $\mathcal{D}$ 中的工具链帕累托前沿


---



Practical implementation. In our experiments, each harness is a single-file Python program that modifies task-specific prompting, retrieval, memory, and orchestration logic. In our experiments,the proposer $P$ is Claude Code [4] with Opus-4.6. The proposer is guided by a minimal domain-specific skill that describes where to write new harnesses, how to inspect previous harnesses and their execution traces, and what files it can and cannot modify. The base model $M$ varies by domain and is always frozen; see Section 4 for details. In our experiments, a typical run evaluates roughly 60 harnesses over 20 iterations. We provide additional tips for implementing Meta-Harness in a new domain in Appendix D.
实际实现。在我们的实验中，每个 harness 都是一个单文件 Python 程序，用于修改特定任务的提示词、检索、记忆和编排逻辑。在我们的实验中，提案者 $P$ 为使用 Opus-4.6 的 Claude Code [4]。提案者受限于一种最小化的领域特定技能，该技能描述了在何处编写新 harness、如何检查之前的 harness 及其执行轨迹，以及它可以或不可以修改哪些文件。基础模型 $M$ 因领域而异且始终保持冻结；详情请参阅第 4 节。在我们的实验中，一次典型运行会在 20 次迭代中评估约 60 个 harness。我们在附录 D 中提供了在特定新领域实现 Meta-Harness 的额外建议。


## 4 Experiments
## 4 实验


We evaluate Meta-Harness on three task domains: online text classification, math reasoning, and agentic coding. In each domain, we compare harnesses discovered by our search against domain-appropriate baselines using the standard evaluation metric. Please refer to each subsection for the precise experimental setup.
我们在三个任务领域评估了 Meta-Harness：在线文本分类、数学推理和智能体编码。在每个领域中，我们都将搜索发现的 harness 与使用标准评估指标的领域适用基线进行了比较。请参阅各小节以了解精确的实验设置。


We compare against two main classes of methods. (1) Human-designed strategies: these are hand-crafted harnesses for each domain, representing the current state of the art in context construction. We describe these baselines in the corresponding subsections. (2) Program-search methods: these methods search over candidate harnesses using feedback and reward signals, but are designed for smaller-scale settings than harness engineering.
我们主要与两类方法进行比较。(1) 人工设计策略：这些是针对每个领域手工制作的 harness，代表了当前上下文构建的最高水平。我们在相应的小节中描述了这些基线。(2) 程序搜索方法：这些方法利用反馈和奖励信号在候选 harness 中进行搜索，但其设计初衷是针对比 harness 工程规模更小的场景。


### 4.1 Online Text Classification
### 4.1 在线文本分类


We follow the online text classification setup of Zhang et al. [59]; Ye et al. [52]: an LLM receives labeled examples one at a time, updates its memory, and is evaluated on a held-out test set. We use GPT-OSS-120B as the LLM text classifier, and consider the problem of designing a harness for text classification. We use three datasets, chosen for difficulty and domain diversity: LawBench (Law) [16] predicts criminal charges from case descriptions (215 classes); Symptom2Disease (S2D) [19] predicts diseases from symptom descriptions (22 classes); and USPTO-50k [41] predicts precursor reactants from product molecules (180 classes). We initialize the search population $\mathcal{H}$ from the main baseline harnesses in this setting: zero-shot, few-shot, ACE, and MCE. We ran 20 evolution iterations with two candidates per iteration, producing 40 candidate harnesses.
我们遵循 Zhang 等人 [59] 和 Ye 等人 [52] 的在线文本分类设置：LLM 逐个接收带标签的示例，更新其记忆，并在留出的测试集上进行评估。我们使用 GPT-OSS-120B 作为 LLM 文本分类器，并考虑为文本分类设计 harness 的问题。我们使用了三个因难度和领域多样性而选定的数据集：LawBench (Law) [16] 根据案例描述预测刑事指控（215 个类别）；Symptom2Disease (S2D) [19] 根据症状描述预测疾病（22 个类别）；以及 USPTO-50k [41] 根据产物分子预测前体反应物（180 个类别）。我们从该设置下的主要基线 harness 初始化搜索种群 $\mathcal{H}$：zero-shot、few-shot、ACE 和 MCE。我们进行了 20 次进化迭代，每次迭代产生两个候选者，共生成 40 个候选 harness。


<table><tr><td rowspan="2">Harness</td><td colspan="3">Datasets</td><td colspan="2">Avg.</td></tr><tr><td>USPTO</td><td>S2D</td><td>Law</td><td>Acc</td><td>Ctx↓</td></tr><tr><td>Zero-Shot</td><td>12.0</td><td>63.2</td><td>7.0</td><td>27.4</td><td>0</td></tr><tr><td>Few-Shot (8)</td><td>14.0</td><td>67.9</td><td>21.0</td><td>34.3</td><td>2.0</td></tr><tr><td>Few-Shot (32)</td><td>13.0</td><td>72.2</td><td>21.0</td><td>35.4</td><td>7.9</td></tr><tr><td>Few-Shot (all)</td><td>15.0</td><td>78.3</td><td>29.0</td><td>40.8</td><td>12.3</td></tr><tr><td>MCE [52]†</td><td>14.0</td><td>83.0</td><td>23.0</td><td>40.0</td><td>28.5</td></tr><tr><td>ACE [59]†</td><td>16.0</td><td>77.8</td><td>29.0</td><td>40.9</td><td>50.8</td></tr><tr><td>Meta-Harness</td><td>14.0</td><td>86.8</td><td>45.0</td><td>48.6</td><td>11.4</td></tr></table>
<table><tbody><tr><td rowspan="2">Harness</td><td colspan="3">数据集</td><td colspan="2">平均值</td></tr><tr><td>USPTO</td><td>S2D</td><td>Law</td><td>准确率</td><td>上下文↓</td></tr><tr><td>零样本</td><td>12.0</td><td>63.2</td><td>7.0</td><td>27.4</td><td>0</td></tr><tr><td>少样本 (8)</td><td>14.0</td><td>67.9</td><td>21.0</td><td>34.3</td><td>2.0</td></tr><tr><td>少样本 (32)</td><td>13.0</td><td>72.2</td><td>21.0</td><td>35.4</td><td>7.9</td></tr><tr><td>少样本 (全部)</td><td>15.0</td><td>78.3</td><td>29.0</td><td>40.8</td><td>12.3</td></tr><tr><td>MCE [52]†</td><td>14.0</td><td>83.0</td><td>23.0</td><td>40.0</td><td>28.5</td></tr><tr><td>ACE [59]†</td><td>16.0</td><td>77.8</td><td>29.0</td><td>40.9</td><td>50.8</td></tr><tr><td>Meta-Harness</td><td>14.0</td><td>86.8</td><td>45.0</td><td>48.6</td><td>11.4</td></tr></tbody></table>


Table 2: Test-set metrics for all harnesses on the three datasets. Ctx denotes additional input tokens in context (thousands). †: implementation from Ye et al. [52]. $\downarrow$ : lower is better. Meta-Harness improves online text classification accuracy while using a smaller input context.
表 2：三个数据集上所有测试工具的指标。Ctx 表示上下文中额外的输入标记（千）。†：来自 Ye 等人 [52] 的实现。$\downarrow$：数值越低越好。Meta-Harness 在使用更小输入上下文的同时，提高了在线文本分类的准确率。


<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/05/2026_05_11__23_09_36_2a5a46.jpg"/>



Figure 3: Pareto frontier of accuracy vs. context tokens on online text classification. Meta-Harness achieves a stronger accuracy-context Pareto frontier than all comparison methods.
图 3：在线文本分类中准确率与上下文标记的帕累托前沿。Meta-Harness 实现了比所有对比方法更强的准确率-上下文帕累托前沿。


Comparison vs text optimizers. We compare Meta-Harness against representative methods for optimizing text. For a fair comparison, we use the same proposer configuration (Opus-4.6 with max reasoning), select candidates solely based on search-set performance, and hold out the test sets until the final evaluation. Since evaluation is the main computational bottleneck, we give each method the same budget of proposal harness evaluations. We consider the following points of comparison:
与文本优化器的比较。我们将 Meta-Harness 与代表性的文本优化方法进行了对比。为确保公平，我们使用相同的提议者配置（Opus-4.6，最大推理），仅根据搜索集的表现选择候选方案，并在最终评估前保留测试集。由于评估是主要的计算瓶颈，我们为每种方法提供了相同的提议工具评估预算。我们考虑了以下对比点：


- Best-of-N: independent samples from the seed with no search structure; a compute-matched control for whether search matters at all.
- Best-of-N：从种子中进行独立采样，无搜索结构；作为计算量匹配的对照组，以验证搜索是否重要。


- OpenEvolve [43]: evolutionary search over programs with LLM mutation.
- OpenEvolve [43]：基于大模型变异的程序演化搜索。


- TTT-Discover [55]: we use only the text-optimization component of their method, i.e., proposal selection via the PUCT reuse rule.
- TTT-Discover [55]：我们仅使用其方法中的文本优化组件，即通过 PUCT 重用规则进行提议选择。


In this setting, Meta-Harness matches the best prior text optimizers (OpenEvolve, TTT-Discover) in ${0.1} \times$ the evaluations,and its final accuracy surpasses theirs by more than 10 points (Figure 1 and Table 4). We attribute this speedup to the intentional design choices that impose minimum necessary structure on the outer loop (Section 3). In particular, Meta-Harness preserves full experience history using a filesystem and allows the proposer to inspect anything necessary, whereas both OpenEvolve and TTT-Discover operate with more structured and substantially more limited proposer inputs than full filesystem access. We note that online text classification is the smallest-context setting we study (Table 1), so if structure-heavy text optimizers already lag here, their limitations may only grow in harder regimes.
在此设置下，Meta-Harness 在 ${0.1} \times$ 次评估中与最优的现有文本优化器（OpenEvolve、TTT-Discover）持平，且其最终准确率超过它们 10 个百分点以上（图 1 和表 4）。我们将这种加速归功于外循环中施加最小必要结构的有意识设计选择（第 3 节）。特别是，Meta-Harness 通过文件系统保留了完整的经验历史，并允许提议者检查任何必要内容，而 OpenEvolve 和 TTT-Discover 的提议者输入比完整的文件系统访问更具结构性且受到极大限制。我们注意到，在线文本分类是我们研究中上下文最小的设置（表 1），因此如果结构繁重的文本优化器在此处已显滞后，它们的局限性在更难的场景中可能会进一步扩大。


## Meta-Harness is ${10} \times$ Faster and Converges to a Better Harness
## Meta-Harness 的速度提升了 ${10} \times$ 倍，并收敛至更优的工具


In this setting, Meta-Harness matches the best prior text optimizers (OpenEvolve, TTT-Discover) with ${10} \times$ fewer full evaluations,and its final accuracy surpasses theirs by more than 10 points.
在此设置下，Meta-Harness 以 ${10} \times$ 倍少的完整评估次数与最优的现有文本优化器（OpenEvolve、TTT-Discover）持平，且其最终准确率超过它们 10 个百分点以上。


To isolate which parts of the proposer interface matter most, we compare three conditions in online text classification: a scores-only condition, a scores-plus-summary condition in which the proposer receives LLM-generated summaries but no raw traces, and the full Meta-Harness interface with access to execution traces (Table 3). The results show a large gap in favor of the full interface: scores-only reaches 34.6 median and 41.3 best accuracy, while scores-plus-summary reaches 34.9 median and 38.7 best. By contrast, Meta-Harness reaches 50.0 median and 56.7 best accuracy, and even its median candidate outperforms the best candidate found under either ablation. We interpret this as evidence that full access to execution traces is the most important component of the interface: summaries do not recover the missing signal, and may even hurt by compressing away diagnostically useful details.
为了分离提议者接口中哪些部分最为关键，我们在在线文本分类中比较了三种条件：仅评分条件、评分加摘要条件（提议者接收大模型生成的摘要而非原始追踪），以及具有执行追踪访问权限的完整 Meta-Harness 接口（表 3）。结果显示完整接口具有显著优势：仅评分条件的中位数准确率为 34.6，最佳为 41.3；评分加摘要条件的中位数为 34.9，最佳为 38.7。相比之下，Meta-Harness 的中位数准确率达到 50.0，最佳达到 56.7，甚至其候选方案的中位数表现也优于任一消融实验下的最佳候选方案。我们认为这证明了对执行追踪的完全访问是接口中最重要的组件：摘要无法恢复丢失的信号，甚至可能因压缩掉具有诊断价值的细节而产生负面影响。


<table><tr><td>Method</td><td>Scores</td><td>Code</td><td>Summ.</td><td>Traces</td><td>Median↑</td><td>Best Acc↑</td><td>$> \mathrm{{ZS}}$</td></tr><tr><td>Scores Only</td><td>✓</td><td>✓</td><td>✘</td><td>✘</td><td>34.6</td><td>41.3</td><td>26</td></tr><tr><td>Scores + Summary</td><td>✓</td><td>✓</td><td>✓</td><td>✘</td><td>34.9</td><td>38.7</td><td>23</td></tr><tr><td>Meta-Harness (full)</td><td>✓</td><td>✓</td><td>-</td><td>✓</td><td>50.0</td><td>56.7</td><td>39</td></tr></table>
<table><tbody><tr><td>方法</td><td>分数</td><td>代码</td><td>摘要</td><td>轨迹</td><td>中位数↑</td><td>最佳准确率↑</td><td>$> \mathrm{{ZS}}$</td></tr><tr><td>仅分数</td><td>✓</td><td>✓</td><td>✘</td><td>✘</td><td>34.6</td><td>41.3</td><td>26</td></tr><tr><td>分数 + 摘要</td><td>✓</td><td>✓</td><td>✓</td><td>✘</td><td>34.9</td><td>38.7</td><td>23</td></tr><tr><td>Meta-Harness（完整版）</td><td>✓</td><td>✓</td><td>-</td><td>✓</td><td>50.0</td><td>56.7</td><td>39</td></tr></tbody></table>


Table 3: Ablation of the information available to the proposer in online text classification. $>$ ZS: number of runs whose accuracy exceeded the zero-shot baseline. The full Meta-Harness interface substantially outperforms scores-only and scores-plus-summary ablations. Access to raw execution traces is the key ingredient for enabling harness search.
表3：在线文本分类中提议者可用信息的消融实验。$>$ ZS：准确率超过零样本基线的运行次数。完整的Meta-Harness接口表现显著优于仅含分数和含分数加摘要的消融版本。获取原始执行轨迹是实现工具链搜索的关键要素。


Comparison vs state-of-the-art harnesses. Our primary points of comparison are hand-designed harnesses for this problem setting: Agentic Context Engineering (ACE, Zhang et al. [59]), which uses reflective memory curation to build context over time, and Meta Context Engineering (MCE, Ye et al. [52]), which maintains and evolves a library of natural-language skills for context construction. As additional baselines, we evaluate zero-shot prompting and few-shot prompting with $N \in  \{ 4,8,{16},{32}$ ,all $\}$ examples. Results in Table 2 show that Meta-Harness improves substantially over prior hand-designed harnesses. The selected Meta-Harness reaches 48.6% accuracy, outperforming ACE by 7.7 points and MCE by 8.6 points. These gains do not come from using more context: Meta-Harness uses only 11.4K context tokens, versus 50.8K for ACE and 28.5K for MCE.
与现有最先进工具链的对比。我们的主要对比对象是针对该问题场景人工设计的工具链：Agentic Context Engineering (ACE, Zhang et al. [59])，它利用反射式记忆整理随时间构建上下文；以及 Meta Context Engineering (MCE, Ye et al. [52])，它维护并演进一个用于上下文构建的自然语言技能库。作为额外基线，我们评估了零样本提示和包含$N \in  \{ 4,8,{16},{32}$、所有$\}$示例的少样本提示。表2中的结果显示，Meta-Harness较先前的人工设计工具链有显著提升。选定的Meta-Harness达到了48.6%的准确率，分别比ACE和MCE高出7.7个和8.6个百分点。这些增益并非源于使用更多上下文：Meta-Harness仅使用11.4K个上下文Token，而ACE和MCE分别为50.8K和28.5K。


<table><tr><td>Method</td><td>Median</td><td>Best</td></tr><tr><td>GEPA [1]</td><td>32.6</td><td>40.2</td></tr><tr><td>Best-of-N</td><td>34.0</td><td>44.2</td></tr><tr><td>OpenEvolve 43</td><td>39.1</td><td>43.3</td></tr><tr><td>TTT-Discover |55|</td><td>34.1</td><td>45.6</td></tr><tr><td>Meta-Harness</td><td>50.0</td><td>56.7</td></tr></table>
<table><tbody><tr><td>方法</td><td>中位数</td><td>最佳</td></tr><tr><td>GEPA [1]</td><td>32.6</td><td>40.2</td></tr><tr><td>Best-of-N</td><td>34.0</td><td>44.2</td></tr><tr><td>OpenEvolve 43</td><td>39.1</td><td>43.3</td></tr><tr><td>TTT-Discover |55|</td><td>34.1</td><td>45.6</td></tr><tr><td>Meta-Harness</td><td>50.0</td><td>56.7</td></tr></tbody></table>


Table 4: Text classification accuracies of the harnesses proposed by different text optimizers (search set). Meta-Harness is substantially more effective at harness optimization.
表 4：不同文本优化器所提议的 harness 在文本分类任务上的准确率（搜索集）。Meta-Harness 在 harness 优化方面表现出显著更高的有效性。


Accuracy-Context Tradeoffs. Because Meta-Harness performs free-form optimization over harness code, we can express a joint preference for both accuracy and context cost rather than committing to a single scalar objective in advance. Given only the current metrics and the desired trade-off, the proposer is able to discover harnesses across a broad range of the frontier, yielding a smooth accuracy-context Pareto curve in Figure 3 This allows us to trade additional context for higher test accuracy in a controlled way, rather than committing to a single hand-designed operating point.
准确率与上下文的权衡。由于 Meta-Harness 对 harness 代码执行自由形式的优化，我们可以表达对准确率和上下文成本的联合偏好，而无需预先确定单一的标量目标。仅给定当前指标和期望的权衡，提议器就能在广泛的边界范围内发现 harness，从而在图 3 中生成平滑的准确率-上下文帕累托曲线。这使我们能够以可控的方式用额外的上下文换取更高的测试准确率，而不必局限于单一的人工设计操作点。


Out-of-distribution (OOD) task evaluation. We evaluate whether the discovered harness generalizes to entirely new datasets unseen during search. We consider nine diverse datasets, and describe them in detail in Appendix C.1. The selected Meta-Harness system achieves the best average accuracy (73.1%), outperforming ACE (70.2%) and all few-shot baselines (Table 5). Notably, we observe that naively adding more few-shot examples beyond 32 hurts performance in 7/9 tasks. Meta-Harness shows the highest performance on 6/9 datasets, suggesting that the discovered harness captures generally effective strategies for text classification rather than overfitting to the specific datasets used during search.
分布外（OOD）任务评估。我们评估了所发现的 harness 是否能泛化到搜索过程中未见的全新数据集。我们考虑了九个不同的数据集，并在附录 C.1 中进行了详细描述。选定的 Meta-Harness 系统实现了最佳平均准确率（73.1%），优于 ACE（70.2%）及所有少样本基线（表 5）。值得注意的是，我们观察到在 7/9 的任务中，盲目增加超过 32 个少样本示例反而会损害性能。Meta-Harness 在 6/9 的数据集上表现出最高性能，这表明所发现的 harness 捕捉到了文本分类中普遍有效的策略，而非仅仅过拟合于搜索期间使用的特定数据集。


### 4.2 Harnesses for Retrieval-Augmented Reasoning
### 4.2 用于检索增强推理的 Harness


We study a somewhat non-standard setup for olympiad math solving: augmenting the model with the ability to retrieve examples from a large corpus. There is a good reason to expect retrieval to help mathematical reasoning in principle, because solutions often share reusable proof patterns, so previous reasoning traces contain information that a model may
我们研究了一种用于奥数求解的非标准设置：增强模型从大型语料库中检索示例的能力。原则上，有充分的理由预期检索有助于数学推理，因为解题过程通常共享可复用的证明模式，因此先前的推理轨迹包含了模型可能需要的信息。


---



${}^{2}$ We slightly overload terminology for brevity: in the tables,Meta-Harness denotes the best discovered harness, whereas elsewhere it refers to the entire harness search procedure.
${}^{2}$ 为简洁起见，我们对术语进行了轻微的重载：在表格中，Meta-Harness 指代发现的最佳 harness，而在其他地方则指代整个 harness 搜索过程。


---



<table><tr><td>Harness</td><td>SciC</td><td>FiNER</td><td>Amz5</td><td>FPB</td><td>GoEmo</td><td>Bank77</td><td>News</td><td>SciT</td><td>TwHate</td><td>Avg Acc</td><td>Ctx↓</td></tr><tr><td>Zero-shot</td><td>32.7</td><td>56.0</td><td>52.7</td><td>90.0</td><td>42.0</td><td>80.7</td><td>84.7</td><td>89.3</td><td>75.3</td><td>67.0</td><td>-</td></tr><tr><td>Few-shot (8)</td><td>34.0</td><td>63.0</td><td>54.0</td><td>90.0</td><td>44.0</td><td>82.7</td><td>84.7</td><td>91.3</td><td>76.7</td><td>68.9</td><td>2.2</td></tr><tr><td>Few-shot (32)</td><td>38.7</td><td>62.0</td><td>53.3</td><td>90.7</td><td>43.3</td><td>86.0</td><td>85.3</td><td>90.7</td><td>76.7</td><td>69.6</td><td>5.2</td></tr><tr><td>Few-shot (all)</td><td>35.3</td><td>61.0</td><td>50.0</td><td>93.3</td><td>42.7</td><td>80.7</td><td>84.0</td><td>90.0</td><td>76.7</td><td>68.2</td><td>7.4</td></tr><tr><td>ACE 59</td><td>40.7</td><td>74.0</td><td>48.0</td><td>96.7</td><td>44.0</td><td>83.3</td><td>86.0</td><td>90.7</td><td>68.7</td><td>70.2</td><td>11.7</td></tr><tr><td>Meta-Harness</td><td>53.3</td><td>67.0</td><td>60.0</td><td>94.0</td><td>46.0</td><td>82.7</td><td>86.7</td><td>91.3</td><td>77.3</td><td>73.1</td><td>7.3</td></tr></table>
<table><tbody><tr><td>Harness</td><td>SciC</td><td>FiNER</td><td>Amz5</td><td>FPB</td><td>GoEmo</td><td>Bank77</td><td>News</td><td>SciT</td><td>TwHate</td><td>平均准确率</td><td>上下文↓</td></tr><tr><td>零样本</td><td>32.7</td><td>56.0</td><td>52.7</td><td>90.0</td><td>42.0</td><td>80.7</td><td>84.7</td><td>89.3</td><td>75.3</td><td>67.0</td><td>-</td></tr><tr><td>少样本 (8)</td><td>34.0</td><td>63.0</td><td>54.0</td><td>90.0</td><td>44.0</td><td>82.7</td><td>84.7</td><td>91.3</td><td>76.7</td><td>68.9</td><td>2.2</td></tr><tr><td>少样本 (32)</td><td>38.7</td><td>62.0</td><td>53.3</td><td>90.7</td><td>43.3</td><td>86.0</td><td>85.3</td><td>90.7</td><td>76.7</td><td>69.6</td><td>5.2</td></tr><tr><td>少样本 (全部)</td><td>35.3</td><td>61.0</td><td>50.0</td><td>93.3</td><td>42.7</td><td>80.7</td><td>84.0</td><td>90.0</td><td>76.7</td><td>68.2</td><td>7.4</td></tr><tr><td>ACE 59</td><td>40.7</td><td>74.0</td><td>48.0</td><td>96.7</td><td>44.0</td><td>83.3</td><td>86.0</td><td>90.7</td><td>68.7</td><td>70.2</td><td>11.7</td></tr><tr><td>Meta-Harness</td><td>53.3</td><td>67.0</td><td>60.0</td><td>94.0</td><td>46.0</td><td>82.7</td><td>86.7</td><td>91.3</td><td>77.3</td><td>73.1</td><td>7.3</td></tr></tbody></table>


Table 5: OOD text classification dataset evaluation. We report test accuracy for each dataset and the average additional context tokens across all nine datasets. Meta-Harness outperforms the next best method by 2.9 points on these 9 previously unseen tasks.
表5：OOD文本分类数据集评估。我们报告了每个数据集的测试准确率以及全部九个数据集的平均额外上下文Token数。在这些此前未见的9项任务中，Meta-Harness的表现比次优方法高出2.9个百分点。


<table><tr><td>Method</td><td>GPT-5.4n</td><td>GPT-5.4m</td><td>Gem-3.1FL</td><td>Gem-3F</td><td>GPT-20B</td><td>Avg.</td></tr><tr><td>No Retriever</td><td>23.0</td><td>28.8</td><td>28.6</td><td>42.6</td><td>47.6</td><td>34.1</td></tr><tr><td>Dense Retrieval $\left( {k = 1}\right)$</td><td>27.1 (+4.1)</td><td>24.5 (-4.3)</td><td>31.3 (+2.7)</td><td>42.3 (-0.3)</td><td>46.9 (-0.7)</td><td>34.4 (+0.3)</td></tr><tr><td>Dense Retrieval (k=5)</td><td>31.1 (+8.1)</td><td>28.3 (-0.5)</td><td>37.1 (+8.5)</td><td>47.2 (+4.6)</td><td>46.7 (-0.9)</td><td>38.1 (+4.0)</td></tr><tr><td>Random Few-shot</td><td>23.1 (+0.1)</td><td>24.5 (-4.3)</td><td>31.0 (+2.4)</td><td>40.4 (-2.2)</td><td>41.8 (-5.8)</td><td>32.2 (-1.9)</td></tr><tr><td>BM25 Retrieval</td><td>30.2 (+7.2)</td><td>29.2 (+0.4)</td><td>32.8 (+4.2)</td><td>46.6 (+4.0)</td><td>48.9 (+1.3)</td><td>37.5 (+3.4)</td></tr><tr><td>Meta-Harness</td><td>31.7 (+8.7)</td><td>30.4 (+1.6)</td><td>34.9 (+6.3)</td><td>46.3 (+3.7)</td><td>50.6 (+3.0)</td><td>38.8 (+4.7)</td></tr></table>
<table><tbody><tr><td>方法</td><td>GPT-5.4n</td><td>GPT-5.4m</td><td>Gem-3.1FL</td><td>Gem-3F</td><td>GPT-20B</td><td>平均值</td></tr><tr><td>无检索器</td><td>23.0</td><td>28.8</td><td>28.6</td><td>42.6</td><td>47.6</td><td>34.1</td></tr><tr><td>稠密检索 $\left( {k = 1}\right)$</td><td>27.1 (+4.1)</td><td>24.5 (-4.3)</td><td>31.3 (+2.7)</td><td>42.3 (-0.3)</td><td>46.9 (-0.7)</td><td>34.4 (+0.3)</td></tr><tr><td>稠密检索 (k=5)</td><td>31.1 (+8.1)</td><td>28.3 (-0.5)</td><td>37.1 (+8.5)</td><td>47.2 (+4.6)</td><td>46.7 (-0.9)</td><td>38.1 (+4.0)</td></tr><tr><td>随机少样本</td><td>23.1 (+0.1)</td><td>24.5 (-4.3)</td><td>31.0 (+2.4)</td><td>40.4 (-2.2)</td><td>41.8 (-5.8)</td><td>32.2 (-1.9)</td></tr><tr><td>BM25 检索</td><td>30.2 (+7.2)</td><td>29.2 (+0.4)</td><td>32.8 (+4.2)</td><td>46.6 (+4.0)</td><td>48.9 (+1.3)</td><td>37.5 (+3.4)</td></tr><tr><td>Meta-Harness</td><td>31.7 (+8.7)</td><td>30.4 (+1.6)</td><td>34.9 (+6.3)</td><td>46.3 (+3.7)</td><td>50.6 (+3.0)</td><td>38.8 (+4.7)</td></tr></tbody></table>


Table 6: Retrieval-augmented math problem solving on 200 IMO-level math problems. We show pass@1 averaged over three samples per problem, with absolute improvement over the baseline in parentheses. The discovered Meta-Harness retrieval strategy improves reasoning on these IMO-level problems across all five held-out models, with a 4.7-point average gain over no retriever.
表6：在200道IMO级数学题上的检索增强数学解题表现。我们展示了每题三次采样的平均pass@1准确率，括号内为相对于基线的绝对提升。Meta-Harness发现的检索策略在所有五个留出模型上均提升了这些IMO级问题的推理能力，平均较无检索基线提升4.7个百分点。


be able to exploit at inference time. Yet retrieval has not become a standard ingredient in this setting, and prior work suggests that it has been much less successful on reasoning-intensive math benchmarks than in more fact-grounded domains [42, 49, 6]. The difficulty is that naive retrieval rarely surfaces the right traces in the right form. This suggests that success depends less on adding retrieval per se than on discovering the right retrieval policy. Rather than hand-designing that policy, we give Meta-Harness a hard set of olympiad problems and allow the retrieval behavior itself to emerge from search.
能够在推理时加以利用。然而，检索尚未成为该场景下的标准配置，且先前研究表明，与事实导向的领域相比，检索在推理密集型数学基准测试中的成功率要低得多 [42, 49, 6]。其难点在于，朴素检索很难以正确的形式呈现出正确的轨迹。这表明，成功与否不仅取决于是否添加检索，更取决于能否发现正确的检索策略。我们没有采用人工设计策略，而是为Meta-Harness提供了一组高难度的奥数题，并允许检索行为本身通过搜索涌现出来。


The retrieval corpus contains $\geq  {500},{000}$ solved problems from eight open-source datasets. We carefully deduplicated and decontaminated it against both evaluation benchmarks and the search set, confirmed that held-out problems have no exact prefix matches under our string-based filter, and manually inspected top BM25 retrievals for held-out examples (appendix C.2). We use Meta-Harness to optimize a harness for 40 iterations over a 250-problem search set of Olympiad-difficulty math problems (OlympiadBench + Omni-MATH hard), producing 109 candidate retrieval harnesses. We initialize the search population $\mathcal{H}$ from the main baseline harnesses in this setting: zero-shot, few-shot, and ACE. We select a single harness based on search-set performance using GPT-OSS-20B (Appendix B.2). We evaluate this harness on 200 previously unseen IMO-level problems drawn from IMO-AnswerBench, IMO-ProofBench, and ArXivMath [30, 6]. In addition to GPT-0SS-20B, we evaluate the same retrieval harness on four models not seen during search: GPT-5.4-nano, GPT-5.4-mini, Gemini-3.1-Flash-Lite, and Gemini-3-Flash. We follow the standard evaluation protocol of prior work [30] and report accuracy averaged over three samples per problem.
检索语料库包含来自八个开源数据集的 $\geq  {500},{000}$ 道已解题目。我们仔细对其进行了去重和去污染处理，排除了评估基准和搜索集中的内容，确认留出题目在基于字符串的过滤下没有精确的前缀匹配，并人工检查了留出示例的顶级BM25检索结果（附录C.2）。我们使用Meta-Harness在包含250道奥数难度数学题（OlympiadBench + Omni-MATH hard）的搜索集上对一个harness进行了40次迭代优化，生成了109个候选检索harness。我们从该场景下的主要基线harness（零样本、少样本和ACE）初始化搜索种群 $\mathcal{H}$。我们根据GPT-OSS-20B在搜索集上的表现选择了一个单一的harness（附录B.2）。我们在来自IMO-AnswerBench、IMO-ProofBench和ArXivMath [30, 6] 的200道此前未见的IMO级问题上评估了该harness。除GPT-0SS-20B外，我们还在搜索过程中未见过的四个模型上评估了相同的检索harness：GPT-5.4-nano、GPT-5.4-mini、Gemini-3.1-Flash-Lite和Gemini-3-Flash。我们遵循先前研究 [30] 的标准评估协议，并报告每题三次采样的平均准确率。


Results. Table 6 compares the discovered harness against no retrieval, dense retrieval using the separate embedding model text-embedding-3-small, random few-shot prompting, and BM25 retrieval. In contrast, Meta-Harness operates entirely in code space on top of the same BM25-based lexical retrieval stack as the sparse baseline, rather than introducing an additional dense encoder. The discovered retrieval harness outperforms the no-retrieval baseline across all five held-out models, with an average gain of 4.7 points. It also matches or exceeds the strongest fixed baselines on average, outperforming BM25 retrieval by 1.3 points overall, while avoiding the regressions observed with dense retrieval and random few-shot prompting across several models.
结果。表6将发现的harness与无检索、使用独立嵌入模型text-embedding-3-small的稠密检索、随机少样本提示以及BM25检索进行了比较。与引入额外稠密编码器的方法不同，Meta-Harness完全在代码空间内运行，并基于与稀疏基线相同的BM25词法检索栈。所发现的检索harness在所有五个留出模型上均优于无检索基线，平均提升4.7个百分点。它在平均表现上也达到或超过了最强的固定基线，总体上比BM25检索高出1.3个百分点，同时避免了在多个模型中观察到的稠密检索和随机少样本提示所带来的性能倒退。


## Meta-Harness Improves Reasoning on IMO-Level Math Problems
## Meta-Harness提升了IMO级数学问题的推理能力


In retrieval-augmented math reasoning, a single discovered retrieval harness transfers across five held-out models, improving accuracy by 4.7 points on average over no retrieval and yielding the strongest overall average among the compared methods.
在检索增强数学推理中，单个发现的检索harness可在五个留出模型间迁移，平均准确率较无检索提升4.7个百分点，在所有对比方法中取得了最优的总体平均表现。


### 4.3 Evaluating Agentic Coding Harnesses on TerminalBench-2
### 4.3 在TerminalBench-2上评估智能体编码harness


TerminalBench-2 [33] evaluates LLM agents on 89 challenging tasks that require long-horizon, fully autonomous execution under complex dependencies, and substantial domain knowledge. Prior work has shown that the choice agent harness has a large effect on performance on this benchmark. We initialize search from two strong open baselines, Terminus 2 [33] and Terminus-KIRA [25]. For this experiment, we perform search and final evaluation on the same 89-task benchmark. We use this benchmark as a discovery problem [54] in which the goal is to discover a harness configuration that improves performance on a hard, publicly contested benchmark. This is standard practice: public writeups already describe repeated benchmark-specific harness iteration on TerminalBench itself [18, 34, 25], and the benchmark is small and expensive enough that introducing a separate split would materially weaken the search signal. We additionally check for overfitting by manual inspection and regex-based audits for task-specific string leakage into evolved harnesses. We note that although the resulting harness is specialized to the TerminalBench-2 regime, autonomous completion of difficult long-horizon tasks from a single instruction is a core capability, and the benchmark consists of many tasks that frontier models and heavily engineered harnesses struggle with.
TerminalBench-2 [33] 在89项具有挑战性的任务上评估LLM智能体，这些任务要求在复杂依赖关系和大量领域知识下进行长程、完全自主的执行。先前研究表明，选择智能体harness对该基准测试的性能有巨大影响。我们从两个强大的开源基线Terminus 2 [33] 和Terminus-KIRA [25] 初始化搜索。在本实验中，我们在相同的89项任务基准上进行搜索和最终评估。我们将此基准用作发现问题 [54]，目标是发现一种能提升在这一高难度、公开竞争基准上表现的harness配置。这是标准做法：公开的报告中已经描述了在TerminalBench本身上进行的重复的、针对基准的harness迭代 [18, 34, 25]，且该基准规模较小且成本高昂，引入单独的划分会实质性削弱搜索信号。我们还通过人工检查和基于正则表达式的审计来检查是否存在针对特定任务的字符串泄露，以防过拟合。我们注意到，尽管最终的harness是针对TerminalBench-2机制专门优化的，但从单一指令自主完成困难的长程任务是一项核心能力，且该基准包含许多前沿模型和经过深度工程化设计的harness都难以应对的任务。


Results. We report results on the full benchmark in Table 7, evaluated on two base models: Claude Opus 4.6 and Claude Haiku 4.5. On Opus 4.6, Meta-Harness discovers a harness achieving 76.4% pass rate, surpassing the hand-engineered Terminus-KIRA (74.7%) and ranking #2 among all Opus 4.6 agents on the TerminalBench-2 leader-board. The only higher-scoring Opus 4.6 agent is ForgeCode (81.8%); however, we were unable to reproduce their reported result from the publicly available code alone, suggesting their leaderboard scores depend on components beyond the published repository. On the weaker Haiku 4.5 model, the improvement is larger: Meta-Harness achieves 37.6%, outperforming the next-best reported agent (Goose, 35.5%) by 2.1 points. TerminalBench-2 is an actively contested benchmark with multiple teams directly optimizing for it, so the fact that an automatic search method can achieve benefits at this frontier is encouraging for long-horizon text-optimization loops.
结果。我们在表7中报告了完整基准测试的结果，评估对象为两个基础模型：Claude Opus 4.6和Claude Haiku 4.5。在Opus 4.6上，Meta-Harness发现的工具链达到了76.4%的通过率，超过了人工设计的Terminus-KIRA（74.7%），并在TerminalBench-2排行榜的所有Opus 4.6智能体中排名第二。唯一得分更高的Opus 4.6智能体是ForgeCode（81.8%）；然而，我们仅凭公开代码无法复现其报告的结果，这表明其排行榜分数依赖于已发布代码库之外的组件。在较弱的Haiku 4.5模型上，提升幅度更大：Meta-Harness达到了37.6%，比排名第二的智能体（Goose，35.5%）高出2.1个百分点。TerminalBench-2是一个竞争激烈的基准测试，有多个团队对其进行直接优化，因此自动搜索方法能在这一前沿领域取得优势，对于长程文本优化循环而言令人鼓舞。


<table><tr><td>Harness</td><td>Auto</td><td>Pass (%)</td></tr><tr><td colspan="3">Claude Opus 4.6</td></tr><tr><td>Claude Code</td><td>✘</td><td>58.0</td></tr><tr><td>Terminus 2</td><td>✘</td><td>62.9</td></tr><tr><td>Mux</td><td>✘</td><td>66.5</td></tr><tr><td>Droid</td><td>✘</td><td>69.9</td></tr><tr><td>TongAgents</td><td>✘</td><td>71.9</td></tr><tr><td>MAYA-V2</td><td>✘</td><td>72.1</td></tr><tr><td>Terminus-KIRA</td><td>✘</td><td>74.7</td></tr><tr><td>Capy</td><td>✘</td><td>75.3</td></tr><tr><td>ForgeCode</td><td>✘</td><td>81.8</td></tr><tr><td>Meta-Harness</td><td>✓</td><td>76.4</td></tr><tr><td colspan="3">Claude Haiku 4.5</td></tr><tr><td>OpenHands</td><td>✘</td><td>13.9</td></tr><tr><td>Claude Code</td><td>✘</td><td>27.5</td></tr><tr><td>Terminus 2</td><td>✘</td><td>28.3</td></tr><tr><td>Mini-SWE-Agent</td><td>✘</td><td>29.8</td></tr><tr><td>Terminus-KIRA</td><td>✘</td><td>33.7</td></tr><tr><td>Goose</td><td>✘</td><td>35.5</td></tr><tr><td>Meta-Harness</td><td>✓</td><td>37.6</td></tr></table>
<table><tbody><tr><td>Harness</td><td>自动</td><td>通过率 (%)</td></tr><tr><td colspan="3">Claude Opus 4.6</td></tr><tr><td>Claude Code</td><td>✘</td><td>58.0</td></tr><tr><td>Terminus 2</td><td>✘</td><td>62.9</td></tr><tr><td>Mux</td><td>✘</td><td>66.5</td></tr><tr><td>Droid</td><td>✘</td><td>69.9</td></tr><tr><td>TongAgents</td><td>✘</td><td>71.9</td></tr><tr><td>MAYA-V2</td><td>✘</td><td>72.1</td></tr><tr><td>Terminus-KIRA</td><td>✘</td><td>74.7</td></tr><tr><td>Capy</td><td>✘</td><td>75.3</td></tr><tr><td>ForgeCode</td><td>✘</td><td>81.8</td></tr><tr><td>Meta-Harness</td><td>✓</td><td>76.4</td></tr><tr><td colspan="3">Claude Haiku 4.5</td></tr><tr><td>OpenHands</td><td>✘</td><td>13.9</td></tr><tr><td>Claude Code</td><td>✘</td><td>27.5</td></tr><tr><td>Terminus 2</td><td>✘</td><td>28.3</td></tr><tr><td>Mini-SWE-Agent</td><td>✘</td><td>29.8</td></tr><tr><td>Terminus-KIRA</td><td>✘</td><td>33.7</td></tr><tr><td>Goose</td><td>✘</td><td>35.5</td></tr><tr><td>Meta-Harness</td><td>✓</td><td>37.6</td></tr></tbody></table>


Table 7: Pass rate on TerminalBench- 2. Results or others are from the official leaderboard. Meta-Harness ranks #2 among all Opus-4.6 agents and #1 among all Haiku-4.5 agents on this competitive task.
表 7：TerminalBench-2 上的通过率。其他结果来自官方排行榜。Meta-Harness 在此项竞争性任务中，在所有 Opus-4.6 智能体中排名第 2，在所有 Haiku-4.5 智能体中排名第 1。


Qualitative behavior of the proposer. The harness search trajectory helps explain why Meta-Harness achieves these gains; we provide a detailed summary in Appendix A. In early iterations, the proposer combined plausible structural fixes with prompt-template edits and observed that both candidates regressed. It then explicitly hypothesized that the regressions were confounded by the shared prompt intervention, isolated the structural changes from the prompt rewrite, and ultimately pivoted toward a safer additive modification that became the best candidate in the run. This provides qualitative evidence that filesystem access enables the proposer to inspect prior experience in enough detail to form causal hypotheses and revise the harness accordingly.
提案者的定性行为。Harness 搜索轨迹有助于解释 Meta-Harness 为何能取得这些提升；我们在附录 A 中提供了详细总结。在早期迭代中，提案者将合理的结构性修复与提示词模板编辑相结合，并观察到两个候选方案均出现倒退。随后，它明确假设倒退是由共享的提示词干预所混淆，将结构性更改与提示词重写分离开来，最终转向了一种更安全的加性修改，该修改成为了本次运行中的最佳候选方案。这提供了定性证据，表明文件系统访问权限使提案者能够足够详细地检查过往经验，从而形成因果假设并相应地修订 Harness。


## Meta-Harness Surpasses Hand-Engineered Agents on TerminalBench-2
## Meta-Harness 在 TerminalBench-2 上超越了人工设计的智能体


On TerminalBench-2, Meta-Harness automatically discovers harnesses that surpass Terminus-KIRA on Opus 4.6 and rank #1 among all Haiku 4.5 agents.
在 TerminalBench-2 上，Meta-Harness 自动发现的 Harness 超越了 Opus 4.6 上的 Terminus-KIRA，并在所有 Haiku 4.5 智能体中排名第 1。


## 5 Discussion
## 5 讨论


Beyond outperforming existing harnesses, Meta-Harness has several practical advantages. Discovered harnesses generalize to out-of-distribution classification datasets (Table 5) and to unseen base models in the math setting (Table 6). A search run completes in a few hours of wall-clock time, yet produces readable, transferable strategies that can be reused across models, including future, stronger ones. Overfitting in code space is also more inspectable: brittle if-chains or hard-coded class mappings are visible on inspection in a way that weight-space overfitting is not. More broadly, our results suggest that the main advantage of Meta-Harness is not just search over code, but search with selective access to prior diagnostic experience. The proposer is not limited to scalar rewards or fixed summaries; it can inspect raw code, execution traces, and prior failures, then use that information to form and test hypotheses about what to change. The qualitative search trajectories in Appendix A.2 illustrate this behavior directly.
除了优于现有的 Harness 外，Meta-Harness 还具有多项实际优势。发现的 Harness 可泛化至分布外分类数据集（表 5）以及数学设置中未见过的基础模型（表 6）。搜索运行仅需数小时的挂钟时间，却能产生可读、可迁移的策略，并可在不同模型（包括未来更强大的模型）之间重复使用。代码空间中的过拟合也更易于检查：脆弱的 if-else 链或硬编码的类映射在检查时清晰可见，这与权重空间的过拟合不同。更广泛地说，我们的结果表明，Meta-Harness 的主要优势不仅在于代码搜索，还在于对过往诊断经验的选择性访问。提案者不仅限于标量奖励或固定摘要；它能够检查原始代码、执行轨迹和过往失败记录，然后利用这些信息形成并测试关于如何修改的假设。附录 A.2 中的定性搜索轨迹直接展示了这种行为。


Our findings reflect a recurring pattern in machine learning [45]: once a search space becomes accessible, stronger general-purpose agents can outperform hand-engineered solutions. A natural next step for future work is to co-evolve the harness and the model weights, letting the strategy shape what the model learns and vice versa. While we evaluate on three diverse domains, our experiments demonstrate that harness search can work with one particularly strong coding-agent proposer (Claude Code); a broader study of how the effect varies across proposer agents remains for future work.
我们的发现反映了机器学习中一个反复出现的模式 [45]：一旦搜索空间变得可访问，更强大的通用智能体就能超越人工设计的解决方案。未来工作的一个自然步骤是协同进化 Harness 和模型权重，让策略塑造模型的学习内容，反之亦然。虽然我们在三个不同的领域进行了评估，但我们的实验证明，Harness 搜索可以与一个特别强大的编码智能体提案者（Claude Code）配合使用；关于该效果在不同提案者智能体之间如何变化的更广泛研究，留待未来工作探讨。


## Acknowledgements
## 致谢


We thank KRAFTON AI for providing API credit support. This work is supported by OpenAI, KFAS, and Schmidt Sciences AI2050. We thank Anikait Singh and Jubayer Ibn Hamid for their valuable feedback and suggestions, and Sienna J. Lee for patiently listening to YL's half-formed thoughts during the early stages of this work.
感谢 KRAFTON AI 提供的 API 额度支持。本工作得到 OpenAI、KFAS 和 Schmidt Sciences AI2050 的支持。感谢 Anikait Singh 和 Jubayer Ibn Hamid 提供的宝贵反馈和建议，以及 Sienna J. Lee 在本工作早期阶段耐心倾听 YL 不成熟的想法。


## References
## 参考文献


[1] Lakshya A Agrawal, Shangyin Tan, Dilara Soylu, Noah Ziems, Rishi Khare, Krista Opsahl-Ong, Arnav Singhvi, Herumb Shandilya, Michael J Ryan, Meng Jiang, et al. Gepa: Reflective prompt evolution can outperform reinforcement learning. arXiv preprint arXiv:2507.19457, 2025.
[1] Lakshya A Agrawal, Shangyin Tan, Dilara Soylu, Noah Ziems, Rishi Khare, Krista Opsahl-Ong, Arnav Singhvi, Herumb Shandilya, Michael J Ryan, Meng Jiang, 等。Gepa: 反思性提示词演化可以优于强化学习。arXiv 预印本 arXiv:2507.19457, 2025。


[2] Ekin Akyürek, Dale Schuurmans, Jacob Andreas, Tengyu Ma, and Denny Zhou. What learning algorithm is in-context learning? investigations with linear models, 2023. URL https://arxiv.org/abs/2211.15661
[2] Ekin Akyürek, Dale Schuurmans, Jacob Andreas, Tengyu Ma, 和 Denny Zhou。上下文学习是什么学习算法？基于线性模型的调查, 2023。URL https://arxiv.org/abs/2211.15661


[3] Marcin Andrychowicz, Misha Denil, Sergio Gomez, Matthew W Hoffman, David Pfau, Tom Schaul, Brendan Shillingford, and Nando De Freitas. Learning to learn by gradient descent by gradient descent. Advances in neural information processing systems, 29, 2016.
[3] Marcin Andrychowicz, Misha Denil, Sergio Gomez, Matthew W Hoffman, David Pfau, Tom Schaul, Brendan Shillingford, 和 Nando De Freitas。通过梯度下降学习梯度下降。神经信息处理系统进展, 29, 2016。


[4] Anthropic. Claude code: An agentic coding tool. https://www.anthropic.com/claude -code, 2025.
[4] Anthropic。Claude code: 一种智能体编码工具。https://www.anthropic.com/claude-code, 2025。


[5] Anthropic and community contributors. agentskills/agentskills. GitHub repository https://github.com/agentskills/agentskills.Specification and documentation for Agent Skills, accessed March 27, 2026.
[5] Anthropic 和社区贡献者。agentskills/agentskills。GitHub 仓库 https://github.com/agentskills/agentskills。Agent Skills 的规范与文档，访问于 2026 年 3 月 27 日。


[6] Mislav Balunović, Jasper Dekoninck, Ivo Petrov, Nikola Jovanović, and Martin Vechev. Matharena: Evaluating llms on uncontaminated math competitions, February 2025. URLhttps://matharena.ai/
[6] Mislav Balunović, Jasper Dekoninck, Ivo Petrov, Nikola Jovanović, 和 Martin Vechev。Matharena: 在未污染的数学竞赛中评估 LLM, 2025 年 2 月。URL https://matharena.ai/


[7] Francesco Barbieri, Jose Camacho-Collados, Leonardo Neves, and Luis Espinosa-Anke. Tweeteval: Unified benchmark and comparative evaluation for tweet classification, 2020. URL https://arxiv.org/abs/2010.12421
[7] Francesco Barbieri, Jose Camacho-Collados, Leonardo Neves, 和 Luis Espinosa-Anke. Tweeteval: 推文分类的统一基准与比较评估, 2020. URL https://arxiv.org/abs/2010.12421


[8] Luca Beurer-Kellner, Marc Fischer, and Martin Vechev. Prompting is programming: A query language for large language models. Proceedings of the ACM on Programming Languages, 7(PLDI):1946-1969, June 2023. ISSN 2475-1421. doi: 10.1145/3591300. URL http://dx.doi.org/10.1145/3591300
[8] Luca Beurer-Kellner, Marc Fischer, 和 Martin Vechev. 提示即编程：大语言模型的查询语言. ACM 程序语言会议论文集, 7(PLDI):1946-1969, 2023年6月. ISSN 2475-1421. doi: 10.1145/3591300. URL http://dx.doi.org/10.1145/3591300


[9] Birgitta Böckeler. Harness engineering. https://martinfowler.com/articles/explor ing-gen-ai/harness-engineering.html, March 2026. martinfowler.com.
[9] Birgitta Böckeler. 适配器工程 (Harness engineering). https://martinfowler.com/articles/exploring-gen-ai/harness-engineering.html, 2026年3月. martinfowler.com.


[10] Can Böliük. I improved 15 LLMs at coding in one afternoon. only the harness changed. https://blog.can.ac/2026/02/12/the-harness-problem/.February 2026.
[10] Can Böliük. 我在一个下午改进了15个LLM的编码能力，仅仅改变了适配器. https://blog.can.ac/2026/02/12/the-harness-problem/. 2026年2月.


[11] Iñigo Casanueva, Tadas Temčinas, Daniela Gerz, Matthew Henderson, and Ivan Vulić. Efficient intent detection with dual sentence encoders, 2020. URL https://arxiv.org/ abs/2003.04807
[11] Iñigo Casanueva, Tadas Temčinas, Daniela Gerz, Matthew Henderson, 和 Ivan Vulić. 基于双句编码器的高效意图检测, 2020. URL https://arxiv.org/abs/2003.04807


[12] Mert Cemri, Shubham Agrawal, Akshat Gupta, Shu Liu, Audrey Cheng, Qiuyang Mang, Ashwin Naren, Lutfi Eren Erdogan, Koushik Sen, Matei Zaharia, et al. Adae-volve: Adaptive llm driven zeroth-order optimization. arXiv preprint arXiv:2602.20133, 2026.
[12] Mert Cemri, Shubham Agrawal, Akshat Gupta, Shu Liu, Audrey Cheng, Qiuyang Mang, Ashwin Naren, Lutfi Eren Erdogan, Koushik Sen, Matei Zaharia, 等. Adae-volve: 自适应LLM驱动的零阶优化. arXiv 预印本 arXiv:2602.20133, 2026.


[13] Harrison Chase. Langchain, October 2022. URL https://github.com/langchain-ai/ langchain, Software, released 2022-10-17.
[13] Harrison Chase. Langchain, 2022年10月. URL https://github.com/langchain-ai/langchain, 软件, 发布于 2022-10-17.


[14] Arman Cohan, Waleed Ammar, Madeleine van Zuylen, and Field Cady. Structural scaffolds for citation intent classification in scientific publications, 2019. URL https: //arxiv.org/abs/1904.01608
[14] Arman Cohan, Waleed Ammar, Madeleine van Zuylen, 和 Field Cady. 科学出版物中引文意图分类的结构化支架, 2019. URL https://arxiv.org/abs/1904.01608


[15] Dorottya Demszky, Dana Movshovitz-Attias, Jeongwoo Ko, Alan Cowen, Gaurav Nemade, and Sujith Ravi. Goemotions: A dataset of fine-grained emotions, 2020. URL https://arxiv.org/abs/2005.00547
[15] Dorottya Demszky, Dana Movshovitz-Attias, Jeongwoo Ko, Alan Cowen, Gaurav Nemade, 和 Sujith Ravi. Goemotions: 一个细粒度情感数据集, 2020. URL https://arxiv.org/abs/2005.00547


[16] Zhiwei Fei, Xiaoyu Shen, Dawei Zhu, Fengzhe Zhou, Zhuo Han, Alan Huang, Songyang Zhang, Kai Chen, Zhixin Yin, Zongwen Shen, et al. Lawbench: Benchmarking legal knowledge of large language models. In Proceedings of the 2024 conference on empirical methods in natural language processing, pp. 7933-7962, 2024.
[16] Zhiwei Fei, Xiaoyu Shen, Dawei Zhu, Fengzhe Zhou, Zhuo Han, Alan Huang, Songyang Zhang, Kai Chen, Zhixin Yin, Zongwen Shen, 等. Lawbench: 大语言模型法律知识基准测试. 载于 2024年自然语言处理实证方法会议论文集, 第7933-7962页, 2024.


[17] Chelsea Finn, Pieter Abbeel, and Sergey Levine. Model-agnostic meta-learning for fast adaptation of deep networks. In International Conference on Machine Learning, 2017.
[17] Chelsea Finn, Pieter Abbeel, 和 Sergey Levine. 用于深度网络快速适应的模型无关元学习. 载于 国际机器学习会议, 2017.


[18] ForgeCode. Benchmarks don't matter, 2025. URL https://forgecode.dev/blog/bench marks-dont-matter/
[18] ForgeCode. 基准测试并不重要, 2025. URL https://forgecode.dev/blog/benchmarks-dont-matter/


[19] Gretel AI. Symptom to diagnosis dataset. https://huggingface.co/datasets/gretel ai/symptom_to_diagnosis| 2023. Accessed: 2026-01-22.
[19] Gretel AI. 症状到诊断数据集. https://huggingface.co/datasets/gretelai/symptom_to_diagnosis, 2023. 访问日期: 2026-01-22.


[20] Shengran Hu, Cong Lu, and Jeff Clune. Automated design of agentic systems. In The Thirteenth International Conference on Learning Representations, 2025. URL https: //openreview.net/forum?id=t9U3LW7JVX
[20] Shengran Hu, Cong Lu, 和 Jeff Clune. 智能体系统的自动化设计. 载于 第十三届国际学习表征会议, 2025. URL https://openreview.net/forum?id=t9U3LW7JVX


[21] Anthropic Justin Young. Effective harnesses for long-running agents. https://anthro pic.com/engineering/effective-harnesses-for-long-running-agents. November 2025. Anthropic Engineering Blog.
[21] Anthropic Justin Young. 针对长运行智能体的高效适配器. https://anthropic.com/engineering/effective-harnesses-for-long-running-agents. 2025年11月. Anthropic 工程博客.


[22] Phillip Keung, Yichao Lu, György Szarvas, and Noah A. Smith. The multilingual amazon reviews corpus, 2020. URL https://arxiv.org/abs/2010.02573
[22] Phillip Keung, Yichao Lu, György Szarvas, 和 Noah A. Smith. 多语言亚马逊评论语料库, 2020. URL https://arxiv.org/abs/2010.02573


[23] Omar Khattab, Arnav Singhvi, Paridhi Maheshwari, Zhiyuan Zhang, Keshav San-thanam, Sri Vardhamanan, Saiful Haq, Ashutosh Sharma, Thomas T. Joshi, Hanna Moazam, Heather Miller, Matei Zaharia, and Christopher Potts. Dspy: Compiling declarative language model calls into self-improving pipelines, 2023. URL https://arxiv.org/abs/2310.03714.
[23] Omar Khattab, Arnav Singhvi, Paridhi Maheshwari, Zhiyuan Zhang, Keshav San-thanam, Sri Vardhamanan, Saiful Haq, Ashutosh Sharma, Thomas T. Joshi, Hanna Moazam, Heather Miller, Matei Zaharia, and Christopher Potts. Dspy: 将声明式语言模型调用编译为自改进流水线，2023。URL https://arxiv.org/abs/2310.03714。


[24] Tushar Khot, Ashish Sabharwal, and Peter Clark. Scitail: A textual entailment dataset from science question answering. Proceedings of the AAAI Conference on Artificial Intelligence, 32(1), Apr. 2018. doi: 10.1609/aaai.v32i1.12022. URL https://ojs.aaai.org/index.php/AAAI/article/view/12022
[24] Tushar Khot, Ashish Sabharwal, and Peter Clark. Scitail：一个来自科学问答的文本蕴含数据集。《AAAI人工智能会议论文集》，32(1)，2018年4月。doi: 10.1609/aaai.v32i1.12022。URL https://ojs.aaai.org/index.php/AAAI/article/view/12022


[25] KRAFTON AI and Ludo Robotics. Terminus-kira: Boosting frontier model performance on terminal-bench with minimal harness, 2026. URL https://github.com/krafton-a i/kira
[25] KRAFTON AI and Ludo Robotics. Terminus-kira：通过极简工具链提升前沿模型在Terminal-Bench上的表现，2026。URL https://github.com/krafton-a i/kira


[26] Yoonho Lee, Joseph Boen, and Chelsea Finn. Feedback descent: Open-ended text optimization via pairwise comparison. In arXiv preprint arXiv:2511.07919, 2025.
[26] Yoonho Lee, Joseph Boen, and Chelsea Finn. 反馈下降：通过成对比较进行开放式文本优化。《arXiv预印本 arXiv:2511.07919》，2025。


[27] Joel Lehman, Jonathan Gordon, Shawn Jain, Kamal Ndousse, Cathy Yeh, and Kenneth O. Stanley. Evolution through large models, 2022. URL https://arxiv.org/abs/ 2206.08896
[27] Joel Lehman, Jonathan Gordon, Shawn Jain, Kamal Ndousse, Cathy Yeh, and Kenneth O. Stanley. 通过大模型进行演化，2022。URL https://arxiv.org/abs/ 2206.08896


[28] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, et al. Retrieval-augmented generation for knowledge-intensive nlp tasks. Advances in neural information processing systems, 33:9459-9474, 2020.
[28] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, et al. 知识密集型NLP任务的检索增强生成。《神经信息处理系统进展》，33:9459-9474，2020。


[29] Lefteris Loukas, Manos Fergadiotis, Ilias Chalkidis, Eirini Spyropoulou, Prodromos Malakasiotis, Ion Androutsopoulos, and Georgios Paliouras. Finer: Financial numeric entity recognition for xbrl tagging. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 4419-4431. Association for Computational Linguistics, 2022. doi: 10.18653/v1/2022.acl-long.303. URL http://dx.doi.org/10.18653/v1/2022.acl-long.303
[29] Lefteris Loukas, Manos Fergadiotis, Ilias Chalkidis, Eirini Spyropoulou, Prodromos Malakasiotis, Ion Androutsopoulos, and Georgios Paliouras. Finer：用于XBRL标记的金融数字实体识别。《第60届计算语言学协会年会论文集（第1卷：长论文）》，第4419-4431页。计算语言学协会，2022。doi: 10.18653/v1/2022.acl-long.303。URL http://dx.doi.org/10.18653/v1/2022.acl-long.303


[30] Thang Luong, Dawsen Hwang, Hoang H. Nguyen, Golnaz Ghiasi, Yuri Chervonyi, In-suk Seo, Junsu Kim, Garrett Bingham, Jonathan Lee, Swaroop Mishra, Alex Zhai, Clara Huiyi Hu, Henryk Michalewski, Jimin Kim, Jeonghyun Ahn, Junhwi Bae, Xingyou Song, Trieu H. Trinh, Quoc V. Le, and Junehyuku Jung. Towards robust mathematical reasoning. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, 2025. URL https://aclanthology.org/2025.emnlp-main.1794/
[30] Thang Luong, Dawsen Hwang, Hoang H. Nguyen, Golnaz Ghiasi, Yuri Chervonyi, In-suk Seo, Junsu Kim, Garrett Bingham, Jonathan Lee, Swaroop Mishra, Alex Zhai, Clara Huiyi Hu, Henryk Michalewski, Jimin Kim, Jeonghyun Ahn, Junhwi Bae, Xingyou Song, Trieu H. Trinh, Quoc V. Le, and Junehyuku Jung. 迈向稳健的数学推理。《2025年自然语言处理经验方法会议论文集》，2025。URL https://aclanthology.org/2025.emnlp-main.1794/


[31] Aman Madaan, Niket Tandon, Prakhar Gupta, Skyler Hallinan, Luyu Gao, Sarah Wiegreffe, Uri Alon, Nouha Dziri, Shrimai Prabhumoye, Yiming Yang, et al. Self-refine: Iterative refinement with self-feedback. Advances in neural information processing systems, 36:46534-46594, 2023.
[31] Aman Madaan, Niket Tandon, Prakhar Gupta, Skyler Hallinan, Luyu Gao, Sarah Wiegreffe, Uri Alon, Nouha Dziri, Shrimai Prabhumoye, Yiming Yang, et al. Self-refine：基于自我反馈的迭代优化。《神经信息处理系统进展》，36:46534-46594，2023。


[32] Pekka Malo, Ankur Sinha, Pyry Takala, Pekka Korhonen, and Jyrki Wallenius. Good debt or bad debt: Detecting semantic orientations in economic texts, 2013. URL https://arxiv.org/abs/1307.5336
[32] Pekka Malo, Ankur Sinha, Pyry Takala, Pekka Korhonen, and Jyrki Wallenius. 好债还是坏债：检测经济文本中的语义倾向，2013。URL https://arxiv.org/abs/1307.5336


[33] Mike A Merrill, Alexander G Shaw, Nicholas Carlini, Boxuan Li, Harsh Raj, Ivan Bercovich, Lin Shi, Jeong Yeon Shin, Thomas Walshe, E Kelly Buchanan, et al. Terminal-bench: Benchmarking agents on hard, realistic tasks in command line interfaces. arXiv preprint arXiv:2601.11868, 2026.
[33] Mike A Merrill, Alexander G Shaw, Nicholas Carlini, Boxuan Li, Harsh Raj, Ivan Bercovich, Lin Shi, Jeong Yeon Shin, Thomas Walshe, E Kelly Buchanan, et al. Terminal-bench：在命令行界面中对智能体进行高难度、真实任务的基准测试。《arXiv预印本 arXiv:2601.11868》，2026。


[34] Jack Nichols. How we scored #1 on terminal-bench (52%), Jun 2025. URL https: //www.warp.dev/blog/terminal-bench
[34] Jack Nichols. 我们是如何在Terminal-bench上获得第一名（52%）的，2025年6月。URL https: //www.warp.dev/blog/terminal-bench


[35] Alexander Novikov, Ngân Vü, Marvin Eisenberger, Emilien Dupont, Po-Sen Huang, Adam Zsolt Wagner, Sergey Shirobokov, Borislav Kozlovskii, Francisco JR Ruiz, Abbas Mehrabian, et al. Alphaevolve: A coding agent for scientific and algorithmic discovery. arXiv preprint arXiv:2506.13131, 2025.
[35] Alexander Novikov, Ngân Vü, Marvin Eisenberger, Emilien Dupont, Po-Sen Huang, Adam Zsolt Wagner, Sergey Shirobokov, Borislav Kozlovskii, Francisco JR Ruiz, Abbas Mehrabian 等。Alphaevolve：用于科学与算法发现的编码智能体。arXiv 预印本 arXiv:2506.13131, 2025。


[36] OpenAI. Harness engineering: leveraging Codex in an agent-first world. https: //openai.com/index/harness-engineering/. February 2026. OpenAI Blog.
[36] OpenAI。Harness 工程：在智能体优先的世界中利用 Codex。https://openai.com/index/harness-engineering/。2026 年 2 月。OpenAI 博客。


[37] Charles Packer, Vivian Fang, Shishir-G Patil, Kevin Lin, Sarah Wooders, and Joseph_E Gonzalez. Memgpt: Towards llms as operating systems. 2023.
[37] Charles Packer, Vivian Fang, Shishir-G Patil, Kevin Lin, Sarah Wooders, 和 Joseph_E Gonzalez。Memgpt：迈向作为操作系统的大语言模型。2023。


[38] Reid Pryzant, Dan Iter, Jerry Li, Yin Tat Lee, Chenguang Zhu, and Michael Zeng. Automatic prompt optimization with "gradient descent" and beam search. arXiv preprint arXiv:2305.03495, 2023.
[38] Reid Pryzant, Dan Iter, Jerry Li, Yin Tat Lee, Chenguang Zhu, 和 Michael Zeng。利用“梯度下降”和束搜索进行自动提示词优化。arXiv 预印本 arXiv:2305.03495, 2023。


[39] Bernardino Romera-Paredes, Mohammadamin Barekatain, Alexander Novikov, Matej Balog, M Pawan Kumar, Emilien Dupont, Francisco JR Ruiz, Jordan S Ellenberg, Pengming Wang, Omar Fawzi, et al. Mathematical discoveries from program search with large language models. Nature, 625(7995):468-475, 2024.
[39] Bernardino Romera-Paredes, Mohammadamin Barekatain, Alexander Novikov, Matej Balog, M Pawan Kumar, Emilien Dupont, Francisco JR Ruiz, Jordan S Ellenberg, Pengming Wang, Omar Fawzi 等。通过大语言模型进行程序搜索的数学发现。《自然》，625(7995):468-475, 2024。


[40] Jurgen Schmidhuber. A neural network that embeds its own meta-levels. In IEEE International Conference on Neural Networks, 1993.
[40] Jurgen Schmidhuber。一种嵌入自身元层级的神经网络。载于 IEEE 国际神经网络会议，1993。


[41] Nadine Schneider, Nikolaus Stiefl, and Gregory A Landrum. What's what: The (nearly) definitive guide to reaction role assignment. Journal of chemical information and modeling, 56(12):2336-2346, 2016.
[41] Nadine Schneider, Nikolaus Stiefl, 和 Gregory A Landrum。什么是什么是：反应角色分配的（近乎）权威指南。《化学信息与建模杂志》，56(12):2336-2346, 2016。


[42] Srijan Shakya, Anamaria-Roberta Hartl, Sepp Hochreiter, and Korbinian Pöppel. Adaptive retrieval helps reasoning in llms - but mostly if it's not used, 2026. URL https://arxiv.org/abs/2602.07213
[42] Srijan Shakya, Anamaria-Roberta Hartl, Sepp Hochreiter, 和 Korbinian Pöppel。自适应检索有助于大语言模型的推理——但前提是它不被使用，2026。URL https://arxiv.org/abs/2602.07213


[43] Asankhaya Sharma. Openevolve: an open-source evolutionary coding agent. https: //github.com/algorithmicsuperintelligence/openevolve.2025. URL https: //github.com/algorithmicsuperintelligence/openevolve. GitHub repository.
[43] Asankhaya Sharma。Openevolve：一个开源的进化编码智能体。https://github.com/algorithmicsuperintelligence/openevolve.2025。URL https://github.com/algorithmicsuperintelligence/openevolve。GitHub 仓库。


[44] Jake Snell, Kevin Swersky, and Richard S. Zemel. Prototypical networks for few-shot learning. In Advances in Neural Information Processing Systems, 2017.
[44] Jake Snell, Kevin Swersky, 和 Richard S. Zemel。用于少样本学习的原型网络。载于神经信息处理系统进展，2017。


[45] Rich Sutton. The bitter lesson, 2019. URL http://www.incompleteideas.net/IncIdeas/Bitter-Lesson.html, 2019.
[45] Rich Sutton。苦涩的教训，2019。URL http://www.incompleteideas.net/IncIdeas/Bitter-Lesson.html, 2019。


[46] Sebastian Thrun and Lorien Pratt. Learning to learn: Introduction and overview. In Learning to learn, pp. 3-17. Springer, 1998.
[46] Sebastian Thrun 和 Lorien Pratt。学会学习：介绍与概述。载于《学会学习》，第 3-17 页。Springer 出版社，1998。


[47] Muxin Tian, Zhe Wang, Blair Yang, Zhenwei Tang, Kunlun Zhu, Honghua Dong, Hanchen Li, Xinni Xie, Guangjing Wang, and Jiaxuan You. Swe-bench mobile: Can large language model agents develop industry-level mobile applications? In arXiv preprint, 2026. URL https://api.semanticscholar.org/CorpusID:285462974
[47] Muxin Tian, Zhe Wang, Blair Yang, Zhenwei Tang, Kunlun Zhu, Honghua Dong, Hanchen Li, Xinni Xie, Guangjing Wang, 和 Jiaxuan You。SWE-bench Mobile：大语言模型智能体能开发工业级移动应用吗？载于 arXiv 预印本，2026。URL https://api.semanticscholar.org/CorpusID:285462974


[48] Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabharwal. Interleaving retrieval with chain-of-thought reasoning for knowledge-intensive multi-step questions, 2023. URL https://arxiv.org/abs/2212.10509
[48] Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, 和 Ashish Sabharwal。将检索与思维链推理交织用于知识密集型多步问题，2023。URL https://arxiv.org/abs/2212.10509


[49] Chenghao Xiao, G Thomas Hudson, and Noura Al Moubayed. Rar-b: Reasoning as retrieval benchmark, 2024. URL https://arxiv.org/abs/2404.06347
[49] Chenghao Xiao, G Thomas Hudson, 和 Noura Al Moubayed。RAR-B：推理即检索基准，2024。URL https://arxiv.org/abs/2404.06347


[50] Yiming Xiong, Shengran Hu, and Jeff Clune. Learning to continually learn via meta-learning agentic memory designs. In OpenReview, 2026. URL https://api.semanticsc holar.org/CorpusID:285454009
[50] Yiming Xiong, Shengran Hu, 和 Jeff Clune。通过元学习智能体记忆设计学习持续学习。载于 OpenReview，2026。URL https://api.semanticscholar.org/CorpusID:285454009


[51] Chengrun Yang, Xuezhi Wang, Yifeng Lu, Hanxiao Liu, Quoc V Le, Denny Zhou, and Xinyun Chen. Large language models as optimizers. In The Twelfth International Conference on Learning Representations, 2023.
[51] Chengrun Yang, Xuezhi Wang, Yifeng Lu, Hanxiao Liu, Quoc V Le, Denny Zhou, and Xinyun Chen. Large language models as optimizers. In The Twelfth International Conference on Learning Representations, 2023.


[52] Haoran Ye, Xuning He, Vincent Arak, Haonan Dong, and Guojie Song. Meta context engineering via agentic skill evolution. arXiv preprint arXiv:2601.21557, 2026.
[52] Haoran Ye, Xuning He, Vincent Arak, Haonan Dong, and Guojie Song. Meta context engineering via agentic skill evolution. arXiv preprint arXiv:2601.21557, 2026.


[53] Mert Yuksekgonul, Federico Bianchi, Joseph Boen, Sheng Liu, Zhi Huang, Carlos Guestrin, and James Zou. Textgrad: Automatic "differentiation" via text, 2024. URL https://arxiv.org/abs/2406.07496
[53] Mert Yuksekgonul, Federico Bianchi, Joseph Boen, Sheng Liu, Zhi Huang, Carlos Guestrin, and James Zou. Textgrad: Automatic "differentiation" via text, 2024. URL https://arxiv.org/abs/2406.07496


[54] Mert Yuksekgonul, Daniel Koceja, Xinhao Li, Federico Bianchi, Jed McCaleb, Xiaolong Wang, Jan Kautz, Yejin Choi, James Zou, Carlos Guestrin, and Yu Sun. Learning to discover at test time, 2026. URL https://arxiv.org/abs/2601.16175
[54] Mert Yuksekgonul, Daniel Koceja, Xinhao Li, Federico Bianchi, Jed McCaleb, Xiaolong Wang, Jan Kautz, Yejin Choi, James Zou, Carlos Guestrin, and Yu Sun. Learning to discover at test time, 2026. URL https://arxiv.org/abs/2601.16175


[55] Mert Yuksekgonul, Daniel Koceja, Xinhao Li, Federico Bianchi, Jed McCaleb, Xiaolong Wang, Jan Kautz, Yejin Choi, James Zou, Carlos Guestrin, et al. Learning to discover at test time. arXiv preprint arXiv:2601.16175, 2026.
[55] Mert Yuksekgonul, Daniel Koceja, Xinhao Li, Federico Bianchi, Jed McCaleb, Xiaolong Wang, Jan Kautz, Yejin Choi, James Zou, Carlos Guestrin, et al. Learning to discover at test time. arXiv preprint arXiv:2601.16175, 2026.


[56] Alex L. Zhang, Tim Kraska, and Omar Khattab. Recursive language models, 2026. URL https://arxiv.org/abs/2512.24601
[56] Alex L. Zhang, Tim Kraska, and Omar Khattab. Recursive language models, 2026. URL https://arxiv.org/abs/2512.24601


[57] Guibin Zhang, Haotian Ren, Chong Zhan, Zhenhong Zhou, Junhao Wang, He Zhu, Wangchunshu Zhou, and Shuicheng Yan. Memevolve: Meta-evolution of agent memory systems. arXiv preprint arXiv:2512.18746, 2025.
[57] Guibin Zhang, Haotian Ren, Chong Zhan, Zhenhong Zhou, Junhao Wang, He Zhu, Wangchunshu Zhou, and Shuicheng Yan. Memevolve: Meta-evolution of agent memory systems. arXiv preprint arXiv:2512.18746, 2025.


[58] Jiayi Zhang, Jinyu Xiang, Zhaoyang Yu, Fengwei Teng, Xionghui Chen, Jiaqi Chen, Mingchen Zhuge, Xin Cheng, Sirui Hong, Jinlin Wang, Bingnan Zheng, Bang Liu, Yuyu Luo, and Chenglin Wu. Aflow: Automating agentic workflow generation, 2025. URL https://arxiv.org/abs/2410.10762
[58] Jiayi Zhang, Jinyu Xiang, Zhaoyang Yu, Fengwei Teng, Xionghui Chen, Jiaqi Chen, Mingchen Zhuge, Xin Cheng, Sirui Hong, Jinlin Wang, Bingnan Zheng, Bang Liu, Yuyu Luo, and Chenglin Wu. Aflow: Automating agentic workflow generation, 2025. URL https://arxiv.org/abs/2410.10762


[59] Qizheng Zhang, Changran Hu, Shubhangi Upasani, Boyuan Ma, Fenglu Hong, V. Ka-manuru, Jay Rainton, Chen Wu, Mengmeng Ji, Hanchen Li, Urmish Thakker, James Zou, and K. Olukotun. Agentic context engineering: Evolving contexts for self-improving language models. In arXiv preprint arXiv:2510.04618, 2025.
[59] Qizheng Zhang, Changran Hu, Shubhangi Upasani, Boyuan Ma, Fenglu Hong, V. Ka-manuru, Jay Rainton, Chen Wu, Mengmeng Ji, Hanchen Li, Urmish Thakker, James Zou, and K. Olukotun. Agentic context engineering: Evolving contexts for self-improving language models. In arXiv preprint arXiv:2510.04618, 2025.


[60] Xiang Zhang, Junbo Zhao, and Yann LeCun. Character-level convolutional networks for text classification, 2016. URL https://arxiv.org/abs/1509.01626
[60] Xiang Zhang, Junbo Zhao, and Yann LeCun. Character-level convolutional networks for text classification, 2016. URL https://arxiv.org/abs/1509.01626


<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/05/2026_05_11__23_09_36_ca345f.jpg"/>



Figure 4: Search-set accuracy over evaluations for all compared text optimizers on online text classification. Each point is one candidate harness; lines track the best-so-far. Per-dataset curves are shown alongside the aggregate. Meta-Harness reaches the final accuracy of OpenEvolve and TTT-Discover within the first 4 evaluations and continues improving, ending more than 10 points above all baselines.
图 4：在线文本分类任务中，各对比文本优化器在评估过程中的搜索集准确率。每个点代表一个候选方案（harness）；折线追踪了迄今为止的最优值。图中展示了各数据集的曲线及汇总结果。Meta-Harness 在前 4 次评估内即达到了 OpenEvolve 和 TTT-Discover 的最终准确率，并持续提升，最终领先所有基线超过 10 个百分点。


## A Qualitative Proposer Behavior
## A 提议者行为定性分析


This section examines how the proposer uses the filesystem during search, drawing on the TerminalBench-2 run (10 iterations, Claude Opus 4.6).
本节基于 TerminalBench-2 的运行数据（10 次迭代，Claude Opus 4.6），考察提议者（proposer）在搜索过程中如何使用文件系统。


### A.1 File Access Statistics
### A.1 文件访问统计


To verify that the proposer makes substantive use of the filesystem rather than defaulting to local edits, we recorded all file reads per iteration.
为验证提议者确实对文件系统进行了实质性利用，而非仅依赖本地编辑，我们记录了每次迭代中的所有文件读取操作。


Table 8 summarizes the results. The proposer reads a median of 82 files per iteration (range 69-99), roughly evenly split between prior harness source code (41%) and execution traces (40%), with the remainder going to score summaries (6%) and other files (13%). This confirms that the proposer's access pattern is non-Markovian: it routinely inspects the majority of available history rather than conditioning only on the most recent parent.
表 8 总结了结果。提议者每次迭代平均读取 82 个文件（范围 69-99），其中方案源代码（41%）与执行轨迹（40%）占比大致相当，其余为评分摘要（6%）及其他文件（13%）。这证实了提议者的访问模式是非马尔可夫的：它通常会检查大部分可用历史记录，而非仅依赖于最近的父节点。


<table><tr><td>Statistic</td><td>Value</td></tr><tr><td>Files read per iteration (median)</td><td>82</td></tr><tr><td>Files read per iteration (range)</td><td>69-99</td></tr><tr><td colspan="2">File type breakdown</td></tr><tr><td>Harness source code</td><td>41%</td></tr><tr><td>Execution traces</td><td>40%</td></tr><tr><td>Score/summary files</td><td>6%</td></tr><tr><td>Other</td><td>13%</td></tr></table>
<table><tbody><tr><td>统计指标</td><td>数值</td></tr><tr><td>单次迭代读取文件数（中位数）</td><td>82</td></tr><tr><td>单次迭代读取文件数（范围）</td><td>69-99</td></tr><tr><td colspan="2">文件类型分布</td></tr><tr><td>测试框架源代码</td><td>41%</td></tr><tr><td>执行追踪记录</td><td>40%</td></tr><tr><td>评分/摘要文件</td><td>6%</td></tr><tr><td>其他</td><td>13%</td></tr></tbody></table>


Table 8: Proposer file access statistics from the TerminalBench-2 search run (10 iterations, Claude Opus 4.6). The proposer reads extensively from the filesystem, with roughly equal attention to prior source code and execution traces.
表 8：来自 TerminalBench-2 搜索运行（10 次迭代，Claude Opus 4.6）的提议者文件访问统计。提议者广泛读取文件系统，对既往源代码和执行轨迹的关注度大致相当。


### A.2 Qualitative Behavior: Causal Reasoning Over Prior Failures
### A.2 定性行为：针对既往失败的因果推理


The TerminalBench-2 search log reveals a clear narrative arc in which the proposer learns from its own regressions. Rather than wandering randomly through local edits, it forms an explicit diagnosis of why early candidates failed, then shifts toward a safer design pattern. All text inside the log boxes below is quoted verbatim from the proposer's recorded reasoning at each iteration (emphasis ours).
TerminalBench-2 搜索日志揭示了一条清晰的叙事弧线，即提议者从自身的回归中学习。它并非在局部编辑中盲目摸索，而是对早期候选方案失败的原因进行了明确诊断，随后转向更稳妥的设计模式。下方日志框内的所有文本均逐字引用自提议者在每次迭代中记录的推理过程（重点由我们标注）。


Iterations 1-2: promising bugfixes are confounded by prompt edits. The first two iterations both bundle plausible structural fixes with prompt-template modifications, and both regress sharply from the 64.4% Terminus-KIRA baseline. Iteration 1 targets observation corruption from leaked terminal markers and adds a loop breaker:
迭代 1-2：有前景的错误修复被提示词编辑所干扰。前两次迭代均将合理的结构性修复与提示词模板修改捆绑在一起，且两者均较 64.4% 的 Terminus-KIRA 基准出现了大幅回归。迭代 1 旨在解决终端标记泄露导致的观测损坏，并添加了循环中断机制：


---



&nbsp;&nbsp;&nbsp;&nbsp;Hypothesis: _CMDEND__ marker fragments leak into LLM observations on long-running
&nbsp;&nbsp;&nbsp;&nbsp;假设：_CMDEND__ 标记片段在长时间运行的任务中泄露到 LLM 观测中，


tasks, causing the model to get confused and enter infinite no-tool-call loops.
导致模型混淆并进入无限的无工具调用循环。


&nbsp;&nbsp;&nbsp;&nbsp;Stripping these markers + adding a loop breaker will recover wasted steps.
&nbsp;&nbsp;&nbsp;&nbsp;剥离这些标记并添加循环中断将挽回浪费的步骤。


---



That candidate also introduced a new cleanup-oriented prompt template and a verification checklist. Iteration 2 proposes a different state-machine fix:
该候选方案还引入了一个新的面向清理的提示词模板和一个验证清单。迭代 2 提出了另一种状态机修复方案：


---



Double-confirmation completion mechanism causes verification spirals. Observed in
双重确认完成机制导致验证螺旋。在代理提前解决任务但浪费了 15--40+ 个额外步骤


trajectories where the agent solves the task early but burns 15--40+ additional steps
进行重复验证的轨迹中观察到，因为每个验证命令都会重置 pending_completion，


re-verifying because each verification command resets pending_completion, requiring
重新验证，因为每个验证命令都会重置 pending_completion，需要


&nbsp;&nbsp;&nbsp;&nbsp;another task_complete $\rightarrow$ checklist $\rightarrow$ verify cycle.
&nbsp;&nbsp;&nbsp;&nbsp;另一个 task_complete $\rightarrow$ 清单 $\rightarrow$ 验证周期。


---



This second candidate removes the pending-completion mechanism entirely, while also carrying over the marker stripping and the new prompt. It still regresses, which gives the proposer two failed candidates with different structural changes but one shared prompt intervention.
这第二个候选方案完全移除了 pending-completion 机制，同时也沿用了标记剥离和新提示词。它仍然出现了回归，这使得提议者获得了两个失败的候选方案，它们具有不同的结构性变更，但共享了同一个提示词干预。


Iteration 3: the proposer identifies the confound. By iteration 3, the proposer explicitly infers that the regressions are not primarily due to the structural bugfixes themselves:
迭代 3：提议者识别出混杂因素。到迭代 3 时，提议者明确推断出回归并非主要由结构性错误修复本身引起：


---



Prior attempts: evo_marker_fix (58.9%, -5.6pp), evo_single_confirm (57.8%, -6.7pp)
既往尝试：evo_marker_fix (58.9%, -5.6pp), evo_single_confirm (57.8%, -6.7pp)


--- both regressed. Root cause of regressions: Prompt template changes (cleanup
--- 两者均出现回归。回归的根本原因：提示词模板变更（清理


directives) caused the agent to delete necessary state before task completion. The
指令）导致智能体在任务完成前删除了必要状态。该


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;structural bugfixes were confounded with harmful prompt changes. evo_strip_only
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;结构性错误修复与有害的提示词修改混杂在一起。evo_strip_only


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;isolates the two proven structural fixes.
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;隔离了这两个已证实的结构性修复。


---



This is the key causal step in the trajectory. The proposer notices that the common factor across the first two failures is not the particular bugfix, but the cleanup-heavy prompt rewrite. It therefore reverts to the original prompt and tests only the marker-stripping and loop-breaker. The resulting candidate still underperforms slightly (63.3%, -1.1pp), but it loses far less than the earlier versions, which supports the confound diagnosis.
这是轨迹中关键的因果步骤。提议者注意到前两次失败的共同因素并非特定的错误修复，而是清理力度过大的提示词重写。因此，它回退到原始提示词，仅测试标记剥离和循环中断。最终生成的候选方案表现仍略有不足（63.3%，-1.1pp），但相比早期版本损失大幅减少，这支持了对混杂因素的诊断。


Iterations 4-6: direct fixes to the diagnosed failure mode still regress. The next three iterations continue to probe the same part of the design space, but now with more explicit theories about why the completion logic is fragile. Iteration 4 attributes failures to a concrete state-machine bug in which verification commands reset the completion flag and trap the agent in repeated checklist cycles:
第4-6次迭代：针对已诊断故障模式的直接修复仍然出现倒退。接下来的三次迭代继续探索设计空间的同一部分，但现在对完成逻辑为何脆弱有了更明确的理论。第4次迭代将失败归因于一个具体的状态机错误，即验证命令重置了完成标志，导致智能体陷入重复的检查清单循环：


---



&nbsp;&nbsp;&nbsp;&nbsp;Remove the two self._pending_completion = False lines that reset the completion
&nbsp;&nbsp;&nbsp;&nbsp;移除中间命令运行时重置完成标志的两个 self._pending_completion = False 行。


flag when intermediate commands run. This fixes a state machine bug where: (1)
这修复了一个状态机错误，其中：（1）


Agent calls task_complete $\rightarrow$ sees QA checklist,_pending_completion = True (2) Agent
智能体调用 task_complete $\rightarrow$ 查看 QA 检查清单，_pending_completion = True（2）智能体


runs verification commands $\rightarrow$ _pending_completion = False (bug!) (3) Agent calls
运行验证命令 $\rightarrow$ _pending_completion = False（错误！）（3）智能体再次调用


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;task_complete again $\rightarrow$ sees checklist AGAIN $\rightarrow$ infinite loop.
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;task_complete $\rightarrow$ 再次查看检查清单 $\rightarrow$ 无限循环。


---



The proposer even cites concrete trajectory evidence, noting that configure-git-webserver produced baseline failures with agents stuck in 30-60 step verification spirals after effectively solving the task. Iteration 5 tries to soften the cleanup language while preserving confirmation, but still edits the prompt and regresses badly. Iteration 6 returns to the safer evo_strip_only base and proposes a systems-level optimization:
提议者甚至引用了具体的轨迹证据，指出 configure-git-webserver 在有效解决任务后，智能体仍陷入 30-60 步的验证螺旋，从而导致基准测试失败。第5次迭代尝试在保留确认机制的同时弱化清理措辞，但仍修改了提示词并导致严重倒退。第6次迭代回归到更安全的 evo_strip_only 基础，并提出了系统级的优化：


---



Empty-command turns waste full LLM round-trips when terminal output hasn't changed.
当终端输出未改变时，空命令会导致 LLM 往返请求的浪费。


Smart-waiting (poll pane up to $3 \times  5\mathrm{\;s}$ ) before the next LLM call saves 5--15 turns on
在下一次 LLM 调用前进行智能等待（轮询面板至 $3 \times  5\mathrm{\;s}$ ）可在


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;long-running tasks.
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;长时间运行的任务中节省 5--15 次轮次。


---



That change also regresses. By this point, the proposer has learned a specific empirical lesson: modifications to prompts and completion flow are high risk, even when the local hypothesis sounds reasonable.
该更改同样导致了倒退。至此，提议者已吸取了一个具体的经验教训：即使局部假设听起来合理，对提示词和完成流程的修改也具有高风险。


Iteration 7: the winning candidate. After six consecutive regressions, the proposer shifts strategy from modifying the control loop to adding information before the loop begins:
第7次迭代：获胜候选方案。在连续六次倒退后，提议者将策略从修改控制循环转变为在循环开始前添加信息：


---



All 6 prior iterations regressed from the 64.4% baseline because they modified the
所有6次先前的迭代都较64.4%的基准有所倒退，因为它们修改了


completion flow, prompt template, or observation processing. evo_env_bootstrap takes
补全流程、提示词模板或观察处理。evo_env_bootstrap采取了


a different approach --- purely additive. It gathers an environment snapshot via a
一种不同的方法——纯粹的加法式。它在首次LLM调用前通过


single shell command before the first LLM call and appends it to the initial prompt.
单个shell命令收集环境快照，并将其附加到初始提示词中。


No other methods are changed. This should eliminate 3--5 wasted exploration turns on
其他方法均未改变。这应能消除3--5个在依赖项繁重的任务上


&nbsp;&nbsp;&nbsp;&nbsp;dependency-heavy tasks without risking regression on already-passing tasks.
&nbsp;&nbsp;&nbsp;&nbsp;浪费的探索轮次，且不会导致已通过任务出现倒退的风险。


---



This candidate is the best result so far. The important point is not just that iteration 7 wins, but that the proposer articulates why it should be safer: it avoids touching the previously fragile completion machinery and instead adds information that is useful mainly on hard tasks.
该候选方案是迄今为止的最佳结果。重点不仅在于第7次迭代胜出，还在于提议者阐明了为何它更安全：它避免触碰此前脆弱的补全机制，转而添加主要对困难任务有用的信息。


Iteration 8: composition. Having found one additive improvement, the proposer next attempts to compose it with an earlier structural fix:
第8次迭代：组合。在找到一个加法式改进后，提议者接下来尝试将其与早期的结构性修复相结合：


---



&nbsp;&nbsp;&nbsp;&nbsp;Combining two orthogonal fixes --- env snapshot (saves early exploration turns) +
&nbsp;&nbsp;&nbsp;&nbsp;结合两个正交的修复——环境快照（节省早期探索轮次）+


marker stripping with no-tool-call loop breaker --- will yield +1--3pp because they
带有无工具调用循环中断器的标记剥离——将带来+1--3个百分点的提升，因为它们


address independent failure modes without touching prompts or confirmation flows
解决了独立的故障模式，而未触碰提示词或确认流程


&nbsp;&nbsp;&nbsp;&nbsp;(which caused regressions in 5 of 7 prior iterations).
&nbsp;&nbsp;&nbsp;&nbsp;（这在7次先前迭代中的5次里导致了倒退）。


---



Iteration 10: cross-run transfer. The proposer references results from a separate earlier search run:
第10次迭代：跨运行迁移。提议者引用了之前另一次搜索运行的结果：


---



The evolution history showed "don't cleanup service artifacts" was worth +18pp. Iter
演化历史显示“不要清理服务工件”带来了+18个百分点的提升。第


9 (evo_no_cleanup_directive) targeted the same idea but crashed before evaluation.
9次迭代（evo_no_cleanup_directive）针对同一思路，但在评估前崩溃了。


---



Summary. The search trajectory demonstrates that the proposer does more than random mutation. Across the first seven iterations, it identifies a confound, tests the confound-isolating hypothesis directly, observes that control-flow and prompt edits remain fragile, and then deliberately pivots to a purely additive modification that becomes the best candidate in the run. It subsequently tries to compose that winning idea with earlier fixes and even transfers lessons across runs. This kind of causal reasoning over prior failures is precisely what full-history filesystem access enables and what compressed-feedback optimizers cannot support.
总结。搜索轨迹表明，提议者所做的不仅仅是随机变异。在前七次迭代中，它识别出一个混杂因素，直接测试了隔离该因素的假设，观察到控制流和提示词编辑依然脆弱，随后果断转向纯粹的加法式修改，并使其成为本次运行中的最佳候选方案。随后，它尝试将该胜出思路与早期修复相结合，甚至实现了跨运行的经验迁移。这种针对先前失败的因果推理，正是全历史文件系统访问所能实现的，也是压缩反馈优化器无法支持的。


## B Discovered Harnesses
## B 已发现的工具链


Meta-Harness discovers executable inference-time procedures specific to the problem setup at hand. These harnesses are structured, domain-specific policies, often with nontrivial control flow such as routing, filtering, and conditional context construction, selected solely by whether they improve search-set performance. This section presents compact, method-style abstractions of representative harnesses that summarize the main behaviors and control-flow decisions that drive inference-time behavior. For reference, the full implementation for each discovered harness is on the order of 100-1000 lines of code.
Meta-Harness 能够针对当前问题设置发现可执行的推理时程序。这些工具链是结构化的、特定领域的策略，通常包含路由、过滤和条件上下文构建等非平凡控制流，其选择标准仅在于它们是否能提升搜索集的性能。本节展示了代表性工具链的紧凑型方法论抽象，总结了驱动推理时行为的主要逻辑和控制流决策。作为参考，每个已发现工具链的完整实现代码量约为 100-1000 行。


<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/05/2026_05_11__23_09_36_07108d.jpg"/>



Figure 5: Draft-verification classification harness. The first call produces a draft label from a short retrieved context. The second call retrieves evidence for and against that draft and returns the final prediction.
图 5：草稿验证分类工具链。第一次调用根据简短的检索上下文生成草稿标签。第二次调用检索支持和反对该草稿的证据，并返回最终预测。


### B.1 Text Classification Harness
### B.1 文本分类工具链


In online text classification, Meta-Harness discovers a family of memory-based harnesses rather than a single canonical policy. Table 9 reports the Pareto frontier of non-dominated variants from the main search, all selected solely by search-set performance. We highlight two representative endpoints here: Meta-Harness (Draft Verification), the lowest-context frontier point, and Meta-Harness (Label-Primed Query), the highest-accuracy frontier point used in the main text.
在在线文本分类中，Meta-Harness 发现的是一系列基于记忆的工具链，而非单一的规范策略。表 9 报告了从主搜索中选出的非支配变体的帕累托前沿，所有变体均仅根据搜索集性能进行选择。我们在此重点介绍两个代表性端点：Meta-Harness（草稿验证），即上下文需求最低的前沿点；以及 Meta-Harness（标签引导查询），即正文中使用的准确率最高的前沿点。


Overview. Both harnesses maintain a growing memory of past labeled examples and build prompts from that memory at inference time. What differs is the control flow used to interrogate the memory. Meta-Harness (Draft Verification) uses two short calls and explicitly tests the model's first guess against retrieved counterexamples, while Meta-Harness (Label-Primed Query) spends a larger single-call budget on making the label space and local decision boundaries explicit. Figures 5 and 6 summarize these two programs.
概述。两种工具链都维护着一个不断增长的过往标注示例记忆库，并在推理时利用该记忆构建提示词。不同之处在于查询记忆所使用的控制流。Meta-Harness（草稿验证）使用两次短调用，并明确根据检索到的反例测试模型的初步猜测；而 Meta-Harness（标签引导查询）则在单次调用中投入更多预算，以明确标签空间和局部决策边界。图 5 和图 6 总结了这两个程序。


Meta-Harness (Draft Verification). The corresponding discovered file is draft_verificat ion.py. This lightweight variant turns prediction into a two-call procedure. It first retrieves the 5 most similar labeled examples and makes a draft prediction. It then re-queries the same memory conditioned on that draft label, retrieving 5 confirmers with the same label and 5 challengers with different labels, and asks the model whether to maintain or revise its initial answer. The key discovered behavior is that the second retrieval depends on both the query and the draft prediction, so the harness can surface counterexamples targeted at the model's current guess rather than only generic near neighbors. If too few labeled examples have been accumulated, the program falls back to a standard single-call few-shot prompt.
Meta-Harness（草稿验证）。对应的发现文件为 draft_verification.py。这种轻量级变体将预测转化为两步调用过程。首先检索 5 个最相似的标注示例并做出草稿预测。然后，以该草稿标签为条件再次查询同一记忆库，检索 5 个同标签的确认示例和 5 个不同标签的挑战示例，并询问模型是维持还是修正其初始答案。其发现的关键行为在于，第二次检索同时依赖于查询和草稿预测，因此工具链能够针对模型当前的猜测精准呈现反例，而非仅仅提供通用的近邻示例。如果积累的标注示例过少，程序将回退到标准的单次调用少样本提示。


- Stage 1: Draft. Retrieve the 5 nearest labeled examples and ask for an initial prediction.
- 第一阶段：草稿。检索 5 个最接近的标注示例并要求进行初步预测。


- Stage 2: Verification. Condition retrieval on the draft label, then show both supporting and challenging examples before making the final prediction.
- 第二阶段：验证。以草稿标签为条件进行检索，并在做出最终预测前展示支持和挑战示例。


- Cold start. If fewer than 5 labeled examples are available, skip the two-stage procedure and use a standard single-call few-shot prompt.
- 冷启动。如果标注示例少于 5 个，则跳过两阶段过程，使用标准的单次调用少样本提示。


- Why it is cheap. Both calls use short retrieved contexts, so the overall context cost stays near the low end of the frontier even with two model invocations.
- 低成本原因。两次调用均使用简短的检索上下文，因此即使进行了两次模型调用，整体上下文成本仍保持在前沿的低位。


<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/05/2026_05_11__23_09_36_70b4b1.jpg"/>



Figure 6: Label-primed query-anchored classification harness. The program builds a single prompt that exposes the label space, then populates it with query-relevant coverage examples and local contrastive pairs.
图 6：标签引导查询锚定分类工具链。该程序构建了一个单一提示词，展示了标签空间，并填充了与查询相关的覆盖示例和局部对比对。


Meta-Harness (Label-Primed Query). The corresponding discovered file is label_prime d_query_anchored.py. This strongest variant uses a single larger call built from three parts. It begins with a label primer listing the valid output labels, then constructs a coverage section with one query-relevant example per label, and finally adds query-anchored contrastive pairs that place highly similar examples with different labels side by side. The coverage block exposes the full label space, while the contrastive block sharpens local decision boundaries around the current query. In code, the harness implements this with TF-IDF retrieval over past labeled examples and a query-anchored pairing rule that chooses contrasting examples from the same local neighborhood.
Meta-Harness（标签引导查询）。对应的发现文件为 label_primed_query_anchored.py。这种最强的变体使用由三部分组成的单次较大调用。它首先以列出有效输出标签的标签引导部分开始，然后构建一个包含每个标签一个查询相关示例的覆盖部分，最后添加查询锚定对比对，将标签不同但高度相似的示例并列放置。覆盖块展示了完整的标签空间，而对比块则强化了当前查询周围的局部决策边界。在代码实现上，该工具链通过对过往标注示例进行 TF-IDF 检索，并采用从同一局部邻域选择对比示例的查询锚定配对规则来实现。


- Label primer. List the valid output labels before showing any examples, so the model sees the full answer space up front.
- 标签引导。在展示任何示例前先列出有效输出标签，以便模型预先了解完整的答案空间。


- Coverage block. For each known label, retrieve the most query-relevant labeled example and include one representative example per class.
- 覆盖块。针对每个已知标签，检索最相关的标注示例，并包含每个类别的一个代表性示例。


- Contrastive block. Build pairs of highly similar examples with different labels, so the prompt exposes local decision boundaries around the current query.
- 对比块。构建标签不同但高度相似的示例对，使提示词能够展示当前查询周围的局部决策边界。


- Retrieval rule. Use TF-IDF similarity and query-anchored partner selection rather than label-agnostic nearest neighbors.
- 检索规则。使用 TF-IDF 相似度和基于查询锚点的伙伴选择，而非与标签无关的最近邻算法。


### B.2 Math Retrieval Harness
### B.2 数学检索工具


This subsection describes the retrieval harness discovered by Meta-Harness for mathematical reasoning (Section 4.2). The final harness is a compact four-route BM25 program whose structure emerged through search rather than being manually specified after the fact. All design choices below-the routing predicates, reranking terms, deduplication thresholds, and per-route example counts-were selected by the outer loop across 40 iterations of evolution.
本小节介绍了 Meta-Harness 为数学推理任务发现的检索工具（第 4.2 节）。最终的工具是一个紧凑的四路径 BM25 程序，其结构是通过搜索演化而来，而非事后手动指定。下述所有设计选择——包括路由谓词、重排序项、去重阈值以及各路径的示例数量——均由外循环经过 40 次迭代演化选定。


<table><tr><td rowspan="2">Variant</td><td colspan="3">Datasets</td><td colspan="2">Avg metrics</td></tr><tr><td>USPTO $\uparrow$</td><td>Symptom $\uparrow$</td><td>LawBench↑</td><td>Avg↑</td><td>Ctx $\downarrow$</td></tr><tr><td>Meta-Harness (Draft Verification)</td><td>18.0</td><td>85.4</td><td>17.0</td><td>40.1</td><td>5.4</td></tr><tr><td>Meta-Harness (Error-Annotated)</td><td>9.0</td><td>87.7</td><td>24.0</td><td>40.2</td><td>22.3</td></tr><tr><td>Meta-Harness (CoT Replay)</td><td>13.0</td><td>88.2</td><td>25.0</td><td>42.1</td><td>23.3</td></tr><tr><td>Meta-Harness (Cluster Coverage)</td><td>12.0</td><td>86.8</td><td>33.0</td><td>43.9</td><td>31.2</td></tr><tr><td>Meta-Harness (Cascade Retrieval)</td><td>12.0</td><td>86.8</td><td>36.0</td><td>44.9</td><td>39.2</td></tr><tr><td>Meta-Harness (RRF + Contrastive)</td><td>18.0</td><td>89.6</td><td>35.0</td><td>47.5</td><td>41.4</td></tr><tr><td>Meta-Harness (Relevance + Contrastive)</td><td>18.0</td><td>90.6</td><td>36.0</td><td>48.2</td><td>43.9</td></tr><tr><td>Meta-Harness (Label-Primed Query)</td><td>14.0</td><td>86.8</td><td>45.0</td><td>48.6</td><td>45.5</td></tr></table>
<table><tbody><tr><td rowspan="2">变体</td><td colspan="3">数据集</td><td colspan="2">平均指标</td></tr><tr><td>USPTO $\uparrow$</td><td>Symptom $\uparrow$</td><td>LawBench↑</td><td>平均值↑</td><td>Ctx $\downarrow$</td></tr><tr><td>Meta-Harness（草稿验证）</td><td>18.0</td><td>85.4</td><td>17.0</td><td>40.1</td><td>5.4</td></tr><tr><td>Meta-Harness（错误标注）</td><td>9.0</td><td>87.7</td><td>24.0</td><td>40.2</td><td>22.3</td></tr><tr><td>Meta-Harness（思维链重放）</td><td>13.0</td><td>88.2</td><td>25.0</td><td>42.1</td><td>23.3</td></tr><tr><td>Meta-Harness（聚类覆盖）</td><td>12.0</td><td>86.8</td><td>33.0</td><td>43.9</td><td>31.2</td></tr><tr><td>Meta-Harness（级联检索）</td><td>12.0</td><td>86.8</td><td>36.0</td><td>44.9</td><td>39.2</td></tr><tr><td>Meta-Harness（RRF + 对比学习）</td><td>18.0</td><td>89.6</td><td>35.0</td><td>47.5</td><td>41.4</td></tr><tr><td>Meta-Harness（相关性 + 对比学习）</td><td>18.0</td><td>90.6</td><td>36.0</td><td>48.2</td><td>43.9</td></tr><tr><td>Meta-Harness（标签引导查询）</td><td>14.0</td><td>86.8</td><td>45.0</td><td>48.6</td><td>45.5</td></tr></tbody></table>


Table 9: Pareto-optimal discovered variants from the main text-classification search, trading off average accuracy against context cost. The selected system in the main text is Meta-Harness (Label-Primed Query). Ctx denotes average additional characters in input context (thousands).
表 9：来自主要文本分类搜索的帕累托最优发现变体，权衡了平均准确率与上下文成本。正文中选定的系统是 Meta-Harness（标签引导查询）。Ctx 表示输入上下文中平均增加的字符数（千为单位）。


<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/05/2026_05_11__23_09_36_a4739e.jpg"/>



Figure 7: Search-set vs. test accuracy per dataset for discovered text-classification strategies. Each pink dot is a discovered strategy; baselines are labeled. The dashed diagonal is $y = x$ .
图 7：针对已发现的文本分类策略，各数据集的搜索集准确率与测试准确率对比。每个粉色点代表一种已发现的策略；基线已标注。虚线对角线为 $y = x$ 。


Overview. At inference time, the harness assigns each problem to exactly one of four routes: combinatorics, geometry, number theory, or a default route for algebra and other problems. The gates are implemented as lightweight lexical predicates over the problem statement, including keyword sets and a small number of regex features for geometry notation. The harness does not aggregate outputs across routes: once a route is selected, only that route retrieves examples for the final prompt. All routes use BM25 as the underlying retrieval mechanism over the filtered corpus described above. The BM25 index uses a math-aware tokenizer that preserves LaTeX tokens (e.g., \\frac, ^\{2\}) as atomic units. The selected harness is a merge of two successful search lineages, autonomously combined by the proposer during search: one contributed a stronger geometry route based on raw BM25, while another contributed a stronger combinatorics route based on deduplication and difficulty reranking. Figure 8 gives a compact flowchart view of the final program.
概述。在推理时，该工具链将每个问题精确分配到四条路径之一：组合数学、几何、数论，或代数及其他问题的默认路径。门控机制实现为针对问题陈述的轻量级词法谓词，包括关键词集和少量用于几何符号的正则表达式特征。该工具链不会跨路径聚合输出：一旦选定路径，仅该路径会为最终提示词检索示例。所有路径均使用 BM25 作为上述过滤语料库的基础检索机制。BM25 索引使用了一种数学感知分词器，可将 LaTeX 标记（例如 \\frac, ^\{2\}）保留为原子单元。所选工具链是两个成功搜索谱系的合并，由提议者在搜索过程中自主组合：一个贡献了基于原始 BM25 的更强几何路径，另一个贡献了基于去重和难度重排序的更强组合数学路径。图 8 给出了最终程序的紧凑流程图视图。


- Combinatorics: fetch 20 BM25 candidates, deduplicate to 8, rerank by lexical score and difficulty, then return the top 3. This is the main route where the harness explicitly trades off diversity against hard-problem matching.
- 组合数学：获取 20 个 BM25 候选结果，去重至 8 个，按词法得分和难度重排序，然后返回前 3 个。这是该工具链明确权衡多样性与难题匹配的主要路径。


- Geometry: return 1 hard NuminaMath reference together with 2 raw BM25 neighbors. Search consistently prefers raw structural matches here over difficulty reranking.
- 几何：返回 1 个困难的 NuminaMath 参考示例，连同 2 个原始 BM25 近邻。搜索过程在此处始终偏好原始结构匹配，而非难度重排序。


- Number theory: fetch 12 BM25 candidates and rerank using lexical score, difficulty, and a small bonus for solutions that state a technique early. This favors examples whose proof strategy is explicit.
- 数论：获取 12 个 BM25 候选结果，并使用词法得分、难度以及对早期陈述解题技巧的示例给予少量奖励进行重排序。这偏向于证明策略明确的示例。


- Default: fetch 10 BM25 candidates, rerank by lexical score and difficulty, and choose an adaptive number of examples based on how concentrated the top retrieval scores are.
- 默认：获取 10 个 BM25 候选结果，按词法得分和难度重排序，并根据顶部检索得分的集中程度选择自适应数量的示例。


<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/05/2026_05_11__23_09_36_c18bb3.jpg"/>



Figure 8: Discovered math retrieval harness. A lexical router assigns each query to one of four subject-specific retrieval policies. The selected policy retrieves examples, which are inserted into the final prompt.
图 8：已发现的数学检索工具链。词法路由器将每个查询分配给四个特定学科检索策略之一。所选策略检索示例并将其插入最终提示词中。


### B.3 TerminalBench-2 Harness
### B.3 TerminalBench-2 工具链


The discovered TerminalBench-2 harness builds on Terminus-KIRA [25], inheriting its native tool calling (replacing Terminus 2's ICL-based JSON parsing), 30KB output cap, and multi-perspective completion checklist. The main modification discovered by Meta-Harness is environment bootstrapping: before the agent loop begins, the harness runs a compound shell command to gather a snapshot of the sandbox environment and injects it into the initial prompt. The proposer's hypothesis, recorded verbatim from the search log, was:
已发现的 TerminalBench-2 工具链基于 Terminus-KIRA [25] 构建，继承了其原生工具调用（取代了 Terminus 2 基于 ICL 的 JSON 解析）、30KB 输出上限以及多视角完成检查清单。Meta-Harness 发现的主要改进是环境引导：在智能体循环开始前，工具链运行一个复合 shell 命令以获取沙盒环境快照，并将其注入初始提示词中。提议者从搜索日志中逐字记录的假设如下：


---



Hypothesis: 'Injecting an environment snapshot (OS, installed languages, package
假设：“在第一个 LLM 回合前注入环境快照（操作系统、已安装语言、包


managers, /app contents) before the first LLM turn will reduce wasted exploration
管理器、/app 内容）将减少依赖密集型任务中


&nbsp;&nbsp;&nbsp;&nbsp;episodes by 3--5 turns on dependency-heavy tasks''
&nbsp;&nbsp;&nbsp;&nbsp;3--5 个回合的无效探索”


Changes: "Added gather_env_snapshot() that runs a single compound shell command to
变更：“添加了 gather_env_snapshot()，它运行单个复合 shell 命令以


collect working directory, /app listing, available languages (python, gcc, node, java,
收集工作目录、/app 列表、可用语言（python, gcc, node, java,


&nbsp;&nbsp;&nbsp;&nbsp;rustc, go), package managers (pip, apt) [...] and injects as [Environment Snapshot]
&nbsp;&nbsp;&nbsp;&nbsp;rustc, go）、包管理器（pip, apt）[...] 并将其注入为 [环境快照]”


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;block"'
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;block"'


---



The snapshot includes: the working directory, a listing of /app (truncated to 20 entries for large directories), available programming languages and their versions (Python, GCC, G++, Node, Java, Rust, Go), installed package managers (pip, apt-get), and available memory. This eliminates the 2-4 exploratory turns that agents typically spend discovering what tools and files are available, allowing the model to begin productive work immediately. The bootstrapping command is guarded by a 15-second timeout and fails silently, so it does not break the agent in unusual environments. The full implementation adds roughly 80 lines on top of Terminus-KIRA. Figure 9 summarizes the harness structure.
快照包含：工作目录、/app 目录列表（大型目录截断为前 20 项）、可用编程语言及其版本（Python、GCC、G++、Node、Java、Rust、Go）、已安装的包管理器（pip、apt-get）以及可用内存。这省去了智能体通常用于探索可用工具和文件所需的 2-4 个回合，使模型能够立即开始高效工作。引导命令设有 15 秒超时限制且静默失败，因此不会在异常环境中导致智能体崩溃。完整实现是在 Terminus-KIRA 基础上增加了约 80 行代码。图 9 总结了该工具链的结构。


Per-task analysis. Compared to Terminus-KIRA, the discovered harness gains on 7 of 89 tasks, with the largest improvements on protein-assembly and path-tracing. The gaining tasks share a common property: they require domain-specific tooling whose availability cannot be assumed in advance (bioinformatics libraries, rendering pipelines, chess engines, cryptographic utilities, CoreWars simulators). Without the bootstrap, the agent spends its
分任务分析。与 Terminus-KIRA 相比，所发现的工具链在 89 个任务中的 7 个上有所提升，其中在蛋白质组装和路径追踪任务上的改进最为显著。这些获益任务具有一个共同特征：它们需要预先无法确定的特定领域工具（生物信息学库、渲染流水线、国际象棋引擎、加密工具、CoreWars 模拟器）。如果没有引导程序，智能体需要花费其


<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/05/2026_05_11__23_09_36_1dff77.jpg"/>



Figure 9: Discovered TerminalBench-2 harness. The harness inherits Terminus-KIRA's native tool calling, output cap, and completion checklist (green). The environment bootstrap (red) is the component discovered by Meta-Harness: it gathers a sandbox snapshot before the agent loop begins, eliminating early exploratory turns.
图 9：发现的 TerminalBench-2 工具链。该工具链继承了 Terminus-KIRA 的原生工具调用、输出上限和完成清单（绿色）。环境引导程序（红色）是 Meta-Harness 发现的组件：它在智能体循环开始前获取沙箱快照，从而消除了早期的探索性回合。


first 2-4 turns probing the environment; on tasks with tight turn budgets or where early wrong assumptions cascade, those wasted turns can be the difference between pass and fail. This suggests that the bootstrap's value is largest when the environment is non-obvious, and the task requires the agent to match its strategy to what is actually installed.
最初的 2-4 个回合来探测环境；在回合预算紧张或早期错误假设会产生连锁反应的任务中，这些浪费的回合可能就是成功与失败的区别。这表明，当环境不直观且任务要求智能体根据实际安装情况调整策略时，引导程序的价值最大。


## C Dataset Details
## C 数据集详情


### C.1 OOD Text Classification Datasets
### C.1 OOD 文本分类数据集


- SciCite is a 3-way citation-intent classification benchmark introduced by Cohan et al. [14]. Each example consists of a citation context from a scientific paper, labeled by the citation's rhetorical role, such as background, method, or result. The task tests whether a model can infer why one paper cites another from the local scientific context.
- SciCite 是由 Cohan 等人 [14] 提出的三分类引文意图基准。每个样本包含一篇科学论文中的引文上下文，并根据引文的修辞作用（如背景、方法或结果）进行标注。该任务旨在测试模型能否根据局部科学语境推断出一篇论文引用另一篇论文的原因。


- FiNER-139 is a financial numeric entity recognition benchmark introduced by Loukas et al. [29]. It consists of word-level annotations from financial filings with 139 fine-grained XBRL entity types, making it substantially more fine-grained than standard sentence-level classification tasks. The benchmark tests whether a model can identify and classify numeric financial entities from context.
- FiNER-139 是由 Loukas 等人 [29] 提出的金融数值实体识别基准。它由来自金融备案文件的词级标注组成，包含 139 种细粒度 XBRL 实体类型，其粒度远高于标准的句子级分类任务。该基准旨在测试模型能否从上下文中识别并分类数值型金融实体。


- Amazon Reviews is the English portion of the Multilingual Amazon Reviews Corpus introduced by Keung et al. [22]. In our setting, it is used as a 5-way review rating prediction task, where the label corresponds to the review's star rating. This benchmark evaluates general-domain sentiment and rating prediction from product review text.
- Amazon Reviews 是 Keung 等人 [22] 提出的多语言亚马逊评论语料库的英文部分。在我们的设置中，它被用作五分类评论评分预测任务，标签对应评论的星级。该基准用于评估基于产品评论文本的通用领域情感与评分预测能力。


- Financial PhraseBank is a 3-way financial sentiment benchmark introduced by Malo et al. [32]. It consists of sentences from financial news and related economic text labeled as positive, neutral, or negative with respect to market sentiment. The task evaluates domain-specific sentiment classification in finance.
- Financial PhraseBank 是由 Malo 等人 [32] 提出的三分类金融情感基准。它由来自金融新闻及相关经济文本的句子组成，并根据市场情绪标注为正面、中性或负面。该任务旨在评估金融领域的特定情感分类能力。


- GoEmotions is a fine-grained emotion classification benchmark introduced by Demszky et al. [15]. It contains English Reddit comments annotated with 27 emotion categories plus a neutral category, and is commonly treated as a 28-way classification task. The benchmark tests nuanced affect recognition beyond coarse positive-negative sentiment.
- GoEmotions 是由 Demszky 等人 [15] 引入的细粒度情感分类基准。它包含标注了 27 种情感类别外加一个中性类别的英文 Reddit 评论，通常被视为一个 28 分类任务。该基准测试的是超越粗略正负情绪的细微情感识别能力。


- Banking77 is a fine-grained intent classification benchmark introduced by Casanueva et al. [11]. It contains online banking user utterances labeled with 77 intents, covering a wide range of customer service requests. The task evaluates single-domain intent detection with a large label space.
- Banking77 是由卡萨努埃瓦（Casanueva）等人 [11] 引入的细粒度意图分类基准。它包含标有 77 种意图的网上银行用户话语，涵盖了广泛的客户服务请求。该任务评估具有大标签空间的单领域意图检测。


- AG News is a 4-way news topic classification benchmark commonly associated with the text classification setup of Zhang et al. [60]. Examples are labeled with broad news categories such as world, sports, business, and science/technology. It is a standard general-domain benchmark for topic classification.
- AG新闻是一个四分类新闻主题分类基准，通常与Zhang等人的文本分类设置相关 [60]。示例被标记为广泛的新闻类别，如世界、体育、商业和科学/技术。它是主题分类的标准通用领域基准。


- SciTail is a science-domain textual entailment benchmark in which the task is to predict whether a hypothesis is entailed by a premise sentence in a science-focused inference setting [24].
- SciTail是一个科学领域的文本蕴含基准，其任务是在以科学为重点的推理环境中预测一个假设是否由一个前提句子所蕴含 [24]。


- TweetEval (Hate) is the hate-speech subset of the TweetEval benchmark introduced by Barbieri et al. [7]. It is a binary tweet classification task for detecting hateful versus non-hateful content within a unified social-media evaluation suite. This benchmark tests robust classification in noisy, short-form social media text.
- TweetEval (Hate) 是由 Barbieri 等人 [7] 引入的 TweetEval 基准测试中的仇恨言论子集。它是一项二元推文分类任务，旨在统一的社交媒体评估套件中检测仇恨与非仇恨内容。该基准测试旨在检验在嘈杂、短文本社交媒体内容中的稳健分类能力。


### C.2 Math Retrieval Corpus
### C.2 数学检索语料库


Table 10 lists the datasets composing the retrieval corpus used in Section 4.2 The raw sources contain more problems than the final corpus; several filtering steps were applied before merging. NuminaMath-1.5 was filtered to competition-math subsets (AMC/AIME, olympiad references, number theory, inequalities, and related sources), discarding lower-quality web-scraped entries. OpenMathReasoning was deduplicated to one solution per problem (retaining the solution with the highest pass rate on an independent verifier), and problems whose source matched any evaluation benchmark family (IMO, AIME, HMMT, SMT, USAMO, Putnam) were removed before deduplication. The entire corpus was then decontaminated against all evaluation benchmarks and the search set used during harness search, using exact prefix matching followed by fuzzy Jaccard similarity (threshold 0.8); any corpus problem matching an eval problem under either criterion was discarded. Solutions from OpenMathReasoning and DeepMath are truncated to 5,000 characters to limit retrieval context length. At runtime, the selected harness further restricts retrieval to entries with non-empty solutions shorter than 4,000 characters. Retrieved solutions are truncated again to 3,000 characters when inserted into the prompt. For the geometry route, the harness also constructs a separate hard-reference index from NuminaMath problems with difficulty greater than 6.
表10列出了第4.2节所用检索语料库的组成数据集。原始来源包含的问题多于最终语料库；在合并前应用了若干过滤步骤。NuminaMath-1.5被过滤为竞赛数学子集（AMC/AIME、奥数参考、数论、不等式及相关来源），剔除了质量较低的网络抓取条目。OpenMathReasoning经过去重，每个问题仅保留一个解（保留在独立验证器上通过率最高的解），且在去重前移除了来源与任何评估基准系列（IMO、AIME、HMMT、SMT、USAMO、Putnam）匹配的问题。随后，整个语料库针对所有评估基准及搜索过程中使用的搜索集进行了去污染处理，采用精确前缀匹配及模糊Jaccard相似度（阈值0.8）；任何在任一准则下与评估问题匹配的语料库问题均被剔除。来自OpenMathReasoning和DeepMath的解被截断至5,000字符，以限制检索上下文长度。运行时，所选harness进一步将检索限制为非空且短于4,000字符的条目。检索到的解在插入提示词时再次被截断至3,000字符。对于几何路径，harness还利用难度大于6的NuminaMath问题构建了一个独立的硬参考索引。


### C.3 Math IMO-level Test Set
### C.3 数学IMO级别测试集


The main text aggregates results over 200 IMO-level problems drawn from IMO-AnswerBench, IMO-ProofBench, ArXivMath December 2025, and ArXivMath January 2026. The 200-problem evaluation set consists of a stratified 100-problem subset of IMO-AnswerBench, together with all problems from the other three benchmarks. This per-benchmark breakdown is useful because the four datasets mix answer-style, proof, and research-style problems, which are aggregated together in the main paper for brevity. When included, the table in this section should report each benchmark separately for both Base and Meta-Harness across the five held-out models.
正文汇总了来自IMO-AnswerBench、IMO-ProofBench、ArXivMath 2025年12月版及ArXivMath 2026年1月版的200道IMO级别问题的结果。该200题评估集由IMO-AnswerBench的分层100题子集，以及其他三个基准测试中的所有问题组成。这种按基准测试的细分很有必要，因为这四个数据集混合了答案型、证明型和研究型问题，为简洁起见在正文中进行了合并。若包含本节表格，应针对五个留出模型，分别报告Base和Meta-Harness在各基准测试上的表现。


<table><tr><td>Dataset</td><td>Problems</td><td>Sol. Len</td><td>Proof</td></tr><tr><td>OpenMathReasoning</td><td>281,743</td><td>5,000†</td><td>34%</td></tr><tr><td>DeepMath-103K</td><td>103,021</td><td>5,000†</td><td>0%</td></tr><tr><td>NuminaMath-1.5</td><td>129,520</td><td>1,376</td><td>13%</td></tr><tr><td>PolyMath</td><td>11,083</td><td>363</td><td>0%</td></tr><tr><td>Omni-MATH</td><td>4,289</td><td>829</td><td>0%</td></tr><tr><td>FineProofs-SFT</td><td>4,275</td><td>3,977</td><td>100%</td></tr><tr><td>AIME 1983–2024</td><td>933</td><td>-</td><td>0%</td></tr><tr><td>Putnam-AXIOM</td><td>492</td><td>888</td><td>100%</td></tr><tr><td>Total</td><td>535,356</td><td>5,000†</td><td>22%</td></tr></table>
<table><tbody><tr><td>数据集</td><td>题目数量</td><td>解题长度</td><td>证明</td></tr><tr><td>OpenMathReasoning</td><td>281,743</td><td>5,000†</td><td>34%</td></tr><tr><td>DeepMath-103K</td><td>103,021</td><td>5,000†</td><td>0%</td></tr><tr><td>NuminaMath-1.5</td><td>129,520</td><td>1,376</td><td>13%</td></tr><tr><td>PolyMath</td><td>11,083</td><td>363</td><td>0%</td></tr><tr><td>Omni-MATH</td><td>4,289</td><td>829</td><td>0%</td></tr><tr><td>FineProofs-SFT</td><td>4,275</td><td>3,977</td><td>100%</td></tr><tr><td>AIME 1983–2024</td><td>933</td><td>-</td><td>0%</td></tr><tr><td>Putnam-AXIOM</td><td>492</td><td>888</td><td>100%</td></tr><tr><td>总计</td><td>535,356</td><td>5,000†</td><td>22%</td></tr></tbody></table>


${}^{ \dagger  }$ Truncated at 5,000 characters; actual solutions are longer.
${}^{ \dagger  }$ 截断至 5,000 字符；实际解答更长。


Table 10: Datasets in the math retrieval corpus (535K problems total). Sol. Len is the median solution length in characters. Proof indicates whether the dataset contains proof-type problems (by answer or problem type field).
表 10：数学检索语料库中的数据集（共 53.5 万道题目）。Sol. Len 为解答长度的中位数（以字符计）。Proof 指示该数据集是否包含证明类题目（根据答案或题目类型字段判定）。


<table><tr><td>Dataset</td><td>Problems</td></tr><tr><td>IMO-AnswerBench</td><td>100</td></tr><tr><td>IMO-ProofBench</td><td>60</td></tr><tr><td>ArXivMath Dec. 2025</td><td>17</td></tr><tr><td>ArXivMath Jan. 2026</td><td>23</td></tr><tr><td>Total</td><td>200</td></tr></table>
<table><tbody><tr><td>数据集</td><td>题目数量</td></tr><tr><td>IMO-AnswerBench</td><td>100</td></tr><tr><td>IMO-ProofBench</td><td>60</td></tr><tr><td>ArXivMath 2025年12月</td><td>17</td></tr><tr><td>ArXivMath 2026年1月</td><td>23</td></tr><tr><td>总计</td><td>200</td></tr></tbody></table>


Table 11: Breakdown of the 200-problem IMO-level evaluation set.
表 11：200 道 IMO 级别评估题目的细分。


## D Practical Implementation Tips
## D 实践实施建议


Meta-Harness is largely domain-agnostic: we expect it to apply in any setting where a language model is wrapped by a task-specific harness. Applying it in a new domain, however, requires operating in a relatively new regime of LLM-assisted coding, where the proposer conditions on long-horizon histories of prior runs and writes programs whose effects may only become visible many steps later. In getting this workflow to work reliably, we found a small set of practical choices that mattered consistently across the three domains studied in this paper. The guidelines below are not themselves scientific claims about the method; they are engineering lessons from building and running the system, which we hope will make it easier for future work to apply Meta-Harness in other domains.
Meta-Harness 在很大程度上与领域无关：我们预期它适用于任何由特定任务工具包（harness）封装语言模型的场景。然而，在新的领域应用它，需要进入大模型辅助编程的一个相对较新的范式，即提案者（proposer）基于长周期的历史运行记录进行条件推理，并编写其效果可能在多步之后才显现的程序。为了使该工作流可靠运行，我们总结了一套在本文研究的三个领域中始终有效的实践选择。以下指南本身并非关于该方法的科学论断；它们是从构建和运行系统过程中获得的工程经验，我们希望这些经验能使未来的工作更容易将 Meta-Harness 应用于其他领域。


- Write a good skill. The skill text is the primary interface for steering the search, and its quality is the strongest lever on whether the loop works. The proposer receives a natural-language skill [5] that defines its role, the directory layout, CLI commands, and output format. In practice, the skill should constrain outputs and safety-relevant behavior, not the proposer's diagnosis procedure: it should specify what is forbidden, what artifacts to produce, and what objectives to optimize, while leaving the model free to inspect scores, traces, and prior code as needed. Our intuition from inspecting logs from Meta-Harness runs is that after enough iterations, the accumulated traces often shape the proposer's behavior more than the skill itself. In our experience, iterating on the skill text had a larger effect on search quality than changing iteration count or population size. Expect to run a few short evolution runs (3-5 iterations each) specifically to debug and refine the skill before committing to a full run.
- 编写一个优质技能。技能文本是引导搜索的主要界面，其质量对搜索循环能否有效运行起着关键作用。提议者会收到一份自然语言形式的技能说明 [5]，其中定义了其角色、目录布局、命令行界面（CLI）命令以及输出格式。实际上，技能应限制输出和与安全相关的行为，而非提议者的诊断流程：它应明确禁止的事项、要生成的工件以及要优化的目标，同时让模型能够根据需要自由检查分数、跟踪信息和先前的代码。我们从 Meta - Harness 运行日志中观察到的直觉是，经过足够多的迭代后，累积的跟踪信息往往比技能本身更能影响提议者的行为。根据我们的经验，对技能文本进行迭代比改变迭代次数或种群规模对搜索质量的影响更大。在进行完整运行之前，预计要专门进行几次短时间的进化运行（每次 3 - 5 次迭代），以调试和完善技能。


- Start with a baseline harness and a search set that is hard for it. Write a simple baseline (e.g., few-shot prompting), then construct the search set by either filtering for examples that the baseline gets wrong or selecting a diverse subset of difficult instances. The search has little to optimize if the baseline already saturates the evaluation. Keep the search set small enough for roughly 50 full evaluations per run (50-100 examples in our classification experiments, 88 problems for math retrieval); a fast, discriminative eval is more valuable than a large one.
- 从基准工具包和一个对其而言较难的搜索集开始。编写一个简单的基准（例如少样本提示），然后通过筛选基准出错的示例或选择多样化的困难实例子集来构建搜索集。如果基准已经饱和了评估，搜索就几乎没有优化空间。保持搜索集足够小，以保证每次运行大约进行 50 次完整评估（我们的分类实验中为 50-100 个示例，数学检索为 88 个问题）；快速且具有区分度的评估比大规模评估更有价值。


- Log everything in a format that is easy to navigate. Evaluation code should write code, scores, and execution traces in a form that the proposer can query reliably. In practice, this means using machine-readable formats such as JSON, organizing artifacts hierarchically, choosing reasonable and consistent file names, and adopting naming schemes that make simple tools such as regex search work well.
- 以易于导航的格式记录所有内容。评估代码应以提案者能够可靠查询的形式写入代码、分数和执行追踪记录。在实践中，这意味着使用机器可读格式（如 JSON）、分层组织工件、选择合理且一致的文件名，并采用使正则表达式搜索等简单工具能够良好工作的命名方案。


- Make logs queryable through a small CLI (optional, but helpful). Each harness gets a directory containing source code, scores, and execution traces, but as the history grows, raw filesystem access alone becomes cumbersome. A short CLI that lists the Pareto frontier, shows top- $k$ harnesses,and diffs code and results between pairs of runs can make the experience store much easier to use, and querying such CLIs is closely aligned with the workflows on which coding agents are trained. If relevant offline experience exists (rollouts from other models, solved problem corpora, relevant papers), converting it into the same directory structure can also help warm-start exploration and ground new ideas. This layer helps the proposer save tokens it may have wasted on navigation.
- 通过小型命令行界面（CLI）使日志可查询（可选，但很有帮助）。每个测试套件（harness）都有一个包含源代码、分数和执行轨迹的目录，但随着历史记录的增加，仅靠原始文件系统访问会变得繁琐。一个能够列出帕累托前沿、展示前 $k$ 个测试套件，并对比不同运行之间代码和结果差异的简易 CLI，可以显著提升使用体验；且查询此类 CLI 的方式与编码智能体所训练的工作流高度契合。如果存在相关的离线经验（来自其他模型的展开、已解决的问题语料库、相关论文），将其转换为相同的目录结构也有助于冷启动探索并为新想法提供基础。这一层级有助于提议者节省可能浪费在导航上的 token。


- Lightweight validation before expensive benchmarks. Write a small validation test that imports the module, instantiates the class, and calls both methods on a tiny set of examples. Harnesses proposed during the search should pass this test before being fully evaluated. A simple test script can catch most malformed or nonfunctional candidates in seconds and keep the cost of failures near zero.
- 在昂贵的基准测试前进行轻量级验证。编写一个小型验证测试，导入模块、实例化类，并在极少数示例上调用这两个方法。搜索过程中提出的工具包在进行全面评估前应通过此测试。一个简单的测试脚本可以在几秒钟内捕获大多数格式错误或功能异常的候选者，并将失败成本保持在接近零的水平。


- Automate evaluation outside the proposer. Running evals is simple enough that it is not worth making the proposer do it. A separate harness should score candidates and write results to the filesystem.
- 在提案者之外实现评估自动化。运行评估非常简单，不值得让提案者来做。应使用单独的工具包来对候选者进行评分并将结果写入文件系统。


## E Extended Related Work
## E 扩展相关工作


This appendix expands the brief discussion in Section 2 and situates Meta-Harness relative to several neighboring lines of work that we could not cover in detail in the main text. A recurring distinction is that Meta- Harness optimizes executable harness implementations and provides the proposer with selective access to prior code, scores, and execution traces via the filesystem.
本附录扩展了第 2 节中的简要讨论，并将 Meta-Harness 置于我们在正文中无法详细涵盖的几个相关研究领域中进行定位。一个反复出现的区别在于，Meta-Harness 优化了可执行的工具包实现，并通过文件系统为提案者提供对先前代码、分数和执行追踪记录的选择性访问。


AlphaEvolve / OpenEvolve. AlphaEvolve [35] and OpenEvolve [43] evolve code via LLM-guided mutations with structured feedback: the proposer receives a program database with scalar scores (4-22K tokens per step; Table 1) and applies fixed mutation strategies to tournament-selected parents. These methods are designed for algorithm discovery and optimization (mathematical conjectures, scheduling heuristics, hardware kernels), where the search target is a single stateless function with a clean scalar objective, and mutations are local. Harness engineering is a different regime: harnesses are stateful programs that accumulate experience across many examples, and a single design choice (e.g., what to store in memory) can cascade through an entire evaluation sequence. Meta-Harness addresses this by giving an unstructured coding agent full filesystem access, letting it selectively read any prior candidate's source code, execution traces, and scores.
AlphaEvolve / OpenEvolve。AlphaEvolve [35] 和 OpenEvolve [43] 通过大模型引导的变异与结构化反馈来演化代码：提议者接收一个带有标量分数的程序数据库（每步 4-22K token；表 1），并对锦标赛选出的父代应用固定的变异策略。这些方法专为算法发现与优化（数学猜想、调度启发式算法、硬件内核）而设计，其搜索目标是具有明确标量目标的单一无状态函数，且变异是局部的。Harness 工程则处于不同的范畴：Harness 是在多个示例中积累经验的有状态程序，单一的设计选择（例如在内存中存储什么）可能会在整个评估序列中产生连锁反应。Meta-Harness 通过赋予非结构化编码智能体完整的文件系统访问权限，使其能够选择性地读取任何先前候选者的源代码、执行轨迹和分数，从而解决了这一问题。


GEPA. GEPA [1] is the closest text optimizer in terms of feedback richness, providing rollout traces per candidate. It is designed for prompt optimization on tasks with short feedback loops (math problems, instruction-following, code optimization), where each rollout is a single LLM call or a short pipeline. In this regime, per-candidate reflection works well: one prompt, one answer, one score. Harness engineering requires reasoning across many examples and many candidates simultaneously: understanding why a retrieval strategy works for one class of problems but degrades on another requires comparing execution traces across the full population. GEPA operates on one candidate at a time (2-8K tokens per step; Table 1), with a fixed critique format that must anticipate what information is relevant. Meta-Harness gives the proposer access to all prior candidates simultaneously and lets the agent decide what to examine.
GEPA。就反馈丰富度而言，GEPA [1] 是最接近的文本优化器，它为每个候选方案提供展开轨迹。它专为短反馈循环任务（数学问题、指令遵循、代码优化）的提示词优化而设计，其中每次展开仅为单次 LLM 调用或短流水线。在此机制下，针对单个候选方案的反射效果良好：一个提示词、一个答案、一个评分。而 Harness 工程需要同时对大量示例和候选方案进行推理：要理解为何某种检索策略对一类问题有效却在另一类问题上表现下降，就需要比较整个群体中的执行轨迹。GEPA 每次仅处理一个候选方案（每步 2-8K token；表 1），且采用固定的评估格式，必须预先确定哪些信息是相关的。Meta-Harness 则允许提案者同时访问所有先前的候选方案，并让智能体自行决定检查哪些内容。


Prompt orchestration frameworks. Several systems provide structured abstractions for composing multi-stage LLM programs. LMQL [8], LangChain [13], and DSPy [23] make prompt engineering more systematic by providing higher-level interfaces for prompt templates, control flow, and modular LLM pipelines. These frameworks help developers specify and organize LLM programs, but they still typically require manual design of retrieval policies, memory updates, and orchestration logic. Meta-Harness operates at a different level: it searches over the implementation of these policies in executable code, treating the harness itself as the optimization target.
提示词编排框架。若干系统为构建多阶段大模型程序提供了结构化抽象。LMQL [8]、LangChain [13] 和 DSPy [23] 通过为提示词模板、控制流和模块化大模型流水线提供更高层级的接口，使提示词工程更加系统化。这些框架帮助开发者指定并组织大模型程序，但通常仍需手动设计检索策略、记忆更新和编排逻辑。Meta-Harness 则处于不同层面：它在可执行代码中搜索这些策略的实现，将工具链本身视为优化目标。