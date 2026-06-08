# Atomic-Probe Governance for Skill Updates in Compositional Robot Policies*
# 面向组合机器人策略技能更新的原子探针治理*


Xue Qin
秦雪


School of Software
软件学院


Harbin Institute of Technology
哈尔滨工业大学


Harbin, China
中国哈尔滨


qinxue@me.com



Simin Luan
Luan Simin


School of Computer Science and
计算机科学与


Technology
技术学院


Harbin Institute of Technology
哈尔滨工业大学


Harbin, China
中国哈尔滨


luansiminiot@gmail.com



John See
John See


School of Mathematical and
数学与


Computer Sciences
计算机科学学院


Heriot-Watt University, Malaysia
赫瑞-瓦特大学马来西亚校区


Campus
校区


Putrajaya 62200, Malaysia
马来西亚普特拉贾亚 62200


J.See@hw.ac.uk



Zeyd Boukhers
泽德·布赫尔斯


Fraunhofer Institute for Applied
弗劳恩霍夫应用


Information Technology
信息技术研究所


Sankt Augustin, Germany
德国圣奥古斯丁


zeyd.boukhers@fit.fraunhofer.de



Cong Yang*
孔·康*


School of Future Science and
未来科学与


Engineering
工程学院


Soochow University
苏州大学


Suzhou, China
中国苏州


cong.yang@suda.edu.cn



Zhijun Li*
志军·李*


School of Computer Science and Technology
计算机科学与技术学院


Harbin Institute of Technology
哈尔滨工业大学


Harbin, China
中国哈尔滨


lizhijunos@hit.edu.cn



Abstract: Skill libraries in deployed robotic systems are continually updated through fine-tuning, fresh demonstrations, or domain adaptation, yet existing typed-composition methods (BLADE, SymSkill, Generative Skill Chaining) treat the library as frozen at test time and offer no analysis of how composition outcomes change when an underlying skill is replaced. We introduce a paired-sampling cross-version swap protocol on standard robosuite manipulation tasks and characterize this previously unstudied dimension of compositional skill learning. On a representative dual-arm peg-in-hole task we discover a dominant-skill effect: a single ECM in the candidate set achieves 86.7% atomic success rate while every other ECM is at or below 26.7%, and whether this dominant ECM is included in a composition shifts the success rate by up to +50 percentage points (subset-swap group mean; the single-phase paired-swap matrix on REACH contains gains up to +60pp and losses up to -37pp, see Table 3). We characterize the boundary of the effect on a simpler pick task where all atomic policies saturate at 100% and the effect is consequently undefined. Across three independent tasks we further find that off-policy behavioral distance metrics fail to identify the dominant ECM, ruling out the most natural cheap predictor. Building on these observations, we propose an atomic-quality probe and a Hybrid Selector that combines per-skill probes (zero per-decision cost) with selective composition revalidation (full cost). On T6, where the success-rate oracle is well-defined, the atomic-only probe matches the oracle on 64.6% of 48 paired skill-update events versus full-revalidation's 87.5% - a 22.9pp gap that is statistically significant (McNemar exact-binomial $p = {0.013}$ ; cluster-permutation $p = {0.018}$ respecting ECM-level dependence) - and a Hybrid Selector with margin $m = {10}$ closes most of that gap (75.0% oracle match)
摘要：在部署的机器人系统中，技能库会通过微调、新的演示或领域适配不断更新，但现有的有类型组合方法（BLADE、SymSkill、生成式技能链式）在测试时将库视为冻结状态，未对当底层技能被替换时组合结果如何变化进行分析。我们在标准 robosuite 操作任务上提出一种成对采样的跨版本置换协议，并刻画这一先前未被研究的复合技能学习维度。在一个典型的双机械臂“插销入孔”任务中，我们发现了显著的“主导技能效应”：候选集合中单个 ECM 可达到 86.7% 的原子成功率，而其他每个 ECM 的原子成功率都在 26.7% 及以下；是否在组合中包含该主导 ECM，会使成功率最多变化 +50 个百分点（子集置换组的平均值；REACH 上单阶段成对置换矩阵显示增益最高可达 +60pp、损失最高可达 -37pp，见表 3）。我们还在更简单的“取放”任务上刻画该效应的边界：所有原子策略都在 100% 饱和，因此该效应相应地未定义。在三项相互独立的任务中，我们进一步发现，离策略行为距离度量无法识别主导 ECM，从而排除了最自然的廉价预测器。基于这些观察，我们提出一种“原子质量探针”，以及一种混合选择器：将每个技能的探针（零的每次决策成本）与选择性的组合再验证（完整成本）结合起来。在 T6 上，当成功率的真值预言机定义良好时，仅原子探针在 48 个成对的技能更新事件中有 64.6% 与预言机一致，而全量再验证为 87.5%——两者差距为 22.9 个百分点，且具有统计显著性（McNemar 精确二项 $p = {0.013}$；遵循 ECM 级依赖的聚类置换 $p = {0.018}$）——混合选择器的裕度 $m = {10}$ 使该差距大部分得以弥合（预言机匹配 75.0%），


---



*Corresponding authors.
*通讯作者。


---



at 45.8% of full-revalidation cost. The atomic-quality probe is, to our knowledge, the first principled, deployment-ready primitive for skill-update governance in compositional robot policies, sitting at the intersection of typed skill composition and off-policy policy selection. ${}^{2}$
相当于仅原子探针占用全量再验证成本的 45.8%。据我们所知，原子质量探针是首个具有原则性、可部署的技能更新治理基础原语，处于“有类型技能组合”和“离策略策略选择”的交汇处。${}^{2}$


Keywords: Skill Composition; Continual Learning; Skill-Update Governance
关键词：技能组合；持续学习；技能更新治理


## 1 Introduction
## 1 引言


Generalist policies designed to be fine-tuned $\left\lbrack  {1,2,3}\right\rbrack$ make post-deployment skill updates a routine event, yet the typed- composition literature $\left\lbrack  {4,5,6}\right\rbrack$ treats the skill library as fixed at test time, and the adjacent "open-ended skill library" line [7, 8, 9] grows the library without studying what happens when an existing skill is replaced. We ask: when an existing skill in a compositional library is replaced by an independently-trained alternative, what happens to the compositions that depend on it? We introduce a paired-sampling cross-version swap protocol that varies the version of one or more phase ECMs while holding all other factors fixed.
为可进行微调 $\left\lbrack  {1,2,3}\right\rbrack$ 而设计的通用策略，使部署后的技能更新成为常规事件；然而，类型化的组合文献 $\left\lbrack  {4,5,6}\right\rbrack$ 将技能库视为在测试时固定不变，而相邻的“开放式技能库”路线 [7, 8, 9] 在不研究当现有技能被替换时会发生什么的情况下扩充库。我们提问：当组合技能库中的某个既有技能被替换为一个独立训练得到的替代方案时，依赖它的那些组合会怎样？我们提出一种配对采样的跨版本置换协议：在保持所有其他因素不变的前提下，改变一个或多个阶段 ECM 的版本。


On robosuite manipulation, the population-mean swap effect is statistically zero, yet cell-level variance inflates 1.14-2.54 × across the four phases. This apparent null masks a dominant-skill effect: a single ECM drives composition outcomes on the peg-in-hole task with swap-induced shifts up to 50pp; the effect is by construction undefined on a saturated single-arm pick task; and behavioral-distance metrics fail to identify the dominant ECM on all three tasks tested. We then propose an atomic-quality probe and a Hybrid Selector combining atomic probes with selective composition revalidation: on 144 update decisions, atomic-only is within 3pp of full revalidation at zero per-decision cost, under a mixed-oracle caveat we analyze in detail. Our contributions are: (i) the paired-sampling cross-version swap protocol; (ii) empirical characterization of the dominant-skill effect and its saturation boundary; (iii) the atomic-probe Hybrid Selector for skill-update governance, with cost-quality trade-off on 144 decisions across three tasks.
在 robosuite 的操作任务中，群体均值置换效应在统计上为零，但在四个阶段上的“细胞级”方差会膨胀 1.14-2.54 ×。这种表面上的空效应掩盖了一个主导技能效应：在穿孔入榫任务中，单个 ECM 就能驱动组合结果，且置换引发的偏移最高可达 50pp；在已饱和的单臂抓取任务上，该效应在构造上是未定义的；并且，行为距离度量无法在我们测试的三个任务中识别出主导 ECM。随后，我们提出一种原子质量探针，以及一个混合选择器：将原子探针与有选择的组合再验证结合起来。在 144 次更新决策中，仅原子方案与在每次决策零额外成本下的完整再验证结果相差不超过 3pp；在我们将详细分析的“混合型预言机”前提下，我们的发现如此。我们的贡献包括：（i）配对采样的跨版本置换协议；（ii）对主导技能效应及其饱和边界的实证刻画；（iii）用于技能更新治理的原子探针混合选择器，并在三个任务、共 144 次决策中实现成本-质量权衡。


## 2 Related Work
## 2 相关工作


Typed composition with learned pre/post-conditions. A line of recent work pairs learned skills with explicit symbolic interfaces. BLADE [4] extracts each high-level action's pre/post-conditions from language-annotated demonstrations via an LLM and pairs them with neural controllers; SymSkill [5] jointly learns predicates, operators, and skills from unsegmented demonstrations with real-time symbolic recovery. Generative Skill Chaining [6] models the joint distribution of (precondition, parameters, effect) per skill via a diffusion model and is the closest neighbor to a stability-aware view in the present work, although it does not study post-deployment updates. All such methods study composition under the assumption that the constituent skills are static after construction.
带有已学习的前/后置条件的类型化组合。近期一条研究线将学习到的技能与明确的符号接口配对。BLADE [4] 通过 LLM 从带语言标注的示范中提取每个高层动作的前/后置条件，并将其与神经控制器配对；SymSkill [5] 在未分段的示范中联合学习谓词、算子和技能，并通过实时符号恢复实现。生成式技能链式编排 [6] 使用扩散模型来建模每个技能的 (前置条件, 参数, 效果) 联合分布，是目前这项工作中最接近一种“对稳定性敏感”的视角的邻近工作，尽管它并不研究部署后的更新。所有这些方法都在一个假设下研究组合：构建完成后组成技能是静态的。


Skill libraries and skill chaining. A complementary line treats robotic competence as a library of reusable modules. Voyager [7], BOSS [8], and LOTUS [9] are append-only: they grow the library at deployment without removing or updating existing skills. SayCan [10] and Code-as-Policies [11] pair language-model planning with primitive skill calls. The most direct neighbours of our atomic-quality probe sit in the skill-chaining stability thread: T-STAR [12] regularizes terminal states at training time so adjacent skills agree on hand-off distributions; Sequential Dexterity [13] gates dexterous policy chaining with a learned transition-feasibility function; and Value-Informed Skill Chaining [14] gates skill transitions on a state-value function. All three operate at the chain-level (preventing bad transitions) rather than the update-level (deciding whether to admit a new candidate skill into the library), which is the question we ask. None of this literature formalizes the question of what happens to existing compositions when one of the constituent skills is later updated.
技能库与技能链式编排。另一条互补的思路将机器人能力视为可复用模块的库。Voyager [7]、BOSS [8] 和 LOTUS [9] 都是追加式：在部署时扩展库，而不移除或更新已有技能。SayCan [10] 和 Code-as-Policies [11] 将语言模型规划与原语技能调用结合。我们这项“原子质量探针”的最直接邻近工作，出现在技能链式编排稳定性这一脉：T-STAR [12] 在训练时对终端状态进行正则化，以保证相邻技能在交接分布上达成一致；Sequential Dexterity [13] 通过学习到的过渡可行性函数来门控灵巧策略的链式编排；以及 Value-Informed Skill Chaining [14] 通过状态价值函数对技能转移进行门控。三者都在链级别工作（阻止不良转移），而不是更新级别工作（决定是否将一个新的候选技能接纳进库），这也是我们要回答的问题。现有文献都没有形式化：当构成技能之一在之后被更新时，现有组合会发生什么。


---



${}^{2}$ Code and evaluation data: https://github.com/s20sc/atomic-probe-governance.Simulation framework (training and environment library): https://github.com/s20sc/ capability-evolution.
${}^{2}$ 代码与评估数据：https://github.com/s20sc/atomic-probe-governance。仿真框架（训练与环境库）：https://github.com/s20sc/ capability-evolution。


---



Off-policy policy selection. The atomic-quality probe is structurally a per-skill off-policy evaluator, and the Hybrid Selector is structurally an active offline-policy-selection procedure: cheap surrogate estimates warm-start a budgeted online evaluation. Benchmarks for deep OPE [15] and Active Offline Policy Selection [16] establish the surrogate-vs-online trade-off in the single-policy regime; to our knowledge no prior work brings OPE rigor into typed compositional skill libraries (citation-network evidence in section F), which is the contribution we make below.
离策略的策略选择。“原子质量探针”在结构上是一个面向单个技能的离策略评估器，而 Hybrid Selector 在结构上是一种主动的离线策略选择流程：廉价的代理估计用于热启动对预算内在线评估。深度 OPE [15] 与主动离线策略选择 [16] 的基准工作确立了在单策略情形下代理与在线之间的权衡。据我们所知，先前没有工作将 OPE 的严谨性引入到类型化的组合技能库（在 F 节中给出引文网络证据），而我们下面做的正是这一贡献。


Generalist VLA policies and the post-deployment update setting, compositional benchmarks (CompoSuite, LIBERO, CALVIN), continual learning of robot policies, and our statistical-reporting choices are surveyed in section F.
通用型 VLA 策略、部署后的更新设置、组合基准（CompoSuite、LIBERO、CALVIN）、机器人策略的持续学习，以及我们在统计报告方面的选择，均在 F 节中概述。


## 3 Setup
## 3 设置


### 3.1 Compositional Skill Execution
### 3.1 组合式技能执行


Following the spirit of Liu et al. [4], Shao et al. [5], we represent a long-horizon manipulation task as an ordered sequence of phases $\Pi  = \left( {{\pi }_{1},\ldots ,{\pi }_{K}}\right)$ ,each phase served by a phase-specific neural controller (an Embodied Capability Module, ECM), an instance of the temporally-abstract option [17]. At runtime,ECM ${c}_{k}$ is invoked on entry to phase ${\pi }_{k}$ and is replaced by ${c}_{k + 1}$ on a phase-termination predicate. Whereas BLADE and SymSkill extract task-specific phase schedules from demonstrations or LLMs,we use a fixed $K = 4$ decomposition $\Pi  =$ (REACH, GRASP, LIFT, PLACE) for experimental tractability. A composition is an assignment $C = \left( {{c}_{1},\ldots ,{c}_{K}}\right)$ with each ${c}_{k}$ drawn from a pool of candidate ECMs trained for phase ${\pi }_{k}$ .
秉承 Liu 等人 [4]、Shao 等人 [5] 的思路，我们将长时域操作任务表示为按顺序排列的阶段序列 $\Pi  = \left( {{\pi }_{1},\ldots ,{\pi }_{K}}\right)$；每个阶段由特定阶段的神经控制器（具身能力模块，ECM）以及时间抽象选项 [17] 的一个实例来实现。运行时，进入阶段 ${\pi }_{k}$ 时调用 ECM ${c}_{k}$，并在阶段终止谓词触发后由 ${c}_{k + 1}$ 取代。尽管 BLADE 和 SymSkill 会从演示或 LLM 中提取任务特定的阶段调度，我们为实验可操作性采用固定的 $K = 4$ 分解 $\Pi  =$（REACH、GRASP、LIFT、PLACE）。组合是对 $C = \left( {{c}_{1},\ldots ,{c}_{K}}\right)$ 的一种赋值，其中每个 ${c}_{k}$ 都从针对阶段 ${\pi }_{k}$ 训练得到的候选 ECM 集合中选取。


### 3.2 Evaluated Tasks
### 3.2 评估任务


We evaluate on six tasks from the robosuite manipulation benchmark [18]: T1_Pick, T2_Place, T3_Stack, T4_NutAssembly, T5_PickPlaceMulti, and T6_TwoArmPegInHole. Each task is decomposed into the same four phases (REACH, GRASP, LIFT, PLACE), with phase-specific ECMs trained from scratch using SAC.
我们在 robosuite 操作基准 [18] 的六个任务上进行评估：T1_Pick、T2_Place、T3_Stack、T4_NutAssembly、T5_PickPlaceMulti 和 T6_TwoArmPegInHole。每个任务都被分解为相同的四个阶段（REACH、GRASP、LIFT、PLACE），并使用 SAC 从头训练各阶段特定的 ECM。


The dominant-skill effect (section 4) is observable only on tasks where (a) atomic ECMs achieve non-trivial task success and (b) atomic quality varies across versions. Among our six tasks this holds only for T6_TwoArmPegInHole (dual-arm contact-rich peg insertion); on T1_Pick all atomic ECMs saturate at 100% (boundary case, section 5); on T2-T5 all atomic ECMs score 0% under our standard ${50}\mathrm{\;K} \times  {20}$ -iteration SAC schedule (discussed in section 9). We use T6 for the dominant-skill effect and T1 for the boundary; for the behavioral-distance refutation (section 6) and the algorithm benchmark (section 7) we additionally use T3 and T4, on which the oracle is necessarily defined on episode reward rather than task success, a methodological limitation we keep visible throughout.
主导技能效应（第4节）仅在以下任务中可见：（a）原子ECM实现非平凡的任务成功，且（b）不同版本间原子质量存在差异。在我们的六个任务中，只有T6_TwoArmPegInHole（双臂接触丰富的插销插入）满足这一点；在T1_Pick上，所有原子ECM都饱和到100%（边界情形，第5节）；在T2-T5上，按我们的标准${50}\mathrm{\;K} \times  {20}$轮SAC调度，所有原子ECM得分均为0%（第9节讨论）。我们用T6研究主导技能效应，用T1研究边界情况；对于行为距离反驳（第6节）和算法基准（第7节），我们还使用T3和T4，在这些任务上，oracle必须以每回合奖励而非任务成功来定义，这一方法学限制我们始终明确保留。


### 3.3 Skill Versions, Swap Protocol, and Probes
### 3.3 技能版本、置换协议与探针


To produce multiple versions of each phase ECM we train $S = 4$ independent SAC policies [19] per phase with seeds $\{ {42},7,{123},{2024}\}$ ,simulating a deployment in which the same target skill is independently re-trained from different demonstration batches, fine-tuning runs, or domain-adaptation cycles. A swap-set $\sigma  \subseteq  \Pi$ identifies the phases whose ECM is swapped from a primary version $p$ to an alternative $a$ :
为生成每个阶段 ECM 的多个版本，我们在每个阶段训练 $S = 4$ 个相互独立的 SAC 策略 [19] ，其随机种子为 $\{ {42},7,{123},{2024}\}$ ，以模拟一种部署情形：同一目标技能会从不同的示范批次、微调运行或领域自适应周期中被独立地重新训练。一个置换集合 $\sigma  \subseteq  \Pi$ 用于标识将 ECM 从主版本 $p$ 置换到备选版本 $a$ 的阶段：


$$
C\left( {p,a,\sigma }\right)  = {\left( {c}_{k}^{\left( a\right) }\text{ if }{\pi }_{k} \in  \sigma ,{c}_{k}^{\left( p\right) }\text{ otherwise }\right) }_{k = 1}^{K}.
$$



The diagonal cells $\sigma  = \varnothing$ and $\sigma  = \Pi$ are within-version baselines; the remaining ${2}^{K} - 2$ subsets characterize partial cross-version mixing. We use paired episode initial states (same $N = {30}$ init-state seeds per $\left( {p,a,\sigma }\right)$ ),enabling paired $t$ -tests on $\Delta$ success. Two probes characterize ECMs and compositions: the atomic-quality probe $q\left( c\right)  \mathrel{\text{ := }} \mathbb{P}\left( {\text{ success } \mid  c\text{ alone }}\right)$ evaluates an ECM as sole controller of an entire episode (reusable across all swap-sets involving $c$ ), and the composition probe $\mathbb{P}\left( {\text{ success } \mid  C}\right)$ uses the standard phase-scheduled execution. Both use binary task success; reward-based metrics are shaping-dependent and can produce qualitatively misleading rankings on tasks where atomic policies do not actually succeed.
对角单元 $\sigma  = \varnothing$ 和 $\sigma  = \Pi$ 是同版本基线；其余 ${2}^{K} - 2$ 子集刻画了部分跨版本混合。我们使用成对的情节初始状态（对 $\left( {p,a,\sigma }\right)$ 使用相同的 $N = {30}$ 初始种子），从而实现成对的 $t$ -tests 来评估 $\Delta$ 的成功。两个探针用于表征 ECM 及其组合：原子质量探针 $q\left( c\right)  \mathrel{\text{ := }} \mathbb{P}\left( {\text{ success } \mid  c\text{ alone }}\right)$ 将 ECM 评估为整个情节的唯一控制器（可复用于所有包含 $c$ 的置换集合），而组合探针 $\mathbb{P}\left( {\text{ success } \mid  C}\right)$ 使用标准的阶段定时执行。两者都采用二元任务成功；基于奖励的指标会因任务塑形而产生依赖，并可能在原子策略实际上并未成功的任务上给出质性上具有误导性的排序。


### 3.4 Compute and Reproducibility
### 3.4 计算与可复现性


ECMs are trained with SAC on a single NVIDIA RTX 5090 with the robosuite [18] simulator under the framework’s standard schedule. All evaluations use $N = {30}$ episodes per cell with paired init-state seeds in $\left\lbrack  {{10000},{10029}}\right\rbrack$ and seeds $\{ {42},7,{123},{2024}\}$ throughout,enabling McNemar/paired tests over the same episode pool. Per-task wall-time, hyperparameters, and reproducibility details are in section G.
ECM 在单台 NVIDIA RTX 5090 上使用 SAC 进行训练，仿真器采用 robosuite [18]，并在框架的标准训练计划下进行。所有评估均在每个 cell 使用 $N = {30}$ 个回合，初始状态种子在 $\left\lbrack  {{10000},{10029}}\right\rbrack$ 中成对设置，整个过程中种子为 $\{ {42},7,{123},{2024}\}$，从而可在同一回合池上进行 McNemar/配对检验。各任务的每任务实际耗时、超参数以及可复现性细节见第 G 节。


## 4 The Dominant-Skill Effect
## 4 主导技能效应


A naive expectation is that swapping one phase's ECM for an independently-trained sibling will perturb composition outcomes one way or the other. At the level of population mean this expectation fails: across the four phases of T6,paired $t$ -tests on the full $4 \times  4$ cross-seed swap matrices yield $p$ -values of 0.385 (REACH),0.616 (GRASP),0.790 (LIFT), $\approx  {1.0}$ (PLACE). No average effect: Holm-Bonferroni correction at $\alpha  = {0.05}$ leaves all four phases unrejected (adjusted $p \in  \left\lbrack  {{1.0},{1.0}}\right\rbrack$ ),so the population-mean null is robust under multiple-comparison control. Yet the cell-level variance of composition success rate inflates under swap by ${1.14} \times$ to 2.54 $\times$ across the four phases relative to the within-version diagonal,indicating structured rather than random perturbation. We show below that some swaps help while others hurt in a way that cancels in expectation: one specific ECM in the candidate set is disproportionately responsible for composition success, and the sign of any swap is determined by whether this dominant ECM enters or leaves the composition. All proportions in this paper are reported with ${95}\%$ bootstrap confidence intervals (B=5000 resamples) where space permits; full per-cell intervals are tabulated in sections B and C.
一种直觉期待是，把某一相位的 ECM 换成独立训练的同类，会以某种方式扰动组合结果。就种群的均值水平而言，这种期待不成立：在 T6 的四个相位上，针对完整 $4 \times  4$ 的跨种子交换矩阵进行 $t$ -检验，得到 $p$ -值为 0.385（REACH）、0.616（GRASP）、0.790（LIFT）、$\approx  {1.0}$（PLACE）。没有平均效应：在 $\alpha  = {0.05}$ 下的 Holm-Bonferroni 校正使得四个相位都未被拒绝（校正后 $p \in  \left\lbrack  {{1.0},{1.0}}\right\rbrack$ ），因此在多重比较控制下，种群均值的原假设依然稳健。然而，在交换后，组合成功率的单元层方差会被放大到 ${1.14} \times$ 为 2.54 $\times$（相对同一版本的对角线），这表明是有结构的扰动而非随机扰动。我们将在下文展示：有些交换会带来帮助，有些交换会带来损害，但它们在期望意义下相互抵消——候选集合中的某一个特定 ECM 对组合成功的贡献尤为突出，而任何交换的符号取决于该主导 ECM 是进入还是离开组合。本文中的所有比例均在有空间允许的情况下，以 ${95}\%$ bootstrap 置信区间报告（B=5000 次重采样）；各单元的完整区间在第 B 和 C 部分表中给出。


### 4.1 Atomic Quality is Concentrated, Not Distributed
### 4.1 原子质量是集中而非分散的


We first measure the atomic-quality probe $q\left( {c}_{k}^{\left( s\right) }\right)$ for every ECM in the library. On T6 (Table 1) the atomic-quality matrix is highly concentrated: a single cell, the seed $= {2024}$ REACH ECM, achieves an atomic success rate of 86.7%, while every other ECM in the library is at or below 26.7%, and the median atomic success rate across the 16-cell matrix is 0%.
我们首先对库中每个 ECM 测量原子质量探针$q\left( {c}_{k}^{\left( s\right) }\right)$。在 T6（表 1）上，原子质量矩阵高度集中：单个单元，即种子$= {2024}$ REACH ECM，原子成功率达到 86.7%，而库中其他所有 ECM 均不超过 26.7%，16 个单元构成的矩阵中位原子成功率为 0%。


<table><tr><td>Phase</td><td>seed=42</td><td>seed=7</td><td>seed=123</td><td>seed=2024</td></tr><tr><td>REACH</td><td>6.7%</td><td>0.0%</td><td>26.7%</td><td>86.7%</td></tr><tr><td>GRASP</td><td>3.3%</td><td>10.0%</td><td>0.0%</td><td>0.0%</td></tr><tr><td>LIFT</td><td>13.3%</td><td>0.0%</td><td>0.0%</td><td>0.0%</td></tr><tr><td>PLACE</td><td>23.3%</td><td>20.0%</td><td>0.0%</td><td>16.7%</td></tr></table>
<table><tbody><tr><td>阶段</td><td>seed=42</td><td>seed=7</td><td>seed=123</td><td>seed=2024</td></tr><tr><td>到达</td><td>6.7%</td><td>0.0%</td><td>26.7%</td><td>86.7%</td></tr><tr><td>抓取</td><td>3.3%</td><td>10.0%</td><td>0.0%</td><td>0.0%</td></tr><tr><td>抬起</td><td>13.3%</td><td>0.0%</td><td>0.0%</td><td>0.0%</td></tr><tr><td>放置</td><td>23.3%</td><td>20.0%</td><td>0.0%</td><td>16.7%</td></tr></tbody></table>


Table 1: T6 atomic-quality probe $q\left( c\right)$ over 30 episodes. The seed $= {2024}$ REACH ECM is the unique dominant cell.
表1：T6 的原子质量探测 $q\left( c\right)$，覆盖30个回合。种子 $= {2024}$ REACH ECM 是唯一的主导细胞。


We refer to the unique highest- $q\left( \cdot \right)$ cell as the dominant ${ECM}$ for that task. With $S = 4$ we observe a clean gap of more than 60pp between the dominant and the second-best ECM on T6; section B reproduces table 1 as a heatmap.
我们将该任务中最高的、唯一的 $q\left( \cdot \right)$ 细胞称为主导 ${ECM}$。借助 $S = 4$，我们在 T6 上观察到主导 ECM 与第二最佳 ECM 之间超过60个百分点的清晰差距；第 B 节将表1重现为热力图。


### 4.2 Composition Success is Driven by Dominant-ECM Inclusion
### 4.2 组合成功由主导-ECM的纳入驱动


Given the atomic concentration, we next ask whether composition outcome tracks the inclusion of the dominant ECM. For each (primary, alternative) seed pair on T6, we partition the 16 swap-subsets $\sigma$ into two groups: those that include the REACH phase (swapping the REACH ECM in or out) and those that do not. Table 2 reports the mean composition success rate per group.
在给定原子浓度的情况下，我们接着要问：组合结果是否随主导ECM的纳入而变化。在T6上对每一对（主种子、备种子），我们将16个交换子集 $\sigma$ 分成两组：包含REACH阶段（将REACH ECM置入或置出）与不包含REACH阶段的。表2给出了各组的平均组合成功率。


<table><tr><td>(primary, alt) seed pair</td><td>REACH $\in  \sigma$ [CI]</td><td>REACH $\notin  \sigma$ [CI]</td><td>$\Delta$ (pp) [CI]</td></tr><tr><td>(42, 2024)</td><td>66.2% [60.0, 72.1]</td><td>16.2% [11.7, 21.2]</td><td>+50.0 [+42.1, +57.5]</td></tr><tr><td>(123, 2024)</td><td>75.4% [69.6, 80.8]</td><td>28.3% [22.9, 34.2]</td><td>+47.1 [+39.2, +54.6]</td></tr><tr><td>(7, 123)</td><td>32.9% [27.1, 38.8]</td><td>13.8% [9.6, 18.3]</td><td>+19.2 [+11.7,+26.7]</td></tr><tr><td>(42, 7)</td><td>26.7% [21.2, 32.1]</td><td>22.5% [17.5, 27.9]</td><td>${4.2}\left\lbrack  {-{3.3}, + {11.7}}\right\rbrack$</td></tr></table>
<table><tbody><tr><td>（主，备）种子对</td><td>REACH $\in  \sigma$ [CI]</td><td>REACH $\notin  \sigma$ [CI]</td><td>$\Delta$（pp）[CI]</td></tr><tr><td>(42, 2024)</td><td>66.2% [60.0, 72.1]</td><td>16.2% [11.7, 21.2]</td><td>+50.0 [+42.1, +57.5]</td></tr><tr><td>(123, 2024)</td><td>75.4% [69.6, 80.8]</td><td>28.3% [22.9, 34.2]</td><td>+47.1 [+39.2, +54.6]</td></tr><tr><td>(7, 123)</td><td>32.9% [27.1, 38.8]</td><td>13.8% [9.6, 18.3]</td><td>+19.2 [+11.7,+26.7]</td></tr><tr><td>(42, 7)</td><td>26.7% [21.2, 32.1]</td><td>22.5% [17.5, 27.9]</td><td>${4.2}\left\lbrack  {-{3.3}, + {11.7}}\right\rbrack$</td></tr></tbody></table>


Table 2: Subset-swap success rate on T6, partitioned by whether REACH is in the swap-set, with bootstrap ${95}\%$ CIs (B=5000). The first three rows involve seed 2024 or seed 123 (the higher-quality REACH ECMs); swapping their REACH in produces large, CI-significant gains. The last row involves only seeds 42 and 7 (both below 7% atomic); the $\Delta \mathrm{{CI}}$ includes zero.
表2：在T6上子集交换的成功率，按REACH是否在交换集合中划分，并给出自助法${95}\%$置信区间（B=5000）。前三行涉及种子2024或种子123（质量更高的REACH ECM）；将其中的REACH交换进去会带来显著的提升，达到CI显著。最后一行只包含种子42和7（两者均低于7%的原子性）；$\Delta \mathrm{{CI}}$中包含0。


The signal is striking: when one of the seeds in the pair has a high-quality REACH ECM (rows 1-3), the composition success rate moves by +19 to +50pp depending on whether that ECM is in the swap-set. When neither seed has a high-quality REACH ECM (row 4), the swap-set choice is essentially irrelevant.
信号非常醒目：当一对中的某个种子拥有高质量REACH ECM（第1-3行）时，组合成功率会根据该ECM是否在交换集合中，提升+19至+50个百分点。若两者都没有高质量REACH ECM（第4行），交换集合的选择基本不产生影响。


### 4.3 Direction of the Effect
### 4.3 效应方向


The single-phase paired matrix (table 3) covers the $4 \times  4 = {16}$ (primary,alternative) seed combinations on the REACH phase, with the diagonal as the within-seed baseline. The sign-flip predicted by the mechanism is clear: when the primary lacks the dominant ECM (rows 42, 7, 123), swapping in the dominant REACH (column 2024) raises composition success by +37 to +60pp; when the primary has it (row 2024), swapping out for any of the three alternatives lowers success by -20 to -37pp. Column-means span 7.5% to 64.2% and are governed almost entirely by the atomic quality of the swapped-in ECM, not by which version was originally present.
单相配对矩阵（表3）涵盖了REACH阶段的$4 \times  4 = {16}$（主、替代）种子组合，对角线为同种子基线。机制预测的符号翻转十分明显：当主种子缺少主导ECM时（第42、7、123行），换入主导REACH（第2024列）会使组合成功率提高+37至+60个百分点；当主种子已具备它时（第2024行），换成三种替代中的任意一种都会使成功率降低-20至-37个百分点。列均值介于7.5%到64.2%之间，几乎完全由换入ECM的原子质量决定，而非原先存在的是哪个版本。


<table><tr><td></td><td>swap=42</td><td>swap $= 7$</td><td>swap=123</td><td>swap=2024</td><td>diag.</td></tr><tr><td>primary=42</td><td>13.3</td><td>0.0</td><td>33.3</td><td>60.0</td><td>13.3</td></tr><tr><td>primary=7</td><td>40.0</td><td>16.7</td><td>46.7</td><td>76.7</td><td>16.7</td></tr><tr><td>primary=123</td><td>3.3</td><td>0.0</td><td>33.3</td><td>70.0</td><td>33.3</td></tr><tr><td>primary=2024</td><td>16.7</td><td>13.3</td><td>30.0</td><td>50.0</td><td>50.0</td></tr><tr><td>col. mean</td><td>18.3</td><td>7.5</td><td>35.8</td><td>64.2</td><td>-</td></tr><tr><td>95% CI</td><td>[11.7, 25.8]</td><td>[3.3, 12.5]</td><td>[27.5, 45.0]</td><td>[55.8, 72.5]</td><td>-</td></tr></table>
<table><tbody><tr><td></td><td>swap=42</td><td>swap $= 7$</td><td>swap=123</td><td>swap=2024</td><td>对角线</td></tr><tr><td>primary=42</td><td>13.3</td><td>0.0</td><td>33.3</td><td>60.0</td><td>13.3</td></tr><tr><td>primary=7</td><td>40.0</td><td>16.7</td><td>46.7</td><td>76.7</td><td>16.7</td></tr><tr><td>primary=123</td><td>3.3</td><td>0.0</td><td>33.3</td><td>70.0</td><td>33.3</td></tr><tr><td>primary=2024</td><td>16.7</td><td>13.3</td><td>30.0</td><td>50.0</td><td>50.0</td></tr><tr><td>列均值</td><td>18.3</td><td>7.5</td><td>35.8</td><td>64.2</td><td>-</td></tr><tr><td>95%置信区间</td><td>[11.7, 25.8]</td><td>[3.3, 12.5]</td><td>[27.5, 45.0]</td><td>[55.8, 72.5]</td><td>-</td></tr></tbody></table>


Table 3: T6 REACH-phase paired swap matrix (success rate %, $N = {30}$ paired episodes per cell). The dominant column (swap=2024) is CI-disjoint from every other column; the dominant row (primary=2024) loses 20-37pp whenever its REACH is swapped out. Per-cell CIs in section B.
表3：T6 REACH阶段成对交换矩阵（成功率%，每个单元格$N = {30}$对回合）。主导列（swap=2024）与其他所有列均为CI不相交；主导行（primary=2024）只要其REACH被换出，便会下降20-37个百分点。各单元格CI见B节。


Both directions support the same conclusion: composition success is a function of which ECMs end up in the composition, not of how many were swapped or how distant they are from the original.
两个方向都支持同一结论：组合成功取决于最终进入组合的ECM，而不取决于交换了多少，或它们与原始项相距多远。


### 4.4 Negative Controls and Atomic Predictivity
### 4.4 否定性控制与原子可预测性


A structural alternative ("any swap hurts because composed neural controllers are inherently fragile") is ruled out by within-task negative controls. On T6, the three phases other than REACH lack a high-quality ECM in any seed (table 1: max atomic success rate $\leq  {23.3}\%$ , median 0%). Swapping ECMs in these phases produces no systematic column-mean shift on the corresponding $4 \times  4$ paired matrix: on GRASP,LIFT,and PLACE,swap-column means cluster within 7.5, 10.0, and 4.2pp respectively, an order of magnitude tighter than the 56.7pp spread on REACH. The dominant-skill effect therefore requires a true high-quality ECM, not just a swap event. Numerically, the REACH swap-column means rank 7.5%, 18.3%, ${35.8}\% ,{64.2}\%$ in lockstep with the atomic-quality probes ${0.0}\% ,{6.7}\% ,{26.7}\% ,{86.7}\%$ of the corresponding seeds: the atomic probe of the swapped-in ECM monotonically predicts the column mean of the post-swap composition.
一种结构性替代解释（“任何交换都会造成损害，因为组合神经控制器本质上很脆弱”）可通过任务内否定性控制排除。在 T6 上，除 REACH 外的三个阶段在任何随机种子下都缺乏高质量 ECM（表1：原子成功率最大值$\leq  {23.3}\%$，中位数为 0%）。在这些阶段交换 ECM 时，对应$4 \times  4$配对矩阵的列均值并未出现系统性偏移：在 GRASP、LIFT 和 PLACE 上，交换列均值分别聚集在 7.5、10.0 和 4.2 个百分点以内，这比 REACH 上 56.7 个百分点的跨度紧得多一个数量级。因此，主导技能效应要求存在真正的高质量 ECM，而不仅仅是一次交换事件。数值上，REACH 的交换列均值按 7.5%、18.3%、${35.8}\% ,{64.2}\%$排序，与对应种子的原子质量探针${0.0}\% ,{6.7}\% ,{26.7}\% ,{86.7}\%$完全一致：被交换进来的 ECM 的原子探针值能够单调预测交换后组合体的列均值。


## 5 Boundary: When Atomic Quality Saturates
## 5 边界：当原子质量趋于饱和


The dominant-skill effect described in Section 4 requires variation in atomic skill quality across versions: some ECMs must be markedly stronger than others. We characterize the boundary of this effect using a task on which all atomic ECMs are equally strong.
第4节所述的主导技能效应要求各版本间原子技能质量存在差异：某些ECM必须明显强于其他。我们以一个所有原子ECM同样强的任务来刻画这一效应的边界。


On T1_Pick, a single-arm pick of one object from a fixed table position, every (seed, phase) ECM in our library achieves $q\left( c\right)  = {100}\%$ atomic success rate (table 4). All four seeds saturate from the first training iteration and hold 100% across 15 iterations of the standard schedule; on the $N = {30}$ atomic-probe evaluation,every one of the 4 seeds $\times  4$ phases $= {16}$ cells reports ${100}\%$ success (120/120 episodes). The candidate set is uniformly saturated; there is no dominant cell.
在T1_Pick上，这是一项在固定桌面位置从中单臂拾取一个物体的任务，我们库中每个（seed, phase）ECM都达到了$q\left( c\right)  = {100}\%$的原子成功率（表4）。所有四个seed都在第一次训练迭代时就已饱和，并在标准日程的15次迭代中始终保持100%；在$N = {30}$原子探针评估中，4个seed中的每一个$\times  4$ phases $= {16}$ cells都报告了${100}\%$成功（120/120个回合）。候选集呈均匀饱和状态；不存在主导单元。


<table><tr><td>Phase</td><td>seed=42</td><td>seed=7</td><td>seed=123</td><td>seed=2024</td></tr><tr><td>REACH</td><td>100%</td><td>100%</td><td>100%</td><td>100%</td></tr><tr><td>GRASP</td><td>100%</td><td>100%</td><td>100%</td><td>100%</td></tr><tr><td>LIFT</td><td>100%</td><td>100%</td><td>100%</td><td>100%</td></tr><tr><td>PLACE</td><td>100%</td><td>100%</td><td>100%</td><td>100%</td></tr></table>
<table><tbody><tr><td>阶段</td><td>seed=42</td><td>seed=7</td><td>seed=123</td><td>seed=2024</td></tr><tr><td>到达</td><td>100%</td><td>100%</td><td>100%</td><td>100%</td></tr><tr><td>抓取</td><td>100%</td><td>100%</td><td>100%</td><td>100%</td></tr><tr><td>抬起</td><td>100%</td><td>100%</td><td>100%</td><td>100%</td></tr><tr><td>放置</td><td>100%</td><td>100%</td><td>100%</td><td>100%</td></tr></tbody></table>


Table 4: T1 atomic-quality probe,measured across all four seeds $(N = {30}$ episodes per cell, evaluation seeds 10000-10029; 120/120 episodes succeed). Every (seed, phase) ECM saturates, so by the mechanism of section 4 no dominant cell exists and no swap can shift outcomes.
表4：T1 原子质量探针，基于全部四个种子$(N = {30}$个回合/单元进行测量，评估种子为10000-10029；120/120个回合成功）。每个（种子，阶段）ECM 都已饱和，因此按第4节的机制，不存在主导单元，任何替换都无法改变结果。


By the mechanism of section 4, when every candidate ECM contributes equally well, every composition is equally successful and swapping any phase's ECM cannot shift outcomes. T1 is therefore a boundary case consistent with our mechanism: composition is robust to swap, precisely because there is no atomic-quality variation for the dominant-skill effect to act on.
按照第4节的机制，当每个候选ECM都同样出色地贡献时，每种组合都同样成功，替换任一阶段的ECM都无法改变结果。因此，T1 是一个与我们的机制一致的边界情形：组合对替换具有鲁棒性，正因为不存在可供主导技能效应作用的原子质量差异。


## 6 Why Behavioral Distance Fails to Predict Dominance
## 6 为什么行为距离无法预测优势


A natural alternative hypothesis is that the dominant ECM should be behaviorally atypical: with action distribution distinct from the others, the dominant ECM might be detectable by a cheap, model-free distance metric. We test this hypothesis directly. For each phase we compute the per-ECM mean off-diagonal action ${L}^{2}$ distance to its three siblings,evaluated over the paired episode pool, and ask whether the dominant cell ranks first. Within T6 REACH, the dominant ECM (seed=2024, atomic SR 86.7%) is rank 3 of 4 by mean pairwise distance (3.349); the most-distant ECM is the 0%-SR seed=7 at 3.543. Across all 16 T6 (seed, phase) ECMs the dominant cell is rank 7 of 16 by mean pairwise distance: under a uniform-rank null, $P\left( {\operatorname{rank} \leq  7}\right)  = {0.44}$ ,so the observation is consistent with the dominant cell being placed at random with respect to behavioral distance, not at the extreme. T3 and T4 atomic SR is degenerate at $0\%$ on most cells,so within-task ranking is uninformative there. The combined evidence: behavioral distance is a poor detector of the dominant cell - it places it near the middle of its phase, not at the top. Per-(task, phase) summary in table 9; full 12-panel heatmaps in section C.
自然的替代假设是，优势 ECM 应当在行为上“不寻常”：其动作分布与其他 ECM 明显不同，那么优势 ECM 可能能够被一种廉价、无需模型的距离度量所检测。我们直接检验这一假设。对每个阶段，我们计算该 ECM 相对于其三个同胞的、非对角线上的动作均值距离 ${L}^{2}$ ，在配对回合池上进行评估，并询问优势单元是否排名第一。在 T6 REACH 中，优势 ECM（seed=2024，原子 SR 86.7%）的平均成对距离为 3.349，在 4 个中排名第 3；最远的 ECM 是 0%-SR 的 seed=7，其距离为 3.543。在全部 16 个 T6（seed, phase）ECM 中，优势单元在平均成对距离上为第 7/16 名；在均匀排名的零假设下，$P\left( {\operatorname{rank} \leq  7}\right)  = {0.44}$ ，因此观察结果与优势单元相对于行为距离被随机放置一致，而不是处于极端位置。T3 和 T4 的原子 SR 在大多数单元上都在 $0\%$ 处退化，因此任务内排序没有信息量。综合证据表明：行为距离是一个劣质的优势单元探测器——它把优势单元放在其阶段的中间，而非置于最顶部。表 9 中给出了（任务、阶段）的逐任务汇总；完整的 12 面板热力图见 C 节。


## 7 Algorithm: Atomic-Quality Probing for Skill-Update Governance
## 7 算法：用于技能更新治理的原子质量探测


### 7.1 Problem Formulation
### 7.1 问题表述


A skill-update event is a tuple $\left( {p,a,{\pi }_{k}}\right)$ specifying that the phase- ${\pi }_{k}$ ECM in primary version $p$ is candidate to be replaced by the phase- ${\pi }_{k}$ ECM from version $a$ . A selector is a function $\mathcal{S} : \left( {p,a,{\pi }_{k}}\right)  \rightarrow  \{$ ACCEPT,REJECT $\}$ that decides whether to apply the update. The atomic-quality probe is, in spirit, a capability-targeted behavioral test of the candidate skill in isolation, in the lineage of probe-as-test methodologies popularized in NLP by CheckList [20].
一次技能更新事件是一个元组 $\left( {p,a,{\pi }_{k}}\right)$，表示主版本 $p$ 中的 phase- ${\pi }_{k}$ ECM 是候选被版本 $a$ 中的 phase- ${\pi }_{k}$ ECM 替换。选择器是一个函数 $\mathcal{S} : \left( {p,a,{\pi }_{k}}\right)  \rightarrow  \{$ ACCEPT,REJECT $\}$ ，用于决定是否应用该更新。从本质上说，原子质量探测是一种面向能力的、对候选技能进行孤立行为测试的方法，沿袭了 CheckList [20] 在 NLP 中推广的“探测即测试”方法论。


We compare each selector against the oracle selector that has post-hoc access to the true paired composition success rate of the post-update composition ${C}^{\prime }$ and accepts iff $\mathbb{P}$ (success $\left. {C}^{\prime }\right)  \geq  \mathbb{P}$ (success $|C) - \tau$ for a tolerance threshold $\tau$ (we report results for $\tau  = 5\mathrm{{pp}}$ ). Oracle match is the fraction of update events where a selector's decision agrees with the oracle.
我们将每个选择器与 oracle 选择器进行比较；后者可事后获取更新后组合 ${C}^{\prime }$ 的真实配对组合成功率，并且当且仅当 $\mathbb{P}$ (success $\left. {C}^{\prime }\right)  \geq  \mathbb{P}$ (success $|C) - \tau$ 对于容差阈值 $\tau$ 时接受（我们报告 $\tau  = 5\mathrm{{pp}}$ 的结果）。Oracle match 是某个选择器的决策与 oracle 一致的更新事件占比。


### 7.2 Selectors
### 7.2 选择器


We benchmark seven selectors. Naive always accepts (lower bound); Freeze always rejects (the BLADE/SymSkill frozen-library deployment); AtomicOnly accepts iff $q\left( {c}_{a}\right)  \geq  q\left( {c}_{p}\right)  - \tau$ using the atomic probe per ECM (zero per-decision cost, amortized across events involving the same ECM); FullReval accepts iff $\mathbb{P}\left( {\text{ success } \mid  {C}^{\prime }}\right)  \geq  \mathbb{P}\left( {\text{ success } \mid  C}\right)  - \tau$ using the composition probe (one probe per decision); Hybrid $\left( m\right)$ uses AtomicOnly when $\left| {q\left( {c}_{a}\right)  - q\left( {c}_{p}\right) }\right|  \geq  m$ and falls back to FullReval otherwise,with $m \in  \{ {10},{20},{30}\} \mathrm{{pp}}$ .
我们对七种选择器进行基准测试。朴素方案始终接受（下界）；Freeze 始终拒绝（BLADE/SymSkill 冻结库部署）；AtomicOnly 在且仅在 $q\left( {c}_{a}\right)  \geq  q\left( {c}_{p}\right)  - \tau$ 时接受，使用按 ECM 的原子探测（每次决策零成本，在涉及同一 ECM 的事件中摊还）；FullReval 在且仅在 $\mathbb{P}\left( {\text{ success } \mid  {C}^{\prime }}\right)  \geq  \mathbb{P}\left( {\text{ success } \mid  C}\right)  - \tau$ 时接受，使用组合探测（每次决策一次探测）；Hybrid $\left( m\right)$ 在 $\left| {q\left( {c}_{a}\right)  - q\left( {c}_{p}\right) }\right|  \geq  m$ 时使用 AtomicOnly，否则回退到 FullReval，且为 $m \in  \{ {10},{20},{30}\} \mathrm{{pp}}$ 。


Per-decision cost is the number of composition probe-episodes used: FullReval costs $N = {30}$ episodes; AtomicOnly costs 0 (amortized across all events involving the same ECM); Hybrid $\left( m\right)$ costs ${30} \cdot  \mathbb{P}\left( {\left| {{q}_{a} - {q}_{p}}\right|  < m}\right)$ in expectation.
每次决策的成本等于使用的组合探测-回合数：FullReval 需要 $N = {30}$ 个回合；AtomicOnly 成本为 0（在所有涉及同一 ECM 的事件中摊还）；Hybrid $\left( m\right)$ 期望成本为 ${30} \cdot  \mathbb{P}\left( {\left| {{q}_{a} - {q}_{p}}\right|  < m}\right)$ 。


### 7.3 Results on T6
### 7.3 T6 结果


Table 5 reports oracle-match rate at $\tau  = 5\mathrm{{pp}}$ on T6 across 48 update events $(4 \times  3$ ordered seed pairs $\times  4$ phases). Because the events are paired,we test the headline 22.9pp gap between AtomicOnly $\left( {{31}/{48}}\right)$ and FullReval $\left( {{42}/{48}}\right)$ with McNemar's exact-binomial test on the $2 \times  2$ disagreement table (both correct 28,AtomicOnly-only 3,FullReval-only 14,neither 3): equality is rejected at $p = {0.013}$ . To address the event-level dependence introduced by re-using the same swapped-in ECM across multiple events, we additionally run a cluster permutation test with the 16 swapped-in ECMs (phase $\times$ new_seed) as clusters: $p = {0.018}$ (two-sided, $B = {5000}$ ); furthermore,of the 16 clusters 9 favour FullReval,7 tie,and 0 favour AtomicOnly, so the directionality is uniform. The 22.9pp gap is therefore not an artifact of within-cluster correlation. Hybrid $\left( {m = {10}}\right)$ triggers FullReval on 22 of 48 events (45.8% cost) while preserving ${36}/{48} = {75.0}\%$ oracle match,recovering most of the FullReval gap at less than half its cost.
表 5 报告了在 T6 上，48 次更新事件 $(4 \times  3$ 中 $\tau  = 5\mathrm{{pp}}$ 的 oracle-match 率，按种子对相位 $\times  4$ 排序。由于这些事件成对，我们使用 $2 \times  2$ 不一致表上的 McNemar 精确-二项检验来检验 AtomicOnly $\left( {{31}/{48}}\right)$ 与 FullReval $\left( {{42}/{48}}\right)$ 之间头号 22.9pp 的差距（两者均正确 28，AtomicOnly-only 3，FullReval-only 14，neither 3）：在 $p = {0.013}$ 处拒绝等同性。为应对通过在多个事件中重复使用同一个被替换入的 ECM 所引入的事件级依赖，我们还对 16 个被替换入的 ECM（相位 $\times$ new_seed）进行基于簇的置换检验作为簇：$p = {0.018}$（双侧，$B = {5000}$ ）；此外，在这 16 个簇中有 9 个支持 FullReval，7 个并列，0 个支持 AtomicOnly，因此方向性一致。因而，这 22.9pp 的差距并非簇内相关性的产物。Hybrid $\left( {m = {10}}\right)$ 在 48 次事件中 22 次触发 FullReval（成本 45.8%），同时保持 ${36}/{48} = {75.0}\%$ 的 oracle match，在成本不到其一半的情况下恢复了大部分 FullReval 的差距。


<table><tr><td>Selector</td><td>Oracle match [95% CI]</td><td>Cost</td></tr><tr><td>Naive (accept all)</td><td>43.8% 29.2, 58.3]</td><td>0%</td></tr><tr><td>Freeze (reject all; BLADE/SymSkill)</td><td>56.2% 41.7, 68.8]</td><td>0%</td></tr><tr><td>AtomicOnly</td><td>64.6% 52.1, 77.1</td><td>0%</td></tr><tr><td>FullReval</td><td>87.5% 77.1, 95.8</td><td>100%</td></tr><tr><td>Hybrid(m=10)</td><td>75.0% 62.5, 87.5]</td><td>45.8%</td></tr><tr><td>Hybrid $\left( {m = {20}}\right)$</td><td>81.2% 68.8, 91.7</td><td>70.8%</td></tr><tr><td>Hybrid $\left( {m = {30}}\right)$</td><td>87.5% 77.1, 95.8</td><td>87.5%</td></tr></table>
<table><tbody><tr><td>选择器</td><td>Oracle匹配[95%置信区间]</td><td>成本</td></tr><tr><td>朴素（全部接收）</td><td>43.8% 29.2, 58.3]</td><td>0%</td></tr><tr><td>冻结（全部拒绝；BLADE/SymSkill）</td><td>56.2% 41.7, 68.8]</td><td>0%</td></tr><tr><td>仅原子</td><td>64.6% 52.1, 77.1</td><td>0%</td></tr><tr><td>全重评</td><td>87.5% 77.1, 95.8</td><td>100%</td></tr><tr><td>混合（m=10）</td><td>75.0% 62.5, 87.5]</td><td>45.8%</td></tr><tr><td>混合 $\left( {m = {20}}\right)$</td><td>81.2% 68.8, 91.7</td><td>70.8%</td></tr><tr><td>混合 $\left( {m = {30}}\right)$</td><td>87.5% 77.1, 95.8</td><td>87.5%</td></tr></tbody></table>


Table 5: T6 oracle-match rate $\left( {\tau  = 5\mathrm{{pp}}}\right)$ on 48 update events,with bootstrap 95% CIs. FullReval is best at full cost; Hybrid $\left( {m = {20}}\right)$ recovers most of the gain at $\sim  {30}\%$ cost reduction. AtomicOnly is meaningfully better than the two naive baselines (CIs disjoint) at zero per-decision cost.
表 5：T6 的 oracle-match 率 $\left( {\tau  = 5\mathrm{{pp}}}\right)$，共 48 次更新事件，并附 bootstrap 95% 置信区间。FullReval 在全成本下最佳；Hybrid $\left( {m = {20}}\right)$ 以 $\sim  {30}\%$ 的成本降低恢复了大部分增益。AtomicOnly 在零的按决策成本下，显著优于这两个朴素基线（置信区间不重叠）。


<img src="https://cdn.noedgeai.com/bo_d8j3koilb0pc73bdlaig_7.jpg?x=308&y=218&w=1180&h=305&r=0"/>



Figure 1: Cost-accuracy Pareto frontier for the seven selectors, per task and on cross-task average. AtomicOnly sits at zero cost; FullReval at 100%. Hybrid $\left( {m = {10}}\right)$ is robustly competitive with FullReval at substantially reduced cost across all three tasks.
图 1：七种选择器的成本-精度帕累托前沿，按任务分别展示，并给出跨任务平均。AtomicOnly 位于零成本；FullReval 为 100%。Hybrid $\left( {m = {10}}\right)$ 在三个任务上都以显著更低的成本表现出与 FullReval 稳健的竞争力。


### 7.4 Cross-Task Pattern (Mixed-Oracle Caveat)
### 7.4 跨任务模式（混合预言器注意事项）


Extending the benchmark to T3 and T4 introduces a methodological subtlety we surface explicitly. On these two tasks all atomic policies achieve $0\%$ task-success rate (section 9); the SR-based oracle is degenerate, so the oracle is defined on reward instead. The cross-task average in table 6 therefore mixes oracles and should be read as suggestive; the methodologically conservative numbers are the T6 (SR) column above. With that caveat: AtomicOnly (cost 0) is within 3pp of FullReval (cost 100%) on the cross-task average; Hybrid $\left( {m = {10}}\right)$ ties FullReval at 75% of its cost; on T3 and T4 specifically,AtomicOnly beats FullReval, reflecting that when composition signals are noisy or collapsed a per-skill probe can be cleaner than a per-composition probe. We report bootstrap 95% CIs throughout, following Agarwal et al. [21]'s recommendations for sparse-trial RL benchmarks.
将基准扩展到 T3 和 T4，会引入一个我们明确提出的方法学细微差别。在这两个任务上，所有原子策略都达到 $0\%$ 的任务成功率（第 9 节）；基于 SR 的预言器退化，因此预言器改定义在奖励上。因而，第 6 表中的跨任务平均值混合了不同的预言器，只应被视为启发性的；上方的方法学保守数字为 T6（SR）这一列。带着这个注意事项：AtomicOnly（成本 0）在跨任务平均上与 FullReval（成本 100%）相差不超过 3 个百分点；Hybrid $\left( {m = {10}}\right)$ 的性价比与其相当，达到 FullReval 成本的 75%；具体到 T3 和 T4，AtomicOnly 优于 FullReval，这表明当组合信号嘈杂或塌缩时，相比基于组合的探测，逐技能的探测可能更清晰。我们在全文报告自助法（bootstrap）的 95% 置信区间，并遵循 Agarwal 等人 [21] 对稀疏试验 RL 基准的建议。


<table><tr><td>Selector</td><td>T6 (SR)</td><td>T3 (rew)</td><td>T4 (rew)</td><td>Avg. $\left( {n = {144}}\right)$</td></tr><tr><td>Naive</td><td>43.8 [29, 58]</td><td>56.2 [42, 71]</td><td>54.2 [40, 69]</td><td>51.4 43,60]</td></tr><tr><td>Freeze</td><td>56.2 [42, 69]</td><td>43.8 [29, 58]</td><td>45.8 31, 60</td><td>48.6 40,57]</td></tr><tr><td>AtomicOnly</td><td>64.6 5 [52, 77]</td><td>72.9 [60, 85]</td><td>60.4 46, 73</td><td>66.0 58, 74</td></tr><tr><td>FullReval</td><td>87.5 57, 96</td><td>4.6 [52, 77]</td><td>54.2 40,69</td><td>68.8 61, 76]</td></tr><tr><td>Hybrid(m=10)</td><td>75.0 62,85</td><td>75.0 ] [62, 88]</td><td>54.2 40,69</td><td>68.1 60, 76</td></tr><tr><td>Hybrid $\left( {m = {20}}\right)$</td><td>81.2 [68, 90]</td><td>.7 [53, 78]</td><td>54.2 40,67</td><td>67.4 [59, 74]</td></tr><tr><td>Hybrid $\left( {m = {30}}\right)$</td><td>87.5 75, 94]</td><td>64.6 [50,77]</td><td>54.2 40,67</td><td>68.8 61, 76]</td></tr></table>
<table><tbody><tr><td>选择器</td><td>T6（SR）</td><td>T3（rew）</td><td>T4（rew）</td><td>平均值。$\left( {n = {144}}\right)$</td></tr><tr><td>朴素</td><td>43.8 [29, 58]</td><td>56.2 [42, 71]</td><td>54.2 [40, 69]</td><td>51.4 43,60]</td></tr><tr><td>冻结</td><td>56.2 [42, 69]</td><td>43.8 [29, 58]</td><td>45.8 31, 60</td><td>48.6 40,57]</td></tr><tr><td>仅原子</td><td>64.6 5 [52, 77]</td><td>72.9 [60, 85]</td><td>60.4 46, 73</td><td>66.0 58, 74</td></tr><tr><td>完全重评</td><td>87.5 57, 96</td><td>4.6 [52, 77]</td><td>54.2 40,69</td><td>68.8 61, 76]</td></tr><tr><td>混合（m=10）</td><td>75.0 62,85</td><td>75.0 ] [62, 88]</td><td>54.2 40,69</td><td>68.1 60, 76</td></tr><tr><td>混合 $\left( {m = {20}}\right)$</td><td>81.2 [68, 90]</td><td>.7 [53, 78]</td><td>54.2 40,67</td><td>67.4 [59, 74]</td></tr><tr><td>混合 $\left( {m = {30}}\right)$</td><td>87.5 75, 94]</td><td>64.6 [50,77]</td><td>54.2 40,67</td><td>68.8 61, 76]</td></tr></tbody></table>


Table 6: Per-task and cross-task oracle match (%) with bootstrap 95% CIs. T6 uses an SR-based oracle; T3 and T4 use a reward oracle (necessary because all atomic policies score 0% task-success). The cross-task average mixes oracles and should be interpreted as suggestive; T6 numbers are the methodologically conservative ones.
表 6：各任务与跨任务的先验匹配率（%）（自助法 95% 置信区间）。T6 使用基于 SR 的先验；T3 和 T4 使用奖励先验（必要的原因是所有原子策略的任务成功率均为 0%）。跨任务平均混合了先验，应仅作提示性解读；T6 的数值是方法上更为保守的结果。


The Hybrid selector pseudocode appears in section A.
混合选择器的伪代码见附录 A。


## 8 Discussion
## 8 讨论


We interpret the dominant-skill effect as the per-skill instance of the regression-on-update phenomenon studied in the backward-compatibility literature [22, 23, 24]: even when an updated function is on average no worse, it can flip individual downstream decisions in ways the old function did not. The same asymmetry explains the vanishing population mean: swapping toward and away from the dominant ECM produce gains and losses of comparable magnitude, so the signal lives in the conditional structure rather than the marginal mean. The deployment recipe is to probe every candidate update atomically first, invoking the (far more expensive) composition probe only when the atomic margin is insufficient. We measured all three candidate sub-mechanisms (hand-off state coverage, action smoothness, trajectory-length distribution); the dominant ECM is not the absolute outlier on any single channel (section E), suggesting the robustness asymmetry operates through interaction or some more structural property that the atomic-quality probe captures directly but no single behavioral measure recovers.
我们将主导技能效应解释为回归-更新现象在单个技能上的体现，该现象已在向后兼容文献中得到研究[22, 23, 24]：即使更新后的函数平均而言并不更差，它仍可能以旧函数不会的方式翻转个别下游决策。相同的不对称性也解释了总体均值的消失：向主导 ECM 切换以及从其切回所带来的收益与损失幅度相近，因此信号存在于条件结构中，而非边际均值中。部署策略是先对每个候选更新进行原子级探测，只有在原子裕度不足时，才调用代价高得多的组合探测。我们测量了全部三个候选子机制（交接状态覆盖、动作平滑性、轨迹长度分布）；主导 ECM 在任何单一通道上都不是绝对离群值（第 E 节），这表明鲁棒性不对称是通过交互作用或某种更结构性的属性发挥作用，而原子质量探测能够直接捕捉这一点，但没有任何单一行为指标能够恢复它。


## 9 Limitations and Future Work
## 9 局限与未来工作


Empirical scope. The dominant-skill effect is demonstrated on a single positive task (T6); the boundary case (T1) and the behavioral-distance refutation across three tasks (T6, T3, T4) buttress but do not multiply that evidence. Four of our six candidate robosuite tasks (T2-T5) reached 0% atomic success under our standard SAC schedule, so the effect is undefined there. We do not claim it is absent, only that the schedule yields no measurable candidate set. We attempted T3 scaling along both a longer-schedule arm and a reward-shaping arm; neither cleared the ${40}\%  - {80}\%$ Goldilocks zone our paired cross-seed swap matrix requires (section H). The dominant ECM on T6 is identified at $S = 4$ and should be read as "the highest-quality ECM" rather than a strict combinatorial claim; larger $S$ would likely reveal a continuum of quality with the same predictive structure.
经验范围。主导技能效应仅在一个正向任务（T6）上得到验证；边界情形（T1）以及跨三个任务（T6、T3、T4）的行为距离反驳虽可支持这一证据，但并未扩大其覆盖面。我们六个候选 robosuite 任务中的四个（T2-T5）在标准 SAC 训练安排下的原子成功率均为 0%，因此在这些任务上该效应无从定义。我们并不声称它不存在，只是该训练安排未产出可测的候选集合。我们尝试沿着更长训练日程的分支和奖励塑形分支对 T3 进行扩展；两者都未进入我们配对跨种子交换矩阵所要求的 ${40}\%  - {80}\%$ 金发姑娘区间（见第 H 节）。T6 上的主导 ECM 识别于 $S = 4$，应理解为“质量最高的 ECM”，而非严格的组合学断言；更大的 $S$ 很可能会揭示出具有相同预测结构的质量连续谱。


Mixed-oracle algorithm benchmark; idealized update model. The cross-task algorithm comparison (table 6) uses a success-rate oracle on T6 but a reward oracle on T3 and T4 out of necessity, so its average should be read as suggestive. The cross-version swap protocol itself simulates update via independent retraining; realistic continual-learning updates (fine-tuning, RLHF, domain adaptation) likely produce smoother version shifts on which the same atomic-probe primitive should still apply but with different effect sizes.
混合 oracle 算法基准；理想化更新模型。跨任务算法比较（表 6）出于必要在 T6 上使用成功率 oracle，而在 T3 和 T4 上使用奖励 oracle，因此其平均结果应仅作提示性解读。跨版本交换协议本身通过独立重训练来模拟更新；现实中的持续学习更新（微调、RLHF、领域自适应）很可能产生更平滑的版本转移，而同样的原子探测原语在其中仍应适用，只是效应大小会不同。


Single embodiment. All experiments use the same robosuite Panda arm. Cross-embodiment composition stability is an open direction.
单一具身。所有实验均使用同一款 robosuite Panda 机械臂。跨具身组合稳定性是一个开放方向。


## 10 Conclusion
## 10 结论


We characterized composition stability under skill-update events in compositional robot policies. On the dual-arm peg-in-hole task a dominant-skill effect governs composition outcomes: a single high-atomic-quality ECM in the candidate set drives success, and swapping it shifts the rate by up to 50pp; on a saturated single-arm pick task the effect is by construction undefined; and off-policy behavioral-distance metrics fail to identify the dominant ECM on all three tasks tested. The atomic-quality probe and the Hybrid Selector built on it match the oracle within 3pp of full composition revalidation at zero or fractional per-decision cost across 144 update events. To our knowledge, this is the first principled, deployment-ready primitive for skill-update governance in continually-updated skill libraries. The contribution is demonstrated on a representative dual-arm contact-rich task and verified by an independent re-run with state logging (section B); cross-task and cross-embodiment generalization remain the principal open questions, as detailed in section 9.
我们刻画了在组合式机器人策略中，技能更新事件下的组合稳定性。在双臂插销装配任务中，主导技能效应主宰组合结果：候选集里单个高原子质量的 ECM 即能决定成功，而将其替换会使成功率最高变化 50 个百分点；在饱和的单臂抓取任务上，该效应在构造上是未定义的；并且，在我们测试的三项任务中，基于离策略的行为距离指标都无法识别主导 ECM。原子质量探针以及基于其构建的混合选择器能在 144 次更新事件中，以零或分数级的按决策成本，精准到与完整组合再验证相差不超过 3pp 的程度。就我们所知，这是首个具有原则性、可部署的技能更新治理基础原语，用于持续更新的技能库。该贡献在一个具有代表性的双臂接触丰富任务上得到验证，并通过带有状态记录的独立重跑复核（第 B 节）。跨任务与跨形体的泛化仍是主要未解问题，如第 9 节所述。


## References
## 参考文献


[1] M. J. Kim, K. Pertsch, S. Karamcheti, T. Xiao, A. Balakrishna, S. Nair, R. Rafailov, E. Foster, G. Lam, P. Sanketi, et al. OpenVLA: An open-source vision-language-action model. In Conference on Robot Learning (CoRL), 2024. URL https://arxiv.org/ abs/2406.09246.
[1] M. J. Kim，K. Pertsch，S. Karamcheti，T. Xiao，A. Balakrishna，S. Nair，R. Rafailov，E. Foster，G. Lam，P. Sanketi 等。OpenVLA：一个开源的视觉-语言-动作模型。载于机器人学习会议（CoRL），2024。URL https://arxiv.org/ abs/2406.09246。


[2] Octo Model Team, D. Ghosh, H. Walke, K. Pertsch, K. Black, O. Mees, S. Dasari, J. Hejna, T. Kreiman, C. Xu, et al. Octo: An open-source generalist robot policy. In Robotics: Science and Systems (RSS), 2024. URL https://arxiv.org/abs/2405.12213.
[2] Octo Model Team，D. Ghosh，H. Walke，K. Pertsch，K. Black，O. Mees，S. Dasari，J. Hejna，T. Kreiman，C. Xu 等。Octo：一个开源的通用型机器人策略。载于《机器人：科学与系统》（RSS），2024。URL https://arxiv.org/abs/2405.12213。


[3] K. Black, N. Brown, D. Driess, A. Esmail, M. Equi, C. Finn, N. Fusai, L. Groom, K. Hausman, B. Ichter, et al. , To: A vision-language-action flow model for general robot control. In Robotics: Science and Systems (RSS), 2025. URL https://arxiv.org/ abs/2410.24164.
[3] K. Black，N. Brown，D. Driess，A. Esmail，M. Equi，C. Finn，N. Fusai，L. Groom，K. Hausman，B. Ichter 等。To：一种用于通用机器人控制的视觉-语言-动作流模型。载于《机器人：科学与系统》（RSS），2025。URL https://arxiv.org/ abs/2410.24164。


[4] W. Liu, N. Nie, R. Zhang, J. Mao, and J. Wu. Learning compositional behaviors from demonstration and language. In Conference on Robot Learning (CoRL), 2024. URL https://arxiv.org/abs/2505.21981.LLM extracts preconditions/effects for high-level actions; neural controllers per action. Closest neighbor to typed-composition idea.
[4] W. Liu，N. Nie，R. Zhang，J. Mao 和 J. Wu。从演示和语言中学习组合行为。载于机器人学习会议（CoRL），2024。URL https://arxiv.org/abs/2505.21981.LLM 从中提取高层动作的前提/结果；每个动作对应独立的神经控制器。最接近类型化组合的思想。


[5] Y. S. Shao, H. Zheng, Y. Sun, A. Chaudhari, A. Kumar, and N. Figueroa. SymSkill: Symbol and skill co-invention for data-efficient and reactive long-horizon manipulation. arXiv preprint arXiv:2510.01661, 2025. URL https://arxiv.org/abs/2510.01661.Jointly learns predicates, operators, skills from unsegmented demo; RoboCasa 6-step composition with real-time recovery.
[5] Y. S. Shao，H. Zheng，Y. Sun，A. Chaudhari，A. Kumar 和 N. Figueroa。SymSkill：面向数据高效且可反应的长时域操作的符号与技能协同发明。arXiv 预印本 arXiv:2510.01661，2025。URL https://arxiv.org/abs/2510.01661.Jointly learns predicates, operators, skills from unsegmented demo; RoboCasa 6-step composition with real-time recovery.


[6] U. A. Mishra, S. Xue, Y. Chen, and D. Xu. Generative skill chaining: Long-horizon skill planning with diffusion models. In Conference on Robot Learning (CoRL), 2023. URL https://arxiv.org/abs/2401.03360.Learns joint (precondition, skill params, effect) diffusion per skill; conditional sampling for chaining.
[6] U. A. Mishra，S. Xue，Y. Chen 和 D. Xu。生成式技能串联：用扩散模型进行长时域技能规划。载于机器人学习会议（CoRL），2023。URL https://arxiv.org/abs/2401.03360.Learns joint (precondition, skill params, effect) diffusion per skill; conditional sampling for chaining.


[7] G. Wang, Y. Xie, Y. Jiang, A. Mandlekar, C. Xiao, Y. Zhu, L. Fan, and A. Anandkumar. Voyager: An open-ended embodied agent with large language models. arXiv preprint arXiv:2305.16291, 2023. URL https://arxiv.org/abs/2305.16291.GPT-4 auto-generates executable code into a skill library with self-verification; append-only, no typed interfaces.
[7] G. Wang，Y. Xie，Y. Jiang，A. Mandlekar，C. Xiao，Y. Zhu，L. Fan 和 A. Anandkumar。Voyager：一种带有大语言模型的开放式具身体智能体。arXiv 预印本 arXiv:2305.16291，2023。URL https://arxiv.org/abs/2305.16291.GPT-4 自动将可执行代码生成到技能库中，并进行自我验证；仅追加，不使用类型化接口。


[8] J. Zhang, J. Zhang, K. Pertsch, Z. Liu, X. Ren, M. Chang, S.-H. Sun, and J. J. Lim. Bootstrap your own skills: Learning to solve new tasks with large language model guidance. CoRL, 2023. URL https://clvrai.github.io/boss/.BOSS: LLM-guided growing of skill library; chains base skills into long-horizon behaviors.
[8] J. Zhang，J. Zhang，K. Pertsch，Z. Liu，X. Ren，M. Chang，S.-H. Sun 和 J. J. Lim。自助式技能引导：用大语言模型指导学习解决新任务。CoRL，2023。URL https://clvrai.github.io/boss/.BOSS：用 LLM 指导逐步扩展技能库；将基础技能串联成长时域行为。


[9] W. Wan, Y. Zhu, R. Shah, and Y. Zhu. LOTUS: Continual imitation learning for robot manipulation through unsupervised skill discovery. In IEEE International Conference on Robotics and Automation (ICRA), 2024. URL https://arxiv.org/abs/2311.02058.Continual skill discovery from open-vocabulary VLM; append-only skill library.
[9] W. Wan，Y. Zhu，R. Shah 和 Y. Zhu。LOTUS：通过无监督技能发现实现机器人的持续模仿学习。载于 IEEE 国际机器人与自动化会议（ICRA），2024。URL https://arxiv.org/abs/2311.02058.Continual skill discovery from open-vocabulary VLM; append-only skill library.


[10] M. Ahn, A. Brohan, et al. Do as i can, not as i say: Grounding language in robotic affordances. In CoRL, 2022. URL https://arxiv.org/abs/2204.01691.LLM suggests actions weighted by learned affordance value function; foundational LLM+robotics.
[10] M. Ahn，A. Brohan 等。照我能做到的做，而不是照我说的做：在机器人可操作性中落地语言。载于 CoRL，2022。URL https://arxiv.org/abs/2204.01691.LLM suggests actions weighted by learned affordance value function; foundational LLM+robotics.


[11] J. Liang, W. Huang, F. Xia, P. Xu, K. Hausman, B. Ichter, P. Florence, and A. Zeng. Code as policies: Language model programs for embodied control. In ICRA, 2023. URL https://arxiv.org/abs/2209.07753.LLMs generate Python code that composes perception and control primitives.
[11] J. Liang, W. Huang, F. Xia, P. Xu, K. Hausman, B. Ichter, P. Florence, 和 A. Zeng. 代码即策略：用于具身控制的语言模型程序。载于 ICRA，2023。URL https://arxiv.org/abs/2209.07753.LLM 会生成 Python 代码，将感知与控制原语进行组合。


[12] Y. Lee, J. J. Lim, A. Anandkumar, and Y. Zhu. Adversarial skill chaining for long-horizon robot manipulation via terminal state regularization. In Conference on Robot Learning (CoRL), 2021. URL https://arxiv.org/abs/2111.07999.T-STAR: terminal-state regularization for skill chaining; closest neighbor on hand-off-state mismatch.
[12] Y. Lee, J. J. Lim, A. Anandkumar, 和 Y. Zhu. 面向长时域机器人操作的对抗式技能串联：通过终端状态正则化实现。载于机器人学习会议（CoRL），2021。URL https://arxiv.org/abs/2111.07999.T-STAR：用于技能串联的终端状态正则化；以及交接状态不匹配时最接近的邻居。


[13] Y. Chen, C. Wang, L. Fei-Fei, and C. K. Liu. Sequential dexterity: Chaining dexterous policies for long-horizon manipulation. In Conference on Robot Learning (CoRL), 2023. URL https://arxiv.org/abs/2309.00987.Transition feasibility function gates dexterous policy chaining; closest analogue to the atomic-quality probe at the policy-selection level.
[13] Y. Chen, C. Wang, L. Fei-Fei, 和 C. K. Liu. 顺序灵巧性：用于长时域操作的灵巧策略串联。载于机器人学习会议（CoRL），2023。URL https://arxiv.org/abs/2309.00987.Transition feasibility function 用于门控灵巧策略串联；在策略选择层面，它与原子质量探针最接近。


[14] T. Huang, K. Chen, W. Wei, J. Li, Y. Long, and Q. Dou. Value-informed skill chaining for policy learning of long-horizon tasks with surgical robot. In IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2023. URL https: //arxiv.org/abs/2307.16503. State-value function gates skill transitions; same gating mechanism as the atomic probe, different domain.
[14] T. Huang, K. Chen, W. Wei, J. Li, Y. Long, 和 Q. Dou. 面向外科机器人长时域任务的策略学习：基于价值引导的技能串联。载于 IEEE/RSJ 智能机器人与系统国际会议（IROS），2023。URL https: //arxiv.org/abs/2307.16503. 状态价值函数用于门控技能转换；其门控机制与原子探针相同，只是领域不同。


[15] J. Fu, M. Norouzi, O. Nachum, G. Tucker, Z. Wang, A. Novikov, M. Yang, M. R. Zhang, Y. Chen, A. Kumar, C. Paduraru, S. Levine, and T. L. Paine. Benchmarks for deep off-policy evaluation. In International Conference on Learning Representations (ICLR), 2021. URL https://arxiv.org/abs/2103.16596.DOPE: canonical Off-Policy Evaluation benchmark suite; closest methodological neighbour for the per-skill atomic-quality probe.
[15] J. Fu, M. Norouzi, O. Nachum, G. Tucker, Z. Wang, A. Novikov, M. Yang, M. R. Zhang, Y. Chen, A. Kumar, C. Paduraru, S. Levine, 和 T. L. Paine. 深度离策略评估的基准。载于国际学习表征会议（ICLR），2021。URL https://arxiv.org/abs/2103.16596.DOPE：标准的离策略评估基准套件；与面向每个技能的原子质量探针在方法层面最接近的邻居。


[16] K. Konyushkova, Y. Chen, T. Le Paine, C. Gulcehre, C. Paduraru, D. J. Mankowitz, M. Denil, and N. de Freitas. Active offline policy selection. In Advances in Neural Information Processing Systems (NeurIPS), 2021. URL https://arxiv.org/abs/2106.10251.Cheap offline-eval surrogate warm-starts a budgeted online-eval policy selection - the structural template of our Hybrid Selector.
[16] K. Konyushkova, Y. Chen, T. Le Paine, C. Gulcehre, C. Paduraru, D. J. Mankowitz, M. Denil, 和 N. de Freitas. 主动离线策略选择。载于神经信息处理系统进展（NeurIPS），2021。URL https://arxiv.org/abs/2106.10251.Cheap 离线评估替代模型对预算约束的在线评估策略选择进行热启动——即我们混合选择器（Hybrid Selector）的结构模板。


[17] R. S. Sutton, D. Precup, and S. Singh. Between MDPs and semi-MDPs: A framework for temporal abstraction in reinforcement learning. Artificial Intelligence, 112(1-2): 181-211, 1999. doi:10.1016/S0004-3702(99)00052-1.
[17] R. S. Sutton, D. Precup, 和 S. Singh. 在 MDP 与半 MDP 之间：强化学习中的时间抽象框架。人工智能（Artificial Intelligence），112(1-2)：181-211，1999。doi:10.1016/S0004-3702(99)00052-1。


[18] Y. Zhu, J. Wong, A. Mandlekar, R. Martín-Martín, A. Joshi, S. Nasiriany, Y. Zhu, et al. robosuite: A modular simulation framework and benchmark for robot learning. arXiv preprint arXiv:2009.12293, 2020. URL https://arxiv.org/abs/2009.12293.Standard manipulation benchmark used in this paper.
[18] Y. Zhu, J. Wong, A. Mandlekar, R. Martín-Martín, A. Joshi, S. Nasiriany, Y. Zhu 等. robosuite：用于机器人学习的模块化仿真框架与基准。arXiv 预印本 arXiv:2009.12293，2020。URL https://arxiv.org/abs/2009.12293.本文使用的标准操作基准。


[19] T. Haarnoja, A. Zhou, K. Hartikainen, G. Tucker, S. Ha, J. Tan, V. Kumar, H. Zhu, A. Gupta, P. Abbeel, and S. Levine. Soft actor-critic algorithms and applications. arXiv preprint arXiv:1812.05905, 2018. URL https://arxiv.org/abs/1812.05905.
[19] T. Haarnoja, A. Zhou, K. Hartikainen, G. Tucker, S. Ha, J. Tan, V. Kumar, H. Zhu, A. Gupta, P. Abbeel, 和 S. Levine. 软演员-评价（Soft actor-critic）算法及其应用。arXiv 预印本 arXiv:1812.05905，2018。URL https://arxiv.org/abs/1812.05905。


[20] M. T. Ribeiro, T. Wu, C. Guestrin, and S. Singh. Beyond accuracy: Behavioral testing of NLP models with CheckList. In Annual Meeting of the Association for Computational Linguistics (ACL), 2020. URL https://arxiv.org/abs/2005.04118.Probe-as-test methodology that the atomic-quality probe most closely echoes; capability-targeted lightweight tests for behavior assessment.
[20] M. T. Ribeiro, T. Wu, C. Guestrin, 和 S. Singh. 不止准确率：用 CheckList 对 NLP 模型进行行为测试。载于计算语言学协会年会（ACL），2020。URL https://arxiv.org/abs/2005.04118.探针即测试的方法论是原子质量探针最接近的映照；面向能力的轻量级测试，用于评估行为。


[21] R. Agarwal, M. Schwarzer, P. S. Castro, A. C. Courville, and M. G. Bellemare. Deep reinforcement learning at the edge of the statistical precipice. In Advances in Neural Information Processing Systems (NeurIPS), 2021. URL https://arxiv.org/abs/2108.13264.
[21] R. Agarwal, M. Schwarzer, P. S. Castro, A. C. Courville, 和 M. G. Bellemare. 统计临界点的边缘进行深度强化学习。发表于神经信息处理系统进展（NeurIPS），2021。URL https://arxiv.org/abs/2108.13264.


[22] G. Bansal, B. Nushi, E. Kamar, D. S. Weld, W. S. Lasecki, and E. Horvitz. Updates in human-AI teams: Understanding and addressing the performance/compatibility tradeoff. In AAAI Conference on Artificial Intelligence, 2019. URL https://ojs.aaai.org/ index.php/AAAI/article/view/4087. Foundational ML paper on the better-model-breaks-downstream phenomenon; with Shen 2020 and Yan 2021 forms the standard backward-compatibility lineage.
[22] G. Bansal, B. Nushi, E. Kamar, D. S. Weld, W. S. Lasecki, 和 E. Horvitz. 人类-人工智能团队中的更新：理解并应对性能/兼容性权衡。发表于人工智能AAAI会议，2019。URL https://ojs.aaai.org/ index.php/AAAI/article/view/4087. 关于“更好的模型会瓦解下游”的现象的基础机器学习论文；与 Shen 2020 和 Yan 2021 一起，形成标准的向后兼容谱系。


[23] Y. Shen, Y. Xiong, W. Xia, and S. Soatto. Towards backward-compatible representation learning. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2020. URL https://arxiv.org/abs/2003.11942.Foundational representation-learning paper of the backward-compatibility / regression-on-update lineage; precedes Yan et al. (2021).
[23] Y. Shen, Y. Xiong, W. Xia, 和 S. Soatto. 朝向向后兼容的表征学习。发表于IEEE/CVF计算机视觉与模式识别会议（CVPR），2020。URL https://arxiv.org/abs/2003.11942. 研究向后兼容/在更新上的回归这一脉络的基础表征学习论文；位于 Yan 等（2021）之前。


[24] S. Yan, Y. Xiong, K. Kundu, S. Yang, S. Deng, M. Wang, W. Xia, and S. Soatto. Positive-congruent training: Towards regression-free model updates. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2021. URL https://arxiv.org/abs/2011.09161.Foundational formalization of negative flips when classifiers are updated; ML-side analogue of the composition-breaks-after-skill-update phenomenon.
[24] S. Yan, Y. Xiong, K. Kundu, S. Yang, S. Deng, M. Wang, W. Xia, 和 S. Soatto. 正一致训练：通向无需回归的模型更新。发表于IEEE/CVF计算机视觉与模式识别会议（CVPR），2021。URL https://arxiv.org/abs/2011.09161. 当分类器被更新时出现负向翻转的基础形式化工作；也是“技能更新后组合被破坏”现象在机器学习侧的对应表述。


[25] A. Brohan, N. Brown, J. Carbajal, Y. Chebotar, X. Chen, K. Choromanski, T. Ding, D. Driess, A. Dubey, C. Finn, et al. RT-2: Vision-language-action models transfer web knowledge to robotic control. In Conference on Robot Learning (CoRL), 2023. URL https://arxiv.org/abs/2307.15818.
[25] A. Brohan, N. Brown, J. Carbajal, Y. Chebotar, X. Chen, K. Choromanski, T. Ding, D. Driess, A. Dubey, C. Finn 等。RT-2：视觉-语言-动作模型将网页知识迁移到机器人控制。发表于机器人学习会议（CoRL），2023。URL https://arxiv.org/abs/2307.15818.


[26] Open X-Embodiment Collaboration, A. O'Neill, A. Rehman, A. Maddukuri, et al. Open X-embodiment: Robotic learning datasets and RT-X models. In ICRA, 2024. URL https://arxiv.org/abs/2310.08864.
[26] Open X-Embodiment 协作团队，A. O'Neill, A. Rehman, A. Maddukuri 等。Open X-Embodiment：机器人学习数据集与RT-X模型。发表于ICRA，2024。URL https://arxiv.org/abs/2310.08864.


[27] A. Khazatsky, K. Pertsch, S. Nair, A. Balakrishna, S. Dasari, S. Karamcheti, S. Nasiriany, M. K. Srirama, L. Y. Chen, K. Ellis, et al. DROID: A large-scale in-the-wild robot manipulation dataset. In Robotics: Science and Systems (RSS), 2024. URL https: //arxiv.org/abs/2403.12945.
[27] A. Khazatsky, K. Pertsch, S. Nair, A. Balakrishna, S. Dasari, S. Karamcheti, S. Nasiriany, M. K. Srirama, L. Y. Chen, K. Ellis 等。DROID：大规模的真实环境机器人操作数据集。发表于机器人：科学与系统（RSS），2024。URL https: //arxiv.org/abs/2403.12945.


[28] X. Li, K. Hsu, J. Gu, K. Pertsch, O. Mees, H. R. Walke, C. Fang, I. S. Wang, N. Yokoyama, D. Sadigh, S. Levine, J. Wu, C. Finn, et al. Evaluating real-world robot manipulation policies in simulation. In Conference on Robot Learning (CoRL), 2024. URL https://arxiv.org/abs/2405.05941.
[28] X. Li, K. Hsu, J. Gu, K. Pertsch, O. Mees, H. R. Walke, C. Fang, I. S. Wang, N. Yokoyama, D. Sadigh, S. Levine, J. Wu, C. Finn 等。在仿真中评估真实世界的机器人操作策略。发表于机器人学习会议（CoRL），2024。URL https://arxiv.org/abs/2405.05941.


[29] P. Atreya, K. Pertsch, T. Lee, M. J. Kim, A. Jain, et al. RoboArena: Distributed real-world evaluation of generalist robot policies. arXiv preprint arXiv:2506.18123, 2025. URL https://arxiv.org/abs/2506.18123.
[29] P. Atreya, K. Pertsch, T. Lee, M. J. Kim, A. Jain 等。RoboArena：面向通用机器人策略的分布式真实环境评估。arXiv预印本 arXiv:2506.18123，2025。URL https://arxiv.org/abs/2506.18123.


[30] C. Chi, Z. Xu, S. Feng, Y. Du, E. Cousineau, B. Burchfiel, R. Tedrake, and S. Song. Diffusion policy: Visuomotor policy learning via action diffusion. In Robotics: Science and Systems (RSS), 2023. URL https://arxiv.org/abs/2303.04137.
[30] C. Chi, Z. Xu, S. Feng, Y. Du, E. Cousineau, B. Burchfiel, R. Tedrake, 和 S. Song. 扩散策略：通过动作扩散学习视觉运动策略。发表于机器人：科学与系统（RSS），2023。URL https://arxiv.org/abs/2303.04137.


[31] T. Z. Zhao, V. Kumar, S. Levine, and C. Finn. Learning fine-grained bimanual manipulation with low-cost hardware. In Robotics: Science and Systems (RSS), 2023. URL https://arxiv.org/abs/2304.13705.ACT: Action Chunking Transformer; canonical bimanual imitation-learning baseline.
[31]T.Z.Zhao,V.Kumar,S.Levine和C.Finn.利用低成本硬件进行精细双手操作学习。发表于Robotics:Science and Systems(RSS),2023。URLhttps://arxiv.org/abs/2304.13705.ACT：动作块Transformer；经典的双手模仿学习基线。


[32] L. Keller, D. Tanneberg, and J. Peters. Neuro-symbolic imitation learning: Discovering symbolic abstractions for skill learning. arXiv preprint arXiv:2503.21406, 2025. URL https://arxiv.org/abs/2503.21406.Learns PDDL predicates + neural skills from demos; symbolic planning for abstract plans refined by neural skills.
[32]L.Keller,D.Tanneberg和J.Peters.神经符号模仿学习：为技能学习发现符号抽象。arXiv预印本arXiv:2503.21406,2025。URLhttps://arxiv.org/abs/2503.21406.从演示中学习PDDL谓词+神经技能；通过神经技能细化的符号规划来生成抽象计划。


[33] Y. Liang, N. Kumar, H. Tang, A. Weller, J. B. Tenenbaum, T. Silver, J. F. Henriques, and K. Ellis. VisualPredicator: Learning abstract world models with neuro-symbolic predicates for robot planning. arXiv preprint arXiv:2410.23156, 2024. URL https: //arxiv.org/abs/2410.23156. Neuro-symbolic predicates for abstract world model + planning.
[33]Y.Liang,N.Kumar,H.Tang,A.Weller,J.B.Tenenbaum,T.Silver,J.F.Henriques和K.Ellis.VisualPredicator：利用用于机器人规划的神经符号谓词学习抽象世界模型。arXiv预印本arXiv:2410.23156,2024。URLhttps://arxiv.org/abs/2410.23156。用于抽象世界模型+规划的神经符号谓词。


[34] Z. Chen, J. Yin, Y. Chen, J. Huo, P. Tian, J. Shi, Y. Hou, Y. Li, and Y. Gao. DeCo: Task decomposition and skill composition for zero-shot generalization in long-horizon 3d manipulation. arXiv preprint arXiv:2505.00527, 2025. URL https://arxiv.org/abs/ 2505.00527. CoRL 2025. Model-agnostic task decomposition + VLM-guided planning; +53.89% success on unseen compositional tasks.
[34]Z.Chen,J.Yin,Y.Chen,J.Huo,P.Tian,J.Shi,Y.Hou,Y.Li和Y.Gao.DeCo：面向长时程3D操作中零样本泛化的任务分解与技能组合。arXiv预印本arXiv:2505.00527,2025。URLhttps://arxiv.org/abs/2505.00527。CoRL 2025。模型无关的任务分解+VLM引导规划；在未见过的组合任务上成功率提升53.89%。


[35] K. Lin, C. Agia, T. Migimatsu, M. Pavone, and J. Bohg. Text2Motion: From natural language instructions to feasible plans. Autonomous Robots, 2023. URL https:// arxiv.org/abs/2303.12153. LLM planning + Q-function skill feasibility + geometric feasibility planning; 82% on 6+ skill long-horizon tasks.
[35]K.Lin,C.Agia,T.Migimatsu,M.Pavone和J.Bohg.Text2Motion：从自然语言指令到可行计划。Autonomous Robots,2023。URLhttps://arxiv.org/abs/2303.12153。LLM规划+Q函数技能可行性+几何可行性规划；在6+技能长时程任务上达到82%。


[36] K. Pertsch, Y. Lee, and J. J. Lim. Accelerating reinforcement learning with learned skill priors. In CoRL, 2020. URL https://arxiv.org/abs/2010.11944.SPiRL: learn skill embedding + prior from offline data; foundational skill-prior work.
[36]K.Pertsch,Y.Lee和J.J.Lim.利用学习到的技能先验加速强化学习。发表于CoRL,2020。URLhttps://arxiv.org/abs/2010.11944.SPiRL：从离线数据中学习技能嵌入+先验；基础性的技能先验工作。


[37] L. X. Shi, J. J. Lim, and Y. Lee. Skill-based model-based reinforcement learning. In CoRL, 2022. URL https://arxiv.org/abs/2207.07560.SkiMo: skill dynamics model + skill repertoire; 5x more sample efficient than SPiRL.
[37]L.X.Shi,J.J.Lim和Y.Lee.基于技能的模型驱动强化学习。发表于CoRL,2022。URLhttps://arxiv.org/abs/2207.07560.SkiMo：技能动力学模型+技能库；样本效率比SPiRL高5倍。


[38] Z. Feng, H. Luan, K. Y. Ma, and H. Soh. Diffusion meets options: Hierarchical generative skill composition for temporally-extended tasks. arXiv preprint arXiv:2410.02389, 2024. URL https://arxiv.org/abs/2410.02389.DOPPLER: LTL-specified planning + HRL + diffusion options; navigation and manipulation.
[38]Z.Feng,H.Luan,K.Y.Ma和H.Soh.扩散遇见option：用于时序扩展任务的层级生成式技能组合。arXiv预印本arXiv:2410.02389,2024。URLhttps://arxiv.org/abs/2410.02389.DOPPLER：LTL指定规划+HRL+扩散option；用于导航和操作。


[39] C. L. Shek and P. Tokekar. Option discovery using LLM-guided semantic hierarchical reinforcement learning. arXiv preprint arXiv:2503.19007, 2025. URL https://arxiv.org/abs/2503.19007.LDSC: LLM subgoal selection + option reuse; outperforms baselines by 55.9%.
[39]C.L.Shek和P.Tokekar.使用LLM引导的语义层级强化学习进行option发现。arXiv预印本arXiv:2503.19007,2025。URLhttps://arxiv.org/abs/2503.19007.LDSC：LLM子目标选择+option复用；优于基线55.9%。


[40] S. Cheng and D. Xu. LEAGUE: Guided skill learning and abstraction for long-horizon manipulation. IEEE Robotics and Automation Letters, 2023. URL https://arxiv.org/abs/2210.12631.
[40]S.Cheng和D.Xu.LEAGUE：用于长时程操作的引导式技能学习与抽象。IEEE Robotics and Automation Letters,2023。URLhttps://arxiv.org/abs/2210.12631。


[41] Y. Zhu, P. Stone, and Y. Zhu. Bottom-up skill discovery from unsegmented demonstrations for long-horizon robot manipulation. IEEE Robotics and Automation Letters, 2022.
[41]Y.Zhu,P.Stone和Y.Zhu.从未分段演示中自底向上发现长时程机器人操作技能。IEEE Robotics and Automation Letters,2022。


[42] Z. Chen, Z. Ji, J. Huo, and Y. Gao. SCaR: Refining skill chaining for long-horizon robotic manipulation via dual regularization. In Advances in Neural Information Processing Systems (NeurIPS), 2024.
[42]Z.Chen,Z.Ji,J.Huo和Y.Gao.SCaR：通过双重正则化精炼长时程机器人操作中的技能串联。发表于Advances in Neural Information Processing Systems(NeurIPS),2024。


[43] Y. Wang, Y. Zhang, M. Huo, R. Tian, X. Zhang, Y. Xie, C. Xu, P. Ji, W. Zhan, M. Ding, and M. Tomizuka. Sparse diffusion policy: A sparse, reusable, and flexible policy for robot learning. In Conference on Robot Learning (CoRL), 2024.
[43] Y. Wang, Y. Zhang, M. Huo, R. Tian, X. Zhang, Y. Xie, C. Xu, P. Ji, W. Zhan, M. Ding, and M. Tomizuka. 稀疏扩散策略：一种用于机器人学习的稀疏、可复用且灵活的策略。发表于机器人学习会议（CoRL），2024。


[44] J. A. Mendez, M. Hussing, M. Gummadi, and E. Eaton. CompoSuite: A compositional reinforcement learning benchmark. In CoLLAs, 2022. URL https://arxiv.org/abs/ 2207.04136. 256 tasks $=$ robot $\mathrm{x}$ obstacle $\mathrm{x}$ object $\mathrm{x}$ objective; canonical compositional RL benchmark.
[44] J. A. Mendez, M. Hussing, M. Gummadi, and E. Eaton. CompoSuite：一个组合式强化学习基准。发表于 CoLLAs，2022。URL https://arxiv.org/abs/ 2207.04136. 256 个任务 $=$ 机器人 $\mathrm{x}$ 障碍物 $\mathrm{x}$ 物体 $\mathrm{x}$ 目标；经典的组合式强化学习基准。


[45] B. Liu, Y. Zhu, C. Gao, Y. Feng, Q. Liu, Y. Zhu, and P. Stone. LIBERO: Benchmarking knowledge transfer for lifelong robot learning. In NeurIPS Datasets and Benchmarks, 2023. URL https://arxiv.org/abs/2306.03310.130 language-conditioned manipulation tasks; 4 suites including LIBERO-Long for skill chaining.
[45] B. Liu, Y. Zhu, C. Gao, Y. Feng, Q. Liu, Y. Zhu, and P. Stone. LIBERO：用于终身机器人学习的知识迁移基准测试。发表于 NeurIPS Datasets and Benchmarks，2023。URL https://arxiv.org/abs/2306.03310.130 个语言条件操控任务；4 个套件，包括用于技能串联的 LIBERO-Long。


[46] X. Zhou, Y. Xu, G. Tie, Y. Chen, G. Zhang, D. Chu, P. Zhou, and L. Sun. LIBERO-PRO: Towards robust and fair evaluation of vision-language-action models beyond memorization. arXiv preprint arXiv:2510.03827, 2025. URL https://arxiv.org/abs/2510.03827.Extended LIBERO eval across objects/init-states/instructions/environments; SOTA fails near-completely under perturbations.
[46] X. Zhou, Y. Xu, G. Tie, Y. Chen, G. Zhang, D. Chu, P. Zhou, and L. Sun. LIBERO-PRO：超越记忆化，迈向对视觉-语言-动作模型的稳健且公平评估。arXiv 预印本 arXiv:2510.03827, 2025。URL https://arxiv.org/abs/2510.03827.在对象/初始状态/指令/环境上扩展的 LIBERO 评测；即使在扰动下，SOTA 也几乎完全失效。


[47] S. Haresh, D. Dijkman, A. Bhattacharyya, and R. Memisevic. ClevrSkills: Compositional language and visual reasoning in robotics. In NeurIPS Datasets and Benchmarks Track, 2024. URL https://arxiv.org/abs/2411.09052.33 tasks over 3 compositional levels (L0/L1/L2) on ManiSkill2; even pretrained VLMs fail on L1/L2.
[47] S. Haresh, D. Dijkman, A. Bhattacharyya, and R. Memisevic. ClevrSkills：机器人中的组合式语言与视觉推理。发表于 NeurIPS Datasets and Benchmarks Track，2024。URL https://arxiv.org/abs/2411.09052.在 ManiSkill2 上包含 3 个组合层级（L0/L1/L2）的 33 个任务；即使预训练的 VLM 也在 L1/L2 上表现失败。


[48] O. Mees, L. Hermann, E. Rosete-Beas, and W. Burgard. CALVIN: A benchmark for language-conditioned policy learning for long-horizon robot manipulation tasks. IEEE Robotics and Automation Letters, 2022. URL https://arxiv.org/abs/2112.03227.Long-horizon language-conditioned benchmark; chains of up to 5 sub-goals.
[48] O. Mees, L. Hermann, E. Rosete-Beas, and W. Burgard. CALVIN：一个面向长时程机器人操作任务的语言条件策略学习基准。IEEE Robotics and Automation Letters，2022。URL https://arxiv.org/abs/2112.03227.长时程语言条件基准；最长可达 5 个子目标的链式任务。


[49] T. Lesort, V. Lomonaco, A. Stoian, D. Maltoni, D. Filliat, and N. Díaz-Rodríguez. Continual learning for robotics: Definition, framework, learning strategies, opportunities and challenges. Information Fusion, 58:52-68, 2020. doi:10.1016/j.inffus.2019.12.004. URL https://arxiv.org/abs/1907.00182.Authoritative survey of continual learning in robotics; field-anchor citation between EWC and continual fine-tuning of robot policies.
[49] T. Lesort, V. Lomonaco, A. Stoian, D. Maltoni, D. Filliat, and N. Díaz-Rodríguez. 机器人持续学习：定义、框架、学习策略、机遇与挑战。Information Fusion, 58:52-68, 2020。doi:10.1016/j.inffus.2019.12.004。URL https://arxiv.org/abs/1907.00182.机器人持续学习的权威综述；该领域的基准引用，位于 EWC 与机器人策略持续微调之间。


[50] J. Kirkpatrick, R. Pascanu, N. Rabinowitz, J. Veness, G. Desjardins, A. A. Rusu, K. Milan, J. Quan, T. Ramalho, A. Grabska-Barwinska, et al. Overcoming catastrophic forgetting in neural networks. Proceedings of the National Academy of Sciences (PNAS), 114(13):3521-3526, 2017. doi:10.1073/pnas.1611835114.
[50] J. Kirkpatrick, R. Pascanu, N. Rabinowitz, J. Veness, G. Desjardins, A. A. Rusu, K. Milan, J. Quan, T. Ramalho, A. Grabska-Barwinska, et al. 克服神经网络中的灾难性遗忘。Proceedings of the National Academy of Sciences (PNAS), 114(13):3521-3526, 2017。doi:10.1073/pnas.1611835114。


[51] G. M. van de Ven, T. Tuytelaars, and A. S. Tolias. Three types of incremental learning. Nature Machine Intelligence, 4:1185-1197, 2022. doi:10.1038/s42256-022-00568-3.
[51] G. M. van de Ven, T. Tuytelaars, and A. S. Tolias. 三种增量学习类型。Nature Machine Intelligence, 4:1185-1197, 2022。doi:10.1038/s42256-022-00568-3。


[52] Y. Ding, L. Liu, P. Wang, and L. Wang. Evaluating forgetting in pretrained robotic policy networks: A continual learning study with Octo. In DICTA, 2025.
[52] Y. Ding, L. Liu, P. Wang, and L. Wang. 评估预训练机器人策略网络中的遗忘：一项基于 Octo 的持续学习研究。发表于 DICTA，2025。


[53] M. Wortsman, G. Ilharco, S. Y. Gadre, R. Roelofs, R. Gontijo-Lopes, A. S. Morcos, H. Namkoong, A. Farhadi, Y. Carmon, S. Kornblith, and L. Schmidt. Model soups: Averaging weights of multiple fine-tuned models improves accuracy without increasing inference time. In International Conference on Machine Learning (ICML), 2022. URL https://arxiv.org/abs/2203.05482.
[53] M. Wortsman, G. Ilharco, S. Y. Gadre, R. Roelofs, R. Gontijo-Lopes, A. S. Morcos, H. Namkoong, A. Farhadi, Y. Carmon, S. Kornblith, 和 L. Schmidt. Model soups：对多个微调模型的权重求平均，在不增加推理时间的情况下提升准确率。发表于国际机器学习会议（ICML），2022。URL https://arxiv.org/abs/2203.05482。


[54] G. Ilharco, M. T. Ribeiro, M. Wortsman, S. Gururangan, L. Schmidt, H. Hajishirzi, and A. Farhadi. Editing models with task arithmetic. In International Conference on Learning Representations (ICLR), 2023. URL https://arxiv.org/abs/2212.04089.
[54] G. Ilharco, M. T. Ribeiro, M. Wortsman, S. Gururangan, L. Schmidt, H. Hajishirzi, 和 A. Farhadi. 使用任务算术编辑模型。发表于国际表征学习会议（ICLR），2023。URL https://arxiv.org/abs/2212.04089。


## A Hybrid Selector Pseudocode
## 混合选择器伪代码


Algorithm 1 Hybrid Skill-Update Selector
算法 1 混合技能更新选择器


---



Require: Old ECM ${c}_{p}$ ,candidate ${c}_{a}$ ,atomic probes $q\left( \cdot \right)$ ,margin $m$ ,tolerance $\tau$
需要：旧的 ECM ${c}_{p}$ ，候选 ${c}_{a}$ ，原子探针 $q\left( \cdot \right)$ ，边界 $m$ ，容差 $\tau$


&nbsp;&nbsp;&nbsp;&nbsp;${\Delta }_{\text{ atomic }} \leftarrow  q\left( {c}_{a}\right)  - q\left( {c}_{p}\right)$



&nbsp;&nbsp;&nbsp;&nbsp;if $\left| {\Delta }_{\text{ atomic }}\right|  \geq  m$ then
&nbsp;&nbsp;&nbsp;&nbsp;如果 $\left| {\Delta }_{\text{ atomic }}\right|  \geq  m$ 则


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;return ACCEPT if ${\Delta }_{\text{ atomic }} \geq   - \tau$ else REJECT 																				$\vartriangleright$ Trust atomic signal
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;若 ${\Delta }_{\text{ atomic }} \geq   - \tau$ 则接受，否则拒绝 																				$\vartriangleright$ 信任原子信号


&nbsp;&nbsp;&nbsp;&nbsp;else
&nbsp;&nbsp;&nbsp;&nbsp;否则


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;${\Delta }_{\text{ comp }} \leftarrow$ paired composition probe $\left( {N\text{ episodes }}\right)$
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;${\Delta }_{\text{ comp }} \leftarrow$ 配对组合探针 $\left( {N\text{ episodes }}\right)$


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;return ACCEPT if ${\Delta }_{\text{ comp }} \geq   - \tau$ else REJECT 																	$\vartriangleright$ Fall back to expensive probe
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;若 ${\Delta }_{\text{ comp }} \geq   - \tau$ 则接受，否则拒绝 																	$\vartriangleright$ 回退到昂贵探针


&nbsp;&nbsp;&nbsp;&nbsp;end if
&nbsp;&nbsp;&nbsp;&nbsp;结束如果


---



## B T6 Atomic Heatmap and Per-Cell CIs
## B T6 原子热力图与逐单元置信区间


<table><tr><td>Phase</td><td>seed=42</td><td>seed=7</td><td>seed=123</td><td>seed=2024</td></tr><tr><td>REACH</td><td>6.7 0.0, 16.7]</td><td>0.0 0.0, 0.0]</td><td>26.7 [10.0, 43.3]</td><td>86.7 [73.3, 96.7]</td></tr><tr><td>GRASP</td><td>3.3 0.0, 10.0]</td><td>10.0 0.0, 20.0</td><td>${0.0}\left\lbrack  \begin{array}{ll} {0.0}, & {0.0} \end{array}\right\rbrack$</td><td>0.0 [ 0.0, 0.0]</td></tr><tr><td>LIFT</td><td>13.3 [ 3.3, 26.7</td><td>0.0 0.0, 0.0]</td><td>${0.0}\left\lbrack  {{0.0},{0.0}}\right\rbrack$</td><td>0.0 [ 0.0, 0.0]</td></tr><tr><td>PLACE</td><td>23.3 [10.0, 40.0]</td><td>20.0 6.7, 36.7]</td><td>0.0 .00 [0.0, 0.0]</td><td>16.7 [ 3.3, 30.0]</td></tr></table>
<table><tbody><tr><td>阶段</td><td>seed=42</td><td>seed=7</td><td>seed=123</td><td>seed=2024</td></tr><tr><td>到达</td><td>6.7 0.0, 16.7]</td><td>0.0 0.0, 0.0]</td><td>26.7 [10.0, 43.3]</td><td>86.7 [73.3, 96.7]</td></tr><tr><td>抓取</td><td>3.3 0.0, 10.0]</td><td>10.0 0.0, 20.0</td><td>${0.0}\left\lbrack  \begin{array}{ll} {0.0}, & {0.0} \end{array}\right\rbrack$</td><td>0.0 [ 0.0, 0.0]</td></tr><tr><td>抬起</td><td>13.3 [ 3.3, 26.7</td><td>0.0 0.0, 0.0]</td><td>${0.0}\left\lbrack  {{0.0},{0.0}}\right\rbrack$</td><td>0.0 [ 0.0, 0.0]</td></tr><tr><td>放置</td><td>23.3 [10.0, 40.0]</td><td>20.0 6.7, 36.7]</td><td>0.0 .00 [0.0, 0.0]</td><td>16.7 [ 3.3, 30.0]</td></tr></tbody></table>


Table 7: Per-cell T6 atomic success rate (%) with bootstrap 95% CIs (B=5000, N=30). The CI of the dominant cell is $\left\lbrack  {{73.3},{96.7}}\right\rbrack$ ; the next-best cell on REACH (seed=123) has CI [10.0, 43.3], fully disjoint from the dominant cell.
表 7：按单元划分的 T6 原子成功率（%），其自助法 95% 置信区间（B=5000，N=30）。主导单元的置信区间为 $\left\lbrack  {{73.3},{96.7}}\right\rbrack$；在 REACH 上次优的单元（seed=123）置信区间为 [10.0, 43.3]，与主导单元完全不重叠。


<img src="https://cdn.noedgeai.com/bo_d8j3koilb0pc73bdlaig_14.jpg?x=543&y=205&w=701&h=520&r=0"/>



Figure 2: T6 atomic success rate per (seed, phase) ECM (visual companion to table 1). The green-bordered cell (seed=2024, REACH, 86.7%) is the unique dominant ECM.
图 2：按（seed，phase）划分的 T6 原子成功率的 ECM（表 1 的视觉配套）。绿色边框的单元（seed=2024，REACH，86.7%）是唯一的主导 ECM。


<table><tr><td></td><td colspan="2">swap=42</td><td></td><td>swap=7</td><td>swap=123</td><td>swap=2024</td></tr><tr><td>primary=42</td><td>13.3</td><td>3.3,26.7</td><td>0.0</td><td>0.0,0.0]</td><td>33.3 [16.7,50.0]</td><td>60.0 [43.3,76.7]</td></tr><tr><td>primary=7</td><td>40.0</td><td>23.3,56.7</td><td>16.7</td><td>3.3,30.0</td><td>46.7 [30.0,63.3]</td><td>76.7 [60.0,90.0]</td></tr><tr><td>primary=123</td><td>3.3</td><td>0.0,10.0</td><td>0.0</td><td>0.0, 0.0]</td><td>33.3 [16.7,50.0]</td><td>70.0 [53.3,86.7]</td></tr><tr><td>primary=2024</td><td>16.7</td><td>3.3,30.0</td><td>13.3</td><td>3.3,26.7</td><td>30.0 [13.3,46.7]</td><td>50.0 [33.3,66.7]</td></tr></table>
<table><tbody><tr><td></td><td colspan="2">swap=42</td><td></td><td>swap=7</td><td>swap=123</td><td>swap=2024</td></tr><tr><td>primary=42</td><td>13.3</td><td>3.3,26.7</td><td>0.0</td><td>0.0,0.0]</td><td>33.3 [16.7,50.0]</td><td>60.0 [43.3,76.7]</td></tr><tr><td>primary=7</td><td>40.0</td><td>23.3,56.7</td><td>16.7</td><td>3.3,30.0</td><td>46.7 [30.0,63.3]</td><td>76.7 [60.0,90.0]</td></tr><tr><td>primary=123</td><td>3.3</td><td>0.0,10.0</td><td>0.0</td><td>0.0, 0.0]</td><td>33.3 [16.7,50.0]</td><td>70.0 [53.3,86.7]</td></tr><tr><td>primary=2024</td><td>16.7</td><td>3.3,30.0</td><td>13.3</td><td>3.3,26.7</td><td>30.0 [13.3,46.7]</td><td>50.0 [33.3,66.7]</td></tr></tbody></table>


Table 8: Per-cell T6 REACH-phase paired-swap success rate with bootstrap 95% CIs $(N = {30}$ paired episodes per cell; companion to table 3).
表8：按单元格划分的 T6 REACH 阶段配对交换成功率及 bootstrap 95% 置信区间 $(N = {30}$ 每个单元格的配对 episode 数；表3的配套表）。


Independent re-run with state logging (robustness check). The mechanism-(b)/(c) follow-up rollout (section E) is an independent re-execution of the T6 REACH paired cross-seed matrix on the same paired episode-seed pool. The dominant pattern reproduces with comparable or stronger signal: swap $= {2024}$ off-diagonal column mean is ${73.3}\%$ in the re-run versus 64.2% in table 3,and swap=\{42,7\} off-diagonal means are 13.4% and 10.0% respectively (both within $\pm  6\mathrm{{pp}}$ of the original); the dominant cell at primary $= {2024}$ , swap $= {2024}$ is ${86.7}\%$ in both runs. The full re-run cell-level matrix is included in the released artifact bundle.
带状态日志的独立重跑（鲁棒性检查）。机制-(b)/(c) 后续 rollout（第E节）是对同一配对 episode-seed 池上的 T6 REACH 配对跨种子矩阵的独立重新执行。主导模式可复现，且信号相当或更强：交换 $= {2024}$ 的非对角列均值在重跑中为 ${73.3}\%$，而表3中为 64.2%，交换=\{42,7\} 的非对角均值分别为 13.4% 和 10.0%（均在原始值的 $\pm  6\mathrm{{pp}}$ 以内）；主对角 $= {2024}$、交换 $= {2024}$ 的主导单元在两次运行中均为 ${86.7}\%$。完整的重跑单元级矩阵已包含在发布的资源包中。


C Behavioral-Distance Heatmaps (Visual)
C 行为距离热图（可视化）


<table><tr><td>Task</td><td>REACH</td><td>GRASP</td><td>LIFT</td><td>PLACE</td></tr><tr><td>T6</td><td>3.40</td><td>3.00</td><td>3.07</td><td>3.51</td></tr><tr><td>T3</td><td>2.71</td><td>2.66</td><td>2.59</td><td>1.90</td></tr><tr><td>T4</td><td>1.51</td><td>1.06</td><td>1.81</td><td>0.94</td></tr></table>
<table><tbody><tr><td>任务</td><td>REACH</td><td>GRASP</td><td>LIFT</td><td>PLACE</td></tr><tr><td>T6</td><td>3.40</td><td>3.00</td><td>3.07</td><td>3.51</td></tr><tr><td>T3</td><td>2.71</td><td>2.66</td><td>2.59</td><td>1.90</td></tr><tr><td>T4</td><td>1.51</td><td>1.06</td><td>1.81</td><td>0.94</td></tr></tbody></table>


Table 9: Mean off-diagonal action ${L}^{2}$ distance per (task,phase). On T6,the dominant ECM (seed=2024 REACH) has pairwise distance to its siblings of 3.20-3.53, fully within the typical range of the other phases.
表9：每个（任务，阶段）的平均非对角动作${L}^{2}$距离。在T6中，占主导的ECM（seed=2024 REACH）与其同类的成对距离为3.20-3.53，完全落在其他阶段的典型范围内。


## D Negative-Control Paired Matrices on T6
## D T6上的负对照配对矩阵


The three T6 phases other than REACH lack a high-quality ECM: maximum atomic success rate is $\leq  {23.3}\%$ on each,and the median is $0\%$ (table 1). For each of these phases we report the full $4 \times  4$ paired swap matrix; column-mean spreads are an order of magnitude tighter
除REACH外，T6的其余三个阶段都缺少高质量ECM：每个阶段的最大原子成功率均为$\leq  {23.3}\%$，中位数为$0\%$（表1）。对于这些阶段中的每一个，我们报告完整的$4 \times  4$配对交换矩阵；列均值的离散度比REACH上的56.7个百分点离散度（表3）紧一个数量级，支持第4.4节中的论断，即主导技能效应需要真正高质量的ECM。


<img src="https://cdn.noedgeai.com/bo_d8j3koilb0pc73bdlaig_15.jpg?x=308&y=219&w=1178&h=818&r=0"/>



Figure 3: Pairwise action ${L}^{2}$ distance between $\left( {{\text{ seed }}_{i},{\text{ seed }}_{j}}\right)$ ECMs,per (task,phase). All twelve panels are visually uniform; the dominant ECM on T6 (row 1, column 1; seed=2024 REACH) is not behaviorally distant from its peers.
图3：按（任务，阶段）划分的${L}^{2}$个ECM之间的两两动作$\left( {{\text{ seed }}_{i},{\text{ seed }}_{j}}\right)$距离。全部十二个面板在视觉上都很一致；T6上的主导ECM（第1行，第1列；seed=2024 REACH）在行为上并不比其余ECM更远。


than the 56.7pp spread on REACH (table 3), supporting the claim in section 4.4 that the dominant-skill effect requires a true high-quality ECM.



<table><tr><td>GRASP swap</td><td>swap=42</td><td>swap=7</td><td>swap=123</td><td>swap=2024</td><td>col. mean</td></tr><tr><td>primary=42</td><td>20.0</td><td>10.0</td><td>13.3</td><td>33.3</td><td>-</td></tr><tr><td>primary=7</td><td>20.0</td><td>33.3</td><td>23.3</td><td>20.0</td><td>-</td></tr><tr><td>primary=123</td><td>20.0</td><td>30.0</td><td>36.7</td><td>30.0</td><td>-</td></tr><tr><td>primary=2024</td><td>66.7</td><td>80.0</td><td>83.3</td><td>63.3</td><td>-</td></tr><tr><td>column mean</td><td>31.7</td><td>38.3</td><td>39.2</td><td>36.7</td><td>spread 7.5pp</td></tr></table>
<table><tbody><tr><td>GRASP交换</td><td>swap=42</td><td>swap=7</td><td>swap=123</td><td>swap=2024</td><td>列均值</td></tr><tr><td>primary=42</td><td>20.0</td><td>10.0</td><td>13.3</td><td>33.3</td><td>-</td></tr><tr><td>primary=7</td><td>20.0</td><td>33.3</td><td>23.3</td><td>20.0</td><td>-</td></tr><tr><td>primary=123</td><td>20.0</td><td>30.0</td><td>36.7</td><td>30.0</td><td>-</td></tr><tr><td>primary=2024</td><td>66.7</td><td>80.0</td><td>83.3</td><td>63.3</td><td>-</td></tr><tr><td>列均值</td><td>31.7</td><td>38.3</td><td>39.2</td><td>36.7</td><td>差距7.5个百分点</td></tr></tbody></table>


Table 10: T6 GRASP-phase paired swap (success rate %); column spread 7.5pp.
表10：T6 抓取阶段成对交换（成功率 %）；列间离散度 7.5 个百分点。


<table><tr><td>LIFT swap</td><td>swap=42</td><td>swap=7</td><td>swap=123</td><td>swap=2024</td><td>col. mean</td></tr><tr><td>primary=42</td><td>13.3</td><td>13.3</td><td>20.0</td><td>26.7</td><td>-</td></tr><tr><td>primary=7</td><td>23.3</td><td>26.7</td><td>13.3</td><td>26.7</td><td>-</td></tr><tr><td>primary=123</td><td>16.7</td><td>30.0</td><td>46.7</td><td>46.7</td><td>-</td></tr><tr><td>primary=2024</td><td>76.7</td><td>83.3</td><td>76.7</td><td>70.0</td><td>-</td></tr><tr><td>column mean</td><td>32.5</td><td>38.3</td><td>39.2</td><td>42.5</td><td>spread 10.0pp</td></tr></table>
<table><tbody><tr><td>LIFT 交换</td><td>swap=42</td><td>swap=7</td><td>swap=123</td><td>swap=2024</td><td>列均值</td></tr><tr><td>primary=42</td><td>13.3</td><td>13.3</td><td>20.0</td><td>26.7</td><td>-</td></tr><tr><td>primary=7</td><td>23.3</td><td>26.7</td><td>13.3</td><td>26.7</td><td>-</td></tr><tr><td>primary=123</td><td>16.7</td><td>30.0</td><td>46.7</td><td>46.7</td><td>-</td></tr><tr><td>primary=2024</td><td>76.7</td><td>83.3</td><td>76.7</td><td>70.0</td><td>-</td></tr><tr><td>列均值</td><td>32.5</td><td>38.3</td><td>39.2</td><td>42.5</td><td>波动 10.0pp</td></tr></tbody></table>


Table 11: T6 LIFT-phase paired swap (success rate %); column spread 10.0pp.
表11：T6 LIFT阶段配对交换（成功率%）；列间跨度10.0个百分点。


<table><tr><td>PLACE swap</td><td>swap=42</td><td>swap=7</td><td>swap=123</td><td>swap=2024</td><td>col. mean</td></tr><tr><td>primary=42</td><td>20.0</td><td>23.3</td><td>33.3</td><td>16.7</td><td>-</td></tr><tr><td>primary=7</td><td>30.0</td><td>23.3</td><td>13.3</td><td>30.0</td><td>-</td></tr><tr><td>primary=123</td><td>33.3</td><td>20.0</td><td>43.3</td><td>30.0</td><td>-</td></tr><tr><td>primary=2024</td><td>66.7</td><td>80.0</td><td>53.3</td><td>56.7</td><td>-</td></tr><tr><td>column mean</td><td>37.5</td><td>36.7</td><td>35.8</td><td>33.3</td><td>spread 4.2pp</td></tr></table>
<table><tbody><tr><td>PLACE交换</td><td>交换=42</td><td>交换=7</td><td>交换=123</td><td>交换=2024</td><td>列均值</td></tr><tr><td>主=42</td><td>20.0</td><td>23.3</td><td>33.3</td><td>16.7</td><td>-</td></tr><tr><td>主=7</td><td>30.0</td><td>23.3</td><td>13.3</td><td>30.0</td><td>-</td></tr><tr><td>主=123</td><td>33.3</td><td>20.0</td><td>43.3</td><td>30.0</td><td>-</td></tr><tr><td>主=2024</td><td>66.7</td><td>80.0</td><td>53.3</td><td>56.7</td><td>-</td></tr><tr><td>列均值</td><td>37.5</td><td>36.7</td><td>35.8</td><td>33.3</td><td>扩散4.2个百分点</td></tr></tbody></table>


Table 12: T6 PLACE-phase paired swap (success rate %); column spread 4.2pp.
表 12：T6 PLACE 阶段配对置换（成功率 %）；列间离散度 4.2 个百分点。


## E Extended Discussion
## E 扩展讨论


This appendix expands section 8 with the alternative-mechanism breakdown and the connection to the typed-composition literature that we summarized only briefly in the main body.
本附录在第8节的基础上，补充了替代机制的拆解，并阐明了我们仅在正文中简要概述过的与类型化组合文献之间的联系。


Why atomic quality predicts composition, and the mean is zero. A natural prior is that a high-quality atomic ECM should be robust across a wider range of hand-off state distributions than its lower-quality siblings (a per-skill instantiation of the broader observation that updated functions can produce non-trivial downstream behavior shifts even at constant average quality [22, 23, 24]). The measurement in section E below contradicts the wider-coverage version of this prior on T6 REACH: the dominant ECM's phase-end state distribution is in fact narrower and more central than its siblings'. The robustness asymmetry must therefore live somewhere other than wider hand-off coverage - candidate refinements being action smoothness or trajectory-length distribution (left to future work). The robustness-asymmetry story nonetheless explains why the population-mean swap effect vanishes: within a candidate set, swapping toward the dominant ECM (when the primary lacks it) and away from it produce gains and losses of comparable magnitude; the signal lives in the conditional structure, not the marginal mean. A "naive" methodology that reports population-mean $\Delta$ under swap will systematically miss the phenomenon.
为什么原子质量能预测组合，以及均值为何为零。一个自然的先验是，高质量的原子ECM应当比低质量的同类在更广泛的交接状态分布上保持稳健性（这一点是对如下更一般观察的逐技能实例化：即使平均质量保持不变，更新后的函数也能产生不平凡的下游行为偏移 [22, 23, 24]）。下面的E节测量结果否定了这一“更广覆盖”版本的T6 REACH先验：占主导的ECM的相末状态分布实际上比其同类更窄、更居中。因此，这种稳健性不对称性必然存在于“交接覆盖范围之外”——有待进一步完善的候选包括动作平滑性或轨迹长度分布（留待未来工作）。尽管如此，稳健性不对称性的叙事仍能解释群体均值置换效应为何消失：在一个候选集合内，将策略朝向占主导的ECM（当主要ECM不具备时）以及远离它，都会带来幅度相当的收益与损失；信号存在于条件结构中，而非边际均值中。一种“朴素”的方法如果在置换下报告群体均值$\Delta$，将系统性地错过这一现象。


Alternative mechanisms. Several refinements of the robustness-asymmetry story are testable in the current data. (a) Hand-off state coverage: the dominant ECM may simply visit a wider region of phase-end states, so any downstream phase finds itself in-distribution. (b) Action smoothness: smoother action trajectories reduce contact discontinuities at phase transitions, an effect related to T-STAR's terminal-state regularization [12]. (c) Trajectory-length distribution: a dominant ECM may finish its phase faster (or slower) on average, leaving the downstream phase a larger time budget.
替代机制：目前的数据可检验对稳健性不对称叙事的若干改进。（a）交接状态覆盖：占主导的ECM可能只是在相末状态上访问了更广的区域，因此任何下游相都落在分布之内。（b）动作平滑性：更平滑的动作轨迹可减少相变时的接触不连续，这与T-STAR的终端状态正则化有关[12]。（c）轨迹长度分布：占主导的ECM可能在平均意义上更快（或更慢）地结束其阶段，从而给下游阶段留下更大的时间预算。


We measured (a) directly. For each of the four T6 REACH ECMs we collected the phase-end state vector (dim=218) over $N = {120}$ episodes (30 episodes $\times  4$ swap configurations) and computed pairwise Wasserstein-2 distance under a diagonal-Gaussian approximation, the ${\mathrm{L}}^{2}$ shift to the pooled centroid,and the sum of per-dimension variances ( $\sum$ diag Cov). The dominant ECM (seed=2024) is neither shifted further from the pooled centroid nor wider than its siblings:
我们直接测量了（a）。对T6 REACH的四个ECM，我们在$N = {120}$个回合（30个回合对应$\times  4$置换配置）上收集相末状态向量（维度=218），并在对角高斯近似下计算两两Wasserstein-2距离，计算${\mathrm{L}}^{2}$相对于汇聚质心的偏移量，以及逐维方差之和（ $\sum$ diag Cov）。占主导的ECM（seed=2024）既没有进一步偏离汇聚质心，也没有比其同类更宽：


<table><tr><td>seed</td><td>42</td><td>7</td><td>123</td><td>2024 (dominant)</td></tr><tr><td>$\sum$ diag Cov (width)</td><td>3443</td><td>3181</td><td>357</td><td>1635</td></tr><tr><td>${\begin{Vmatrix}{\mu }_{s} - \overline{\mu }\end{Vmatrix}}_{2}$ (shift)</td><td>7.10</td><td>5.81</td><td>5.55</td><td>4.70</td></tr></table>
<table><tbody><tr><td>种子</td><td>42</td><td>7</td><td>123</td><td>2024（主导）</td></tr><tr><td>$\sum$ 对角协方差（宽度）</td><td>3443</td><td>3181</td><td>357</td><td>1635</td></tr><tr><td>${\begin{Vmatrix}{\mu }_{s} - \overline{\mu }\end{Vmatrix}}_{2}$（平移）</td><td>7.10</td><td>5.81</td><td>5.55</td><td>4.70</td></tr></tbody></table>


The mean pairwise ${\mathrm{W}}^{2}$ between the dominant and the three siblings is 28.1,smaller than the sibling-to-sibling mean of 34.7; the dominant distribution is -23.6% less shifted from the pooled centroid and -29.7% narrower than the sibling mean, with greater per-dimension variance on only 4.6% of the 218 dimensions. Mechanism (a) is therefore not supported in
主导者与三个同级之间的平均成对${\mathrm{W}}^{2}$为28.1，低于同级彼此之间34.7的平均值；主导分布相较于合并质心的偏移幅度低23.6%，且比同级均值窄29.7%，仅在218个维度中的4.6%上具有更大的逐维方差。因此，机制(a)并未在


the present data: the dominant ECM's phase-end state distribution is, if anything, more concentrated and more central.
现有数据中得到支持：主导ECM的阶段末状态分布，反而更集中，也更居中。


We then measured (b) and (c) on a follow-up rollout of the same 16 configurations with per-step action and per-step state logging enabled (120 episodes per ECM, reach phase truncated at the framework boundary $T = {62}$ steps). Mechanism (b) is the per-episode mean step-to-step action change ${\left\langle  {\begin{Vmatrix}{\mathbf{a}}_{t} - {\mathbf{a}}_{t - 1}\end{Vmatrix}}_{2}\right\rangle  }_{t}$ ; mechanism (c) is the per-episode ${\mathrm{L}}^{2}$ path length through state space $\mathop{\sum }\limits_{t}{\begin{Vmatrix}{\mathbf{s}}_{t + 1} - {\mathbf{s}}_{t}\end{Vmatrix}}_{2}$ . Lower values on either metric indicate the proposed mechanism (smoother / more efficient).
随后，我们在同一16种配置的后续滚动实验中测量了(b)和(c)，并启用了逐步动作与逐步状态记录（每个ECM 120个回合，reach阶段在框架边界$T = {62}$步处截断）。机制(b)是每回合平均的步间动作变化${\left\langle  {\begin{Vmatrix}{\mathbf{a}}_{t} - {\mathbf{a}}_{t - 1}\end{Vmatrix}}_{2}\right\rangle  }_{t}$；机制(c)是每回合在状态空间中的${\mathrm{L}}^{2}$路径长度$\mathop{\sum }\limits_{t}{\begin{Vmatrix}{\mathbf{s}}_{t + 1} - {\mathbf{s}}_{t}\end{Vmatrix}}_{2}$。这两个指标越低，说明所提机制越符合预期（更平滑/更高效）。


<table><tr><td>seed</td><td>42</td><td>7</td><td>123</td><td>2024 (dominant)</td></tr><tr><td>atomic SR (%)</td><td>6.7</td><td>0.0</td><td>26.7</td><td>86.7</td></tr><tr><td>(b) action smoothness ( $\downarrow   =$ smoother)</td><td>2.27</td><td>2.98</td><td>0.76</td><td>1.48</td></tr><tr><td>(c) traj. length $( \downarrow   =$ shorter)</td><td>4567</td><td>4911</td><td>2787</td><td>3855</td></tr></table>
<table><tbody><tr><td>种子</td><td>42</td><td>7</td><td>123</td><td>2024（主导）</td></tr><tr><td>原子 SR（%）</td><td>6.7</td><td>0.0</td><td>26.7</td><td>86.7</td></tr><tr><td>（b）动作平滑度（$\downarrow   =$更平滑）</td><td>2.27</td><td>2.98</td><td>0.76</td><td>1.48</td></tr><tr><td>（c）轨迹长度（$( \downarrow   =$更短）</td><td>4567</td><td>4911</td><td>2787</td><td>3855</td></tr></tbody></table>


On both (b) and (c) the dominant ECM is not the extreme: it ranks 2 of 4 on each metric, with the lower-quality seed $= {123}$ (atomic SR 26.7%) holding the smoothest and shortest spots. Pairwise bootstrap 95% CIs confirm the dominant ECM is significantly smoother than seeds=42 and 7 (mean $\Delta  < 0$ ,CI excludes zero) but significantly rougher than seed=123 (mean $\Delta  > 0$ ,CI excludes zero); the same pattern holds for trajectory length. The conclusion matches (a): no single channel of (a) hand-off coverage, (b) action smoothness, or (c) trajectory length identifies the dominant ECM as an outlier on its own. The atomic-quality probe captures something the three behavioral channels do not separately recover - consistent with the robustness asymmetry operating through interaction or a more structural property; identifying that property is open future work.
在(b)和(c)中，主导ECM都不是极端值：它在每项指标上都排第2/4，而较低质量的种子$= {123}$（原子SR 26.7%）同时拥有最平滑且最短的点。成对bootstrap 95%置信区间证实，主导ECM比seeds=42和7显著更平滑（均值$\Delta  < 0$，CI不含0），但又比seed=123显著更粗糙（均值$\Delta  > 0$，CI不含0）；轨迹长度也呈现相同模式。结论与(a)一致：无论是(a)中的交接覆盖率、(b)动作平滑性，还是(c)轨迹长度，单一通道都不能单独将主导ECM识别为离群点。原子质量探针捕捉到了三条行为通道各自无法单独恢复的东西——这与通过交互起作用的鲁棒性不对称或某种更结构性的属性一致；识别该属性留待未来工作。


Connection to typed-composition literature and deployment. Our findings reinforce the typed-composition thread $\left\lbrack  {4,5,6}\right\rbrack$ that pre/post- condition structure is the right level at which to reason about compositional behavior. Where prior work stops at constructing such structure, we add a complementary observation: even with identical type signatures, two ECM versions can produce dramatically different composition outcomes, and the difference is captured by an atomic-quality probe rather than by any structural metric. Concretely for deployment: every candidate skill update should be probed atomically first; composition probes (far more expensive) should be invoked only when the atomic margin is insufficient.
与typed-composition文献及部署的联系。我们的发现强化了typed-composition这条脉络$\left\lbrack  {4,5,6}\right\rbrack$，即前/后置条件结构才是推理组合行为的合适层级。前人工作止步于构造这类结构，而我们补充了一点：即使类型签名完全相同，两个ECM版本也可能产生截然不同的组合结果，而这种差异由原子质量探针捕捉，而非任何结构指标。就部署而言，具体做法是：每个候选技能更新都应先进行原子探测；只有当原子裕量不足时，才应调用组合探测（其代价要高得多）。


## F Extended Related Work
## F 扩展相关工作


This appendix expands section 2 with four additional threads (generalist VLA policies, hierarchical RL and skill priors, compositional benchmarks, continual learning) that we kept brief in the main body for space.
本附录扩展了第2节，补充了四个额外脉络（通用 VLA 策略、分层 RL 与技能先验、组合基准、持续学习），这些内容因篇幅在正文中仅作简述。


Generalist VLA policies and the post-deployment update setting. Vision-language-action models such as OpenVLA [1],Octo [2], ${\pi }_{0}$ [3],and RT-2 [25],together with the Open X-Embodiment / RT-X collaboration's large heterogeneous datasets [26] and the DROID in-the-wild dataset [27], are explicitly designed for downstream fine-tuning, making post-deployment skill updates a routine event. Recent benchmarks evaluate such generalist policies in simulation [28] and in distributed real-world setups [29], but these evaluate policies as monolithic units rather than the post-update composition stability we target. Imitation-learning ECM architectures [30, 31] are also candidates for the present protocol but are out of scope here.
通用 VLA 策略与部署后更新设定。OpenVLA [1]、Octo [2]、${\pi }_{0}$ [3] 和 RT-2 [25] 等视觉-语言-动作模型，以及 Open X-Embodiment / RT-X 协作的大型异构数据集 [26] 和 DROID 野外数据集 [27]，都明确面向下游微调，使部署后技能更新成为常态。近期基准在仿真 [28] 和分布式真实世界设置 [29] 中评估这类通用策略，但它们评估的是作为整体单元的策略，而非我们关注的更新后组合稳定性。模仿学习 ECM 架构 [30, 31] 也可作为本协议的候选，但此处不在讨论范围内。


Neuro-symbolic and LLM-planned composition. A complementary thread to the typed-composition methods discussed in the main-body section 2 explicitly synthesizes the symbolic interface either neuro-symbolically or via an LLM planner. Neuro-Symbolic
神经符号与 LLM 规划的组合。与正文第2节讨论的类型化组合方法相辅相成的另一脉络，是以神经符号方式或通过 LLM 规划器显式合成符号接口。神经符号


Imitation Learning [32] discovers PDDL predicates from demonstrations and refines them with neural skills; VisualPredicator [33] learns neuro-symbolic predicates for an abstract world model used by a planner. DeCo [34] pairs LLM-driven task decomposition with skill composition for zero-shot long-horizon generalization, and Text2Motion [35] sequences skills through LLM planning gated by Q-function and geometric feasibility checks. All of these construct compositions assuming the constituent skills are fixed; none studies the compositional consequences of updating an underlying skill, which is the question we ask.
模仿学习 [32] 从演示中发现 PDDL 谓词，并用神经技能加以精炼；VisualPredicator [33] 学习用于规划器的抽象世界模型中的神经符号谓词。DeCo [34] 将 LLM 驱动的任务分解与技能组合结合，以实现零样本长时程泛化；Text2Motion [35] 则通过由 Q 函数和几何可行性检查门控的 LLM 规划来串联技能。上述方法都在假设组成技能固定的前提下构建组合；它们都未研究更新底层技能对组合产生的影响，而这正是我们提出的问题。


Hierarchical RL and skill priors. The methodological foundation of skill modules trained from offline data and reused for downstream tasks is established by SPiRL [36] and SkiMo [37]. DOPPLER [38] combines options with diffusion under linear-temporal-logic constraints; LDSC [39] uses LLM-guided semantic option discovery; LEAGUE [40] performs guided skill abstraction for long-horizon manipulation; bottom-up skill discovery from unsegmented demonstrations [41] is in a similar spirit. T-STAR [12] addresses the closely related problem of terminal-state mismatch between adjacent skills via terminal-state regularization at training time, while SCaR [42] regularizes skill chains via dual regularization. Sparse Diffusion Policy [43] targets continual updates in diffusion-policy ECMs without forgetting, the closest existing approach to our deployment scenario.
分层 RL 与技能先验。由离线数据训练并可复用于下游任务的技能模块这一方法基础，由 SPiRL [36] 和 SkiMo [37] 奠定。DOPPLER [38] 在线性时序逻辑约束下将 options 与扩散结合；LDSC [39] 使用 LLM 引导的语义 option 发现；LEAGUE [40] 对长时程操作进行引导式技能抽象；从未分段演示中自底向上的技能发现 [41] 也与此思路相近。T-STAR [12] 通过训练时的终态正则化，解决相邻技能之间终态不匹配的密切相关问题；SCaR [42] 则通过双重正则化对技能链进行正则化。Sparse Diffusion Policy [43] 面向 diffusion-policy ECM 的持续更新，在不遗忘的情况下进行适配，是与我们的部署场景最接近的现有方法。


Compositional benchmarks. CompoSuite [44] factorizes 256 tasks across robot/object/obstacle/objective. LIBERO [45] and its robustness extension LIBERO-PRO [46] probe language-conditioned policies across lifelong-learning suites. ClevrSkills [47] provides three explicit levels of compositional difficulty over ManiSkill2; CALVIN [48] provides language-conditioned chains of up to five sub-goals. Across all of these, the unit of generalization is a novel task or composition with the underlying skill set held fixed.
组合基准。CompoSuite [44] 在机器人/物体/障碍物/目标之间分解了 256 个任务。LIBERO [45] 及其鲁棒性扩展 LIBERO-PRO [46] 在终身学习套件中检验语言条件策略。ClevrSkills [47] 在 ManiSkill2 上提供了三个显式的组合难度层级；CALVIN [48] 则提供了最多包含五个子目标的语言条件链。对这些基准而言，泛化的单位都是一个新任务或一种新组合，而底层技能集合保持固定。


Continual learning of skills and policies. The continual-learning-for-robotics field is surveyed by Lesort et al. [49], situating skill-update governance between the catastrophic-forgetting tradition (elastic-weight consolidation, Kirkpatrick et al. 50) and the incremental-learning taxonomy [51]: our cross-version- swap protocol corresponds most closely to task-incremental learning where the task identity (the phase) is fixed but the underlying function (the ECM) is replaced. Recent empirical work directly measures forgetting in pretrained robot policies under continual fine-tuning of Octo [52], the closest empirical sibling to our study at the level of single-policy update. An adjacent line of work treats updates not as replacement but as weight-space merging: Model Soups [53] averages weights of fine-tuned variants, while Task Arithmetic [54] edits models via additive task vectors. These approaches keep the library implicitly versioned in weight space; our protocol applies symmetrically to either replacement or merging-based updates, since both ultimately produce a new ECM whose composition stability is the question of interest. Our work is complementary to all of the above: we study the effect of single-skill update on compositions that depend on it, rather than on the single-policy outputs themselves.
技能与策略的持续学习。Lesort 等人 [49] 对机器人持续学习领域进行了综述，将技能更新治理置于灾难性遗忘传统（弹性权重巩固，Kirkpatrick 等人 50）与增量学习分类法 [51] 之间：我们的跨版本交换协议最接近任务增量学习，其中任务身份（阶段）固定，但底层函数（ECM）被替换。近期的实证工作直接测量了在对 Octo [52] 进行持续微调时预训练机器人策略中的遗忘，这是在单一策略更新层面上与我们研究最接近的实证工作。与之相邻的一条研究线并不把更新视为替换，而是视为权重空间合并：Model Soups [53] 对微调变体的权重求平均，Task Arithmetic [54] 则通过加性任务向量编辑模型。这些方法在权重空间中对库进行隐式版本化；我们的协议对替换式或基于合并的更新都同样适用，因为二者最终都会生成一个新的 ECM，而其组合稳定性正是我们关注的问题。我们的工作与上述所有研究互为补充：我们研究的是单一技能更新对依赖它的组合的影响，而不是单策略输出本身。


Statistical reporting. Our paired-sampling protocol with bootstrap 95% CIs follows recommendations from Agarwal et al. [21] for sparse-trial RL benchmarks. The McNemar exact-binomial test in section 7 is supplemented by a cluster-permutation variant that respects the ECM-level dependence structure of the 48 update events.
统计报告。我们采用带 bootstrap 95% 置信区间的配对抽样协议，遵循 Agarwal 等人 [21] 对稀疏试验 RL 基准的建议。第7节中的 McNemar 精确二项检验还补充了一个聚类置换变体，以尊重 48 次更新事件在 ECM 层面的依赖结构。


Off-policy policy selection (extended). The connection of our atomic-quality probe to the wider OPE/Active-OPS literature, summarized in section 2, deserves further unpacking. Citation-network analysis of the typed-composition literature (BLADE, GSC, T-STAR) and the OPE literature (DOPE [15], Active OPS [16]) finds that, while the two share foundational RL classics (PPO, SAC, options), they share no substantive methodological references and no shared post-2021 citing papers. The Hybrid Selector is structurally an active offline-policy-selection procedure in the single-policy regime [16], lifted to the compositional regime where the policy unit is a chain of phase ECMs rather than a single neural network. Bringing OPE rigor into typed compositional skill libraries is, to our knowledge, an unaddressed gap that the present work closes for the skill-update governance setting.
离线策略选择（扩展）。我们对原子质量探针与更广泛的 OPE/Active-OPS 文献之间的联系，在第 2 节已作概述，但仍值得进一步展开。对类型化复合文献（BLADE、GSC、T-STAR）与 OPE 文献（DOPE [15]、Active OPS [16]）的引文网络分析发现：尽管两者都共享基础的强化学习经典（PPO、SAC、options），但在实质性的研究方法引用上互不重合，也没有共享的 2021 年之后引用论文。Hybrid Selector 在单策略设定下在结构上是一种主动的离线策略选择程序 [16]，而在复合设定中，它被提升为：策略单元是一串相位 ECM，而非单个神经网络。将 OPE 的严谨性带入类型化的复合技能库——据我们所知——是一个尚未被解决的空白；本工作在技能更新治理这一场景下予以填补。


## G Compute and Reproducibility
## G 计算与可复现性


ECMs are trained on a single NVIDIA RTX 5090 GPU under the framework's standard SAC schedule: ${50}\mathrm{\;K}$ environment steps $\times  {20}$ iterations per phase on T3/T4/T6,and ${50}\mathrm{\;K}$ steps $\times  {15}$ iterations on T1. Full hyperparameters are in the released configs/default.yaml. Per-seed wall-time ranges from ${2.9}\mathrm{\;h}\left( {\mathrm{\;T}1}\right)$ to ${6.3}\mathrm{\;h}\left( {\mathrm{\;T}3/\mathrm{T}4}\right)$ ; the four-seed multi-task suite (T1, T3, T4 each $\times  4$ seeds) totals approximately $\sim  {60}$ GPU-hours,plus pre-existing T6 checkpoints at four seeds inherited from prior work ( $\sim  {24}$ GPU-hours each, $\sim  {96}$ h total).
ECMs 在框架的标准 SAC 训练日程下于单个 NVIDIA RTX 5090 GPU 上训练：每阶段 ${50}\mathrm{\;K}$ 个环境步、$\times  {20}$ 次迭代（在 T3/T4/T6 上），以及在 T1 上 ${50}\mathrm{\;K}$ 步、$\times  {15}$ 次迭代。全部超参数见已发布的配置文件 configs/default.yaml。每个种子的总墙钟时间范围为 ${2.9}\mathrm{\;h}\left( {\mathrm{\;T}1}\right)$ 至 ${6.3}\mathrm{\;h}\left( {\mathrm{\;T}3/\mathrm{T}4}\right)$；四种种子的多任务套件（T1、T3、T4 各 $\times  4$ 个种子）合计约 $\sim  {60}$ GPU 小时。此外，来自先前工作的四种种子预存 T6 检查点（各 $\sim  {24}$ GPU 小时，$\sim  {96}$ 小时总计）。


All evaluations use $N = {30}$ episodes per cell,with paired initial-state seeds in the range [10000, 10029] shared across every swap matrix; this paired structure enables the McNemar / paired- $t$ tests reported throughout. Random seeds $\{ {42},7,{123},{2024}\}$ are used for ECM training. Code, evaluation data, and figure-regeneration scripts will be released upon acceptance.
所有评估均使用每个 cell 的 $N = {30}$ 个回合，并在每个置换矩阵中共享一对初始状态种子，取值范围 [10000, 10029]；这种成对结构用于贯穿全文报告的 McNemar / 成对-$t$ 检验。ECM 训练使用随机种子 $\{ {42},7,{123},{2024}\}$。代码、评估数据与图再生脚本将在接收后发布。


## H T3 Scaling Attempts (Negative Result)
## H T3扩展尝试（负面结果）


We attempted T3_Stack scaling along two arms before falling back to the deep-T6 framing. The longer-schedule arm trained for 26 iterations on T3_Stack seed=2024 under the default reward and produced no success-rate transients in any iteration, indicating that schedule extension alone is not the bottleneck. The reward-shaping arm modified the environment's reward function $\left( {{r}_{\text{ lift }}\text{ base }1 \rightarrow  2}\right.$ ,alignment bonus ${0.5} \rightarrow  1,{r}_{\text{ stack }}2 \rightarrow  4)$ and trained seed $= {42}$ for 30 iterations; this arm produced 8 success-rate transients all at 1/30 (3.33%) with reward sustained in $\left\lbrack  {4,7}\right\rbrack$ versus the default arm’s $\left\lbrack  {3,5}\right\rbrack$ ,demonstrating that reward shaping does move atomic learning, but not enough to clear the 40%-80% Goldilocks zone our paired cross-seed swap matrix requires. T3's sub-Goldilocks ceiling is therefore robust across both reward design and schedule extension.
我们在回退到深T6框架之前，沿两条路径尝试了T3_Stack扩展。更长训练周期路径在默认奖励下以seed=2024对T3_Stack训练了26次迭代，但在任何迭代中都未产生成功率瞬态，表明仅延长训练周期并不是瓶颈。奖励塑形路径修改了环境的奖励函数$\left( {{r}_{\text{ lift }}\text{ base }1 \rightarrow  2}\right.$，alignment bonus ${0.5} \rightarrow  1,{r}_{\text{ stack }}2 \rightarrow  4)$，并以seed$= {42}$训练了30次迭代；该路径共产生8次成功率瞬态，均为1/30（3.33%），且在$\left\lbrack  {4,7}\right\rbrack$中的奖励持续存在，而默认路径为$\left\lbrack  {3,5}\right\rbrack$，这表明奖励塑形确实会推动原子学习，但仍不足以突破我们的配对跨seed交换矩阵所要求的40%-80% Goldilocks区间。因此，T3低于Goldilocks上限的现象在奖励设计和训练周期延长两方面都同样稳健。